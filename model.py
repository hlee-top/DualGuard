import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import argparse
from torch.optim import lr_scheduler
import torch.nn.functional as F
from utils import set_seed, read_jsonl


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc(x)
        out = self.relu(out)
        out = out + x
        return out


class TransformModel(nn.Module):
    def __init__(self, num_layers=4, input_dim=1024, hidden_dim=500, output_dim=300):
        super(TransformModel, self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.layers.append(ResidualBlock(hidden_dim))

        self.pos_layer = nn.Linear(hidden_dim, output_dim)
        self.neg_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        pos_output = self.pos_layer(x)
        neg_output = self.neg_layer(x)

        return pos_output, neg_output


class VectorDataset(Dataset):
    def __init__(self, vectors):
        self.vectors = vectors

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        if self.vectors[idx]["label"] == "negative":
            label = 1
        else:
            label = 0
        return torch.tensor(self.vectors[idx]["emb"]).squeeze(0), torch.tensor([label])


def cosine_similarity_matrix(batch):
    norm = torch.norm(batch, dim=1).view(-1, 1)
    normed_batch = batch / norm
    similarity = torch.mm(normed_batch, normed_batch.t())
    return similarity


def cosine_similarity(x, y):
    dot_product = torch.sum(x * y, dim=-1)
    norm_x = torch.norm(x, p=2, dim=-1)
    norm_y = torch.norm(y, p=2, dim=-1)
    return dot_product / (norm_x * norm_y)


def row_col_mean_penalty(output):
    row_mean_penalty = torch.mean(output, dim=1).pow(2).sum()
    col_mean_penalty = torch.mean(output, dim=0).pow(2).sum()
    return row_mean_penalty + col_mean_penalty


def abs_value_penalty(output):
    penalties = torch.relu(0.05 - torch.abs(output))
    mask = (penalties > 0).float()  # Create a mask where penalties are non-zero
    non_zero_count = torch.max(mask.sum(), torch.tensor(1.0))  # Avoid division by zero
    return (penalties * mask).sum() / non_zero_count


def get_median_value_of_similarity(all_token_embedding):
    similarity = cosine_similarity_matrix(all_token_embedding)
    median_value = torch.median(similarity)
    mean_value = torch.mean(similarity)
    return mean_value


def sim_loss(input_a, input_b, output_a, output_b, median_value):
    original_similarity = cosine_similarity(input_a, input_b)

    original_similarity = torch.tanh(20 * (original_similarity - median_value))
    transformed_similarity = cosine_similarity(output_a, output_b)
    loss = torch.abs(original_similarity - transformed_similarity).mean()
    return loss


def loss_fn(input_emb_a, output_emb_a_pos, output_emb_a_neg, input_emb_b, output_emb_b_pos, output_emb_b_neg,
            median_value, label):
    sim_loss_a1 = sim_loss(input_emb_a, input_emb_b, output_emb_a_pos, output_emb_b_pos, median_value)
    mean_penalty_a = row_col_mean_penalty(output_emb_a_pos) + row_col_mean_penalty(output_emb_b_pos)
    range_penalty_a = abs_value_penalty(output_emb_a_pos) + abs_value_penalty(output_emb_b_pos)
    loss_a = sim_loss_a1 + mean_penalty_a + range_penalty_a

    sim_loss_b1 = sim_loss(input_emb_a, input_emb_b, output_emb_a_neg, output_emb_b_neg, median_value)
    mean_penalty_b = row_col_mean_penalty(output_emb_a_neg) + row_col_mean_penalty(output_emb_b_neg)
    range_penalty_b = abs_value_penalty(output_emb_a_neg) + abs_value_penalty(output_emb_b_neg)
    loss_b = sim_loss_b1 + mean_penalty_b + range_penalty_b

    if label == "positive":
        dis_loss = 1 - cosine_similarity(output_emb_a_pos, output_emb_a_neg).mean() + 1 - cosine_similarity(
            output_emb_b_pos, output_emb_b_neg).mean()
    else:
        dis_loss = F.relu(cosine_similarity(output_emb_a_pos, output_emb_a_neg) + 0.9).mean() + F.relu(
            cosine_similarity(output_emb_b_pos, output_emb_b_neg) + 0.9).mean()
    all_loss = loss_a + loss_b + dis_loss
    return all_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, help="random seed.")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_model_name", type=str)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument('--emb_model', type=str, default="cbert")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lambda3", type=float, default=1)
    parser.add_argument("--input_dim", type=int, default=1024)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_data = read_jsonl(args.data_path)
    pos_data, neg_data = [], []
    for da in all_data:
        if da["label"] == "negative":
            neg_data.append(da)
        else:
            pos_data.append(da)

    if len(neg_data) > len(pos_data):
        neg_data = neg_data[:len(pos_data)]
    else:
        pos_data = pos_data[:len(neg_data)]

    pos_emb = []
    for da in pos_data:
        pos_emb.append(da["emb"])
    pos_emb = torch.tensor(pos_emb).squeeze(1).to(device)
    pos_median_value = get_median_value_of_similarity(pos_emb)

    neg_emb = []
    for da in neg_data:
        neg_emb.append(da["emb"])
    neg_emb = torch.tensor(neg_emb).squeeze(1).to(device)
    neg_median_value = get_median_value_of_similarity(neg_emb)

    pos_dataset = VectorDataset(pos_data)
    pos_dataloader = DataLoader(pos_dataset, batch_size=args.batch_size, shuffle=True)

    neg_dataset = VectorDataset(neg_data)
    neg_dataloader = DataLoader(neg_dataset, batch_size=args.batch_size, shuffle=True)

    args.model_save_path = "model/{}.pth".format(args.output_model_name)

    model = TransformModel(input_dim=args.input_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.2)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

    epochs = args.epochs
    for epoch in range(epochs):
        pos_batch_iterator = iter(pos_dataloader)
        neg_batch_iterator = iter(neg_dataloader)
        pos_loss, neg_loss = 0, 0
        for _ in range(len(pos_dataloader) // 2):
            input_emb_a, input_label_a = next(pos_batch_iterator)
            input_emb_b, input_label_b = next(pos_batch_iterator)

            input_emb_a = input_emb_a.to(device)
            input_label_a = input_label_a.to(device)
            input_emb_b = input_emb_b.to(device)
            input_label_b = input_label_b.to(device)
            if input_emb_a.shape[0] != input_emb_b.shape[0]:
                continue
            output_emb_a_pos, output_emb_a_neg = model(input_emb_a)
            output_emb_b_pos, output_emb_b_neg = model(input_emb_b)

            loss = loss_fn(input_emb_a, output_emb_a_pos, output_emb_a_neg, input_emb_b, output_emb_b_pos,
                           output_emb_b_neg, median_value=pos_median_value, label="positive")
            pos_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for _ in range(len(neg_dataloader) // 2):
            input_emb_a, input_label_a = next(neg_batch_iterator)
            input_emb_b, input_label_b = next(neg_batch_iterator)

            input_emb_a = input_emb_a.to(device)
            input_label_a = input_label_a.to(device)
            input_emb_b = input_emb_b.to(device)
            input_label_b = input_label_b.to(device)
            if input_emb_a.shape[0] != input_emb_b.shape[0]:
                continue
            output_emb_a_pos, output_emb_a_neg = model(input_emb_a)
            output_emb_b_pos, output_emb_b_neg = model(input_emb_b)
            loss = loss_fn(input_emb_a, output_emb_a_pos, output_emb_a_neg, input_emb_b, output_emb_b_pos,
                           output_emb_b_neg, median_value=neg_median_value, label="negative")
            neg_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {neg_loss}")

    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    torch.save(model.state_dict(), args.model_save_path)
    print("save model", args.model_save_path)
