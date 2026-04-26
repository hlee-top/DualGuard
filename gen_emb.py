import json
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
from utils import EmbeddingModel, write_jsonl


class SentenceEmbeddings:
    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emb_model = EmbeddingModel(model_name, self.device)

        model_path = "model/sentiment-roberta-large-english/"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda")

    def get_embedding(self, sentence):
        """Generate embedding for a sentence."""
        return self.emb_model.get_embedding(sentence).cpu().numpy()

    def generate_embeddings(self, input_path, output_path, size):
        """Generate embeddings for all sentences in the input file."""

        with open("data/sts/train.jsonl", 'r') as f:
            lines = f.readlines()
        all_embeddings = []
        text_list = []
        for line in lines:
            data = json.loads(line)
            for key in ["sentence1", "sentence2"]:
                text = data[key]  # text
                emb = self.get_embedding(text).tolist()
                inputs = self.tokenizer(text, max_length=512, truncation=True, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    toxic_score = self.classifier(**inputs).logits
                    toxic_score = F.softmax(toxic_score, dim=1)[0, 0].item()
                if toxic_score > 0.5:
                    label = "negative"
                else:
                    label = "positive"
                if text not in text_list:
                    print("text", text)
                    print("label", label)
                    text_list.append(text)
                    all_embeddings.append({"emb": emb, "label": label})
        print("len(all_embeddings):", len(all_embeddings))
        write_jsonl(output_path, all_embeddings)


def main():
    parser = argparse.ArgumentParser(description='Generate embeddings for sentences.')
    parser.add_argument('--input_path', type=str, help='Input file path')
    parser.add_argument('--output_path', type=str, required=True, help='Output file path')
    parser.add_argument('--emb_model', type=str, default="cbert", choices=["bert", "sbert", "cbert", "mbert", "jina3"])
    parser.add_argument('--size', type=int, required=False, default=1000,
                        help='Size of the data to generate embeddings for')
    args = parser.parse_args()

    sentence_embeddings = SentenceEmbeddings(args.emb_model)
    sentence_embeddings.generate_embeddings(args.input_path, args.output_path, args.size)


if __name__ == '__main__':
    main()
