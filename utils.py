import json
import numpy as np
from numpy.linalg import norm
import random
import time
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BertTokenizer, BertModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import scipy

llm_path = {
    "llama3.1_8b_instruct": "model/Meta-Llama-3.1-8B-Instruct",
    "opt1.3b": "model/opt-1.3b",
    "cbert": "model/compositional-bert-large-uncased/",
}

vocab_size_dict = {
    "llama3.1_8b_instruct": 128256,
    "opt1.3b": 50272,
    "llama3_70B": 128256,
}


def cosine_similarity(x, y):
    dot_product = torch.sum(x * y, dim=-1)
    norm_x = torch.norm(x, p=2, dim=-1)
    norm_y = torch.norm(y, p=2, dim=-1)
    return dot_product / (norm_x * norm_y)


def load_config_file(path: str) -> dict:
    """Load a JSON configuration file from the specified path and return it as a dictionary."""
    try:
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return config_dict

    except FileNotFoundError:
        print(f"Error: The file '{path}' does not exist.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in '{path}': {e}")
        # Handle other potential JSON decoding errors here
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Handle other unexpected errors here
        return None


class TransformersConfig:
    """Configuration class for transformers."""

    def __init__(self, model, tokenizer, vocab_size=None, device='cuda', *args, **kwargs):
        """
            Initialize the transformers configuration.

            Parameters:
                model (object): The model object.
                tokenizer (object): The tokenizer object.
                vocab_size (int): The vocabulary size.
                device (str): The device to use.
                kwargs: Additional keyword arguments.
        """
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer) if vocab_size is None else vocab_size
        self.gen_kwargs = {}
        self.gen_kwargs.update(kwargs)


def vocabulary_mapping(vocab_size, model_output_dim):
    return [random.randint(0, model_output_dim - 1) for _ in range(vocab_size)]


def read_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def write_jsonl(file_path, all_data):
    with open(file_path, "w", encoding="utf-8") as f:
        for data in all_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    print("sava file: {}".format(file_path))


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.read())


def save_json(save_path, data):
    with open(save_path, "w") as f:
        f.write(json.dumps(data))
    print("save file: {}".format(save_path))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_current_time():
    return time.strftime('%m%d_%H%M', time.localtime())


def get_all_role(event2argument):
    all_role_list = []
    for event_type in event2argument.keys():
        role_list = event2argument[event_type]["role_list"]
        for role in role_list:
            if role not in all_role_list:
                all_role_list.append(role)
    return all_role_list


def computer_similarity(vec1, vec2):
    cos_sim = (vec1 @ vec2.T) / (norm(vec1) * norm(vec2))
    return cos_sim


class EmbeddingModel:
    def __init__(self, model_name, device):
        self.device = device
        self.model_name = model_name
        model_path = llm_path[model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)

    def get_embedding(self, sentence_list):
        if isinstance(sentence_list, list):
            input_ids = self.tokenizer(sentence_list, padding='max_length', truncation=True, max_length=4,
                                       return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model(**input_ids)
        else:
            input_ids = self.tokenizer.encode(sentence_list, max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model(input_ids)
        return output[0][:, 0, :]


def flat_list(input_list):
    input_list = iter(input_list)
    convert_list = []
    while 1:
        try:
            convert_list += str(input_list.__next__())
        except StopIteration:
            break
    return convert_list


def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0


def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def get_topk_idx(data, k):
    data = np.array(data)
    idx = data.argsort()[-k:][::-1]
    return list(idx)


def _compute_p_value(z_score):
    p_value = scipy.stats.norm.sf(z_score)
    return p_value


def get_classifier(model_name):
    if model_name == "roberta":
        model_path = "model/sentiment-roberta-large-english"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda")
    return tokenizer, model


def get_toxicity_score(model_name, tokenizer, model, text):
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model(**inputs).logits
        output = F.softmax(output, dim=1)
    return output[0, 0].item()
