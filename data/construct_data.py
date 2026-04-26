import json
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import jieba


def get_c4_data(data_num):
    bar = tqdm(total=data_num, desc=f"Processing en")
    ds = load_dataset("allenai/c4", "realnewslike", streaming=True, split="validation")
    prompts = []
    responses = []
    for s in ds:
        text = s["text"]
        tokens = jieba.cut(text)
        tokens = list(tokens)
        if len(tokens) >= 200:
            tokens = tokens[:200]
            split_index = int(len(tokens) * 0.1)
            front_tokens = tokens[:split_index]
            back_tokens = tokens[split_index:]

            prompt = "".join(front_tokens)
            response = "".join(back_tokens)

            prompts.append(prompt)
            responses.append(response)
            bar.update(1)
            if len(prompts) == data_num:
                break
    print("data num", len(prompts))
    with open(f"c4_val.jsonl", "w", encoding='utf-8') as f:
        for p, r in zip(prompts, responses):
            f.write(json.dumps({"prompt": p, "response": r}, ensure_ascii=False) + "\n")


def get_booksum_data(data_num):
    bar = tqdm(total=data_num, desc=f"Processing en")
    df = pd.read_csv('booksum/dev.csv')

    prompts = []
    responses = []
    for index, row in df.iterrows():
        text = row["summary_text"]
        print("text", text)
        tokens = jieba.cut(text)
        tokens = list(tokens)
        if len(tokens) >= 200:
            tokens = tokens[:200]
            split_index = int(len(tokens) * 0.1)
            front_tokens = tokens[:split_index]
            back_tokens = tokens[split_index:]

            prompt = "".join(front_tokens)
            response = "".join(back_tokens)

            prompts.append(prompt)
            responses.append(response)
            bar.update(1)
            if len(prompts) == data_num:
                break

    with open(f"booksum_val.jsonl", "w", encoding='utf-8') as f:
        for p, r in zip(prompts, responses):
            f.write(json.dumps({"prompt": p, "response": r}, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    get_c4_data(200)
    get_booksum_data(200)
