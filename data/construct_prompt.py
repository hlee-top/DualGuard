import jieba
import pandas as pd
import sys

sys.path.append('..')
from utils import read_jsonl, write_jsonl


def get_rtp_data(data_num=500):
    all_data = read_jsonl("real-toxicity-prompts/prompts.jsonl")
    return_data = []
    pos_num, neg_num = 0, 0
    for data in all_data:
        tokens = jieba.cut(data["prompt"]["text"])
        tokens = list(tokens)
        if len(tokens) < 20:
            continue
        if "severe_toxicity" in data["prompt"] and data["prompt"]["severe_toxicity"] is not None and data["prompt"][
            "severe_toxicity"] >= 0.8:
            if neg_num < data_num:
                return_data.append({"prompt": data["prompt"]["text"], "response": None, "label": "malicious"})
                neg_num += 1
        elif "severe_toxicity" in data["prompt"] and data["prompt"]["severe_toxicity"] is not None and data["prompt"][
            "severe_toxicity"] <= 0.1:
            if pos_num < data_num:
                return_data.append({"prompt": data["prompt"]["text"], "response": None, "label": "benign"})
                pos_num += 1
        print("pos_num, neg_num", pos_num, neg_num)
        if pos_num == data_num and neg_num == data_num:
            break

    write_jsonl("rtp_data.jsonl", return_data)


def get_rtp_lx_data(data_num=500):
    all_data = read_jsonl("RTP-LX/RTP_LX_EN.json")
    return_data = []
    pos_num, neg_num = 0, 0
    for data in all_data:
        tokens = jieba.cut(data["Prompt"])
        tokens = list(tokens)
        if len(tokens) < 20:
            continue
        token_num = len(data["Prompt"].split())
        return_data.append({"prompt": data["Prompt"], "response": None, "label": "malicious"})
        neg_num += 1
        regular_text = " ".join(data["BenignCompletion"].split()[:token_num])
        return_data.append({"prompt": regular_text, "response": None, "label": "benign"})
        pos_num += 1
        print("pos_num, neg_num", pos_num, neg_num)
        if pos_num == data_num and neg_num == data_num:
            break
    write_jsonl("rtp_lx_data.jsonl", return_data)


if __name__ == '__main__':
    get_rtp_data(data_num=500)
    get_rtp_lx_data(data_num=500)
