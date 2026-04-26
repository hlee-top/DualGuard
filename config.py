import argparse
import torch
from utils import set_seed


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--llm_name', default="opt1.3b", type=str, help="large language model.")
    parser.add_argument("--seed", default=42, type=int, help="random seed.")
    parser.add_argument("--max_new_tokens", default=230, type=int, help="model max_new_tokens")
    parser.add_argument("--min_new_tokens", default=200, type=int, help="model min_new_tokens")

    parser.add_argument('--emb_model', type=str, default="cbert")
    parser.add_argument('--save_result_path', type=str)
    parser.add_argument("--hash_key", default=15485863, type=int)
    parser.add_argument('--transform_model_name', default="model_cbert", type=str)
    parser.add_argument("--transform_model_input_dim", default=1024, type=int)

    parser.add_argument('--dataset', type=str)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--scale_dimension", type=int, default=300)
    parser.add_argument('--gen_unwatermark', action='store_true')
    parser.add_argument('--attack_llm', type=str, default="gpt-4.1-nano")
    parser.add_argument("--delta", default=0.1, type=float)
    parser.add_argument("--alpha", default=1.7, type=float)
    parser.add_argument("--prefix_length", default=12, type=int)
    parser.add_argument("--watermark_prefix", default=1, type=int)

    parser.add_argument('--input_path', type=str)

    args = parser.parse_args()

    if args.dataset == "c4":
        args.input_file = "data/c4_val.jsonl"
    elif args.dataset == "booksum":
        args.input_file = "data/booksum_val.jsonl"
    elif args.dataset == "rtp":
        args.input_file = "data/rtp_data.jsonl"
    elif args.dataset == "rtp_lx":
        args.input_file = "data/rtp_lx_data.jsonl"


    if args.save_result_path is None:
        args.save_result_path = "{}_{}_{}_d{}_a{}_p{}_w{}.jsonl".format(args.dataset, args.llm_name, args.emb_model, args.delta, args.alpha, args.prefix_length, args.watermark_prefix)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.mapping_path = "data/mapping/{}_".format(args.llm_name)
    args.cluster_center_path = "data/mapping/center_{}.json".format(args.llm_name)

    args.transform_model_name = "model/{}.pth".format(args.transform_model_name)



    set_seed(args.seed)
    print("args", args)
    return args
