from config import get_args_parser
from dualguard import DualGuard
from attack_method import AttackByGpt
from evaluate import watermark_evaluate, detect_evaluate, trace_evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, BertForMaskedLM, BertTokenizer
import torch
from utils import llm_path, vocab_size_dict, read_jsonl, write_jsonl, TransformersConfig


def main(args, algorithm_config, transformers_config):
    myWatermark = DualGuard(args, algorithm_config, transformers_config)
    input_data = read_jsonl(args.input_file)
    output_data = []
    watermarked_z_score_sum, unwatermarked_z_score_sum, translate_z_score_sum = 0, 0, 0

    for data in input_data:
        prompt = data["prompt"]

        if args.gen_unwatermark:
            gen_unwatermarked_text = myWatermark.generate_unwatermarked_text(prompt)
            gen_unwatermarked_text = gen_unwatermarked_text[len(prompt):]
            gen_unwatermarked_detect_result = myWatermark.detect_watermark(gen_unwatermarked_text)
        else:
            gen_unwatermarked_text = None
            gen_unwatermarked_detect_result = None

        # Generate and detect
        watermarked_text = myWatermark.generate_watermarked_text(prompt)
        watermarked_text = watermarked_text[len(prompt):]
        print("watermarked_text", watermarked_text)
        watermarked_detect_result = myWatermark.detect_watermark(watermarked_text)
        watermarked_z_score_sum += watermarked_detect_result["score"]
        print("watermarked_detect_result", watermarked_detect_result)

        if "c4" in args.dataset or "booksum" in args.dataset:
            unwatermarked_text = data["response"]
            unwatermarked_detect_result = myWatermark.detect_watermark(unwatermarked_text)
            unwatermarked_z_score_sum += unwatermarked_detect_result["score"]
            data["unwatermarked_text"] = unwatermarked_text
        else:
            unwatermarked_detect_result = None

        data["watermarked_text"] = watermarked_text
        data["watermarked_detect_result"] = watermarked_detect_result
        data["unwatermarked_detect_result"] = unwatermarked_detect_result
        data["gen_unwatermarked_text"] = gen_unwatermarked_text
        data["gen_unwatermarked_detect_result"] = gen_unwatermarked_detect_result
        output_data.append(data)
        print("-" * 30)
    print("watermarked: average: {:.2f} ({}/{})".format(watermarked_z_score_sum / len(output_data),
                                                        watermarked_z_score_sum, len(output_data)))
    print("unwatermarked: average: {:.2f} ({}/{})".format(unwatermarked_z_score_sum / len(output_data),
                                                          unwatermarked_z_score_sum, len(output_data)))
    write_jsonl(args.save_result_path, output_data)

    del myWatermark
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated(args.device) / 1024 ** 3
    print("allocated", allocated)


def attack(args, algorithm_config, transformers_config):
    attack_sys = AttackByGpt(llm_name=args.attack_llm)
    myWatermark = DualGuard(args, algorithm_config, transformers_config)
    input_data = read_jsonl(args.input_path)
    output_data = []
    for data in input_data:
        watermarked_text = data["watermarked_text"]

        para_paraphrase_text = attack_sys.paraphrase(watermarked_text, level="para")
        para_paraphrase_detect_result = myWatermark.detect_watermark(para_paraphrase_text)
        data["para_paraphrase_text"] = para_paraphrase_text
        data["para_paraphrase_detect_result"] = para_paraphrase_detect_result
        print("para_paraphrase_detect_result", para_paraphrase_detect_result)

        para_malicious_text = attack_sys.malicious_modify(watermarked_text, level="para")
        para_malicious_detect_result = myWatermark.detect_watermark(para_malicious_text)
        data["para_malicious_text"] = para_malicious_text
        data["para_malicious_detect_result"] = para_malicious_detect_result
        print("para_malicious_detect_result", para_malicious_detect_result)
        output_data.append(data)
        print("-" * 30)
    write_jsonl(args.save_result_path, output_data)

    if args.dataset == "rtp" or args.dataset == "rtp_lx":
        trace_evaluate(myWatermark, args.save_result_path, ["para_malicious_text"], ["para_paraphrase_text"])
    else:
        watermark_evaluate(args.save_result_path, ["watermarked_detect_result"], ["unwatermarked_detect_result"])
        watermark_evaluate(args.save_result_path, ["para_paraphrase_detect_result"], ["unwatermarked_detect_result"])
        detect_evaluate(myWatermark, args.save_result_path, ["para_malicious_text"], ["para_paraphrase_text"])


if __name__ == '__main__':
    args = get_args_parser()

    model_path = llm_path[args.llm_name]
    model = AutoModelForCausalLM.from_pretrained(model_path).to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if "llama" in args.llm_name:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    vocab_size = vocab_size_dict[args.llm_name]
    algorithm_config = "None"
    transformers_config = TransformersConfig(model=model,
                                             tokenizer=tokenizer,
                                             vocab_size=vocab_size,
                                             device=args.device,
                                             max_new_tokens=args.max_new_tokens,
                                             min_length=args.min_new_tokens,
                                             do_sample=True,
                                             no_repeat_ngram_size=4)
    main(args, algorithm_config, transformers_config)
    args.input_path = args.save_result_path
    attack(args, algorithm_config, transformers_config)
