This is the official repository for paper: 
DualGuard: Dual-stream Large Language Model Watermarking Defense against Paraphrase and Spoofing Attack (ACL 2026 Findings)

## Environment
To run DualGuard, please install all the dependency packages by using the following command:
```
conda create --name dualguard python=3.12.7
pip install -r requirements.txt
```

## Download model
* OPT-1.3B https://huggingface.co/facebook/opt-1.3b
* Llama-3.1-8B-Instruct https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
* cbert https://huggingface.co/perceptiveshawty/compositional-bert-large-uncased/tree/main
* sentiment-roberta-large-english https://huggingface.co/siebert/sentiment-roberta-large-english


### Training watermark mapping model
Directly use model/model_cbert.pth, or use the following command to generate [data](https://huggingface.co/datasets/mteb/stsbenchmark-sts) and train model.

```
python gen_emb.py --output_path data/mapping/train_emb_cbert.txt --emb_model cbert

python model.py --data_path data/mapping/train_emb_cbert.txt --output_model_name model_cbert1 --emb_model cbert
```

## Usage

Generate watermarked text.

```
from config import get_args_parser
from dualguard import DualGuard
from transformers import AutoModelForCausalLM, AutoTokenizer, BertForMaskedLM, BertTokenizer
from utils import llm_path, vocab_size_dict, TransformersConfig


args = get_args_parser()
model_path = llm_path[args.llm_name]
model = AutoModelForCausalLM.from_pretrained(model_path).to(args.device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
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

myWatermark = DualGuard(args, algorithm_config, transformers_config)
prompt = "dualguard"
watermarked_text = myWatermark.generate_watermarked_text(prompt)
print(watermarked_text)

```


Generate watermarked text and evaluate watermark detectability, paraphrase attack robustness, and spoofing attack robustness. Configure your API key in attack_method.py (Line 16) for watermarking attacks.
```
python main.py --attack_llm gpt-4.1-nano --llm_name opt1.3b --transform_model_name model_cbert --emb_model cbert --transform_model_input_dim 1024 --delta 0.1 --alpha 1.7 --dataset c4 --prefix_length 12 --watermark_prefix 1 
```

Generate watermarked text and evaluate spoofing attack traceability.

```
python main.py --attack_llm gpt-4.1-nano --llm_name opt1.3b --transform_model_name model_cbert --emb_model cbert --transform_model_input_dim 1024 --delta 0.1 --alpha 1.7 --dataset rtp --prefix_length 12 --watermark_prefix 1
```

## Citation
```
@inproceedings{li2026dualguard,
  title={DualGuard: Dual-stream Large Language Model Watermarking Defense against Paraphrase and Spoofing Attack},
  author={Li, Hao and Ren, Yubing and Cao, Yanan and Li, Yingjie and Fang, Fang and Wang, Shi and Guo, Li},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  pages={23338--23361},
  year={2026}
}
```
