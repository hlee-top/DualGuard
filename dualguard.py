import json
import torch
import random
import numpy as np
from functools import partial
from transformers import LogitsProcessor, LogitsProcessorList

from utils import TransformersConfig, EmbeddingModel, cosine_similarity, write_jsonl
from model import TransformModel


class DualGuardConfig:
    """Config class for DualGuard algorithm, load config file and initialize parameters."""

    def __init__(self, arg_config, algorithm_config: str, transformers_config: TransformersConfig, *args,
                 **kwargs) -> None:
        """
            Initialize the DualGuard configuration.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        self.arg_config = arg_config
        self.delta = self.arg_config.delta
        self.scale_dimension = self.arg_config.scale_dimension
        self.prefix_length = self.arg_config.prefix_length
        self.watermark_prefix = self.arg_config.watermark_prefix

        self.z_threshold = 0
        self.transform_model_input_dim = self.arg_config.transform_model_input_dim

        self.transform_model_name = self.arg_config.transform_model_name
        self.mapping_path = self.arg_config.mapping_path

        # Load transformer model's configuration
        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs


class DualGuardUtils:
    """Utility class for DualGuard algorithm, contains helper functions."""

    def __init__(self, config: DualGuardConfig, *args, **kwargs) -> None:
        """
            Initialize the DualGuard utility class.

            Parameters:
                config (DualGuardConfig): Configuration for the DualGuard algorithm.
        """
        self.config = config
        self.transform_model = self._get_transform_model(self.config.transform_model_name,
                                                         config.transform_model_input_dim).to(self.config.device)
        self.emb_model = EmbeddingModel(self.config.arg_config.emb_model, self.config.device)
        self.mapping = self._get_mapping(self.config.mapping_path)

    def get_embedding(self, sentence: str) -> torch.FloatTensor:
        """Get the embedding of the input sentence."""
        return self.emb_model.get_embedding(sentence)

    def scale_vector(self, v: np.array) -> np.array:
        """Scale the input vector using tanh function."""
        mean = np.mean(v)
        v_minus_mean = v - mean
        v_minus_mean = np.tanh(1e3 * v_minus_mean)
        return v_minus_mean

    def scale_vectors_batch(self, V: np.ndarray) -> np.ndarray:
        """
        V: shape = (N, D)
        return: shape = (N, D)
        """
        mean = V.mean(axis=1, keepdims=True)
        return np.tanh((V - mean) * 1e3)

    def _get_mapping(self, mapping_name: str) -> list[int]:
        """Get the mapping for the input tokens."""
        input_size = self.config.vocab_size
        mapping_path = mapping_name + "map.json"

        # try loading mapping from the provided mapping path
        try:
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)

        # if the file does not exist, create a new mapping and save it to the provided mapping path
        except:
            mapping = [random.randint(0, self.config.scale_dimension - 1) for _ in range(input_size)]
            with open(mapping_path, 'w') as f:
                json.dump(mapping, f, indent=4)
        return mapping

    def _get_context_sentence(self, input_ids: torch.LongTensor):
        """Get the context sentence from the input_ids."""
        token_len = input_ids.shape[0]
        sentence = self.config.generation_tokenizer.decode(
            input_ids[token_len - self.config.watermark_prefix:token_len], skip_special_tokens=True)
        return sentence

    def _get_transform_model(self, model_name: str, input_dim: int) -> TransformModel:
        """Get the transform model from the provided model name."""
        model = TransformModel(input_dim=input_dim)
        model.load_state_dict(torch.load(model_name))
        return model

    def get_neg_similarity(self, output_pos, output_neg):
        similarity_array_pos = self.scale_vector(output_pos)
        similarity_array_neg = self.scale_vector(output_neg)
        similarity_array_pos = similarity_array_pos[self.mapping]
        similarity_array_neg = similarity_array_neg[self.mapping]
        return similarity_array_pos, similarity_array_neg

    def get_flag(self, input_ids):
        context = self.config.generation_tokenizer.decode(input_ids,
                                                          skip_special_tokens=True)
        context_embedding = self.get_embedding(context)

        pos_logits, neg_logits = self.transform_model(context_embedding)
        diff_score = 1 - cosine_similarity(pos_logits, neg_logits).cpu().detach().numpy().tolist()[0]
        if diff_score >= self.config.arg_config.alpha:
            return pos_logits, neg_logits, False
        else:
            return pos_logits, neg_logits, True

    def get_bias(self, input_ids: torch.LongTensor, saved_flag):
        """Get the bias for the input_ids."""
        token_len = input_ids.shape[0]
        if token_len % self.config.prefix_length == 0:
            _, _, flag = self.get_flag(input_ids)
        else:
            flag = saved_flag
        context_sentence = self._get_context_sentence(input_ids)
        context_embedding = self.get_embedding([context_sentence])
        if flag:
            output, _ = self.transform_model(context_embedding)
        else:
            _, output = self.transform_model(context_embedding)
        output = output.cpu()[0].numpy()
        similarity_array = self.scale_vector(output)[self.mapping]

        return similarity_array, flag


class DualGuardLogitsProcessor(LogitsProcessor):
    def __init__(self, config: DualGuardConfig, utils: DualGuardUtils, *args, **kwargs):
        self.prompt_length = None
        self.saved_flag = None
        self.config = config
        self.utils = utils

    def init_prompt(self, prompt_length):
        self.prompt_length = prompt_length
        self.saved_flag = True

    def _bias_logits(self, scores: torch.LongTensor, batched_bias: torch.FloatTensor) -> torch.FloatTensor:
        """Bias the logits using the batched_bias."""
        scores = torch.mul(scores, (1 + self.config.delta * batched_bias))
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process the logits to add watermark."""
        batched_bias = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            current_bias, flag = self.utils.get_bias(input_ids[b_idx][self.prompt_length:], self.saved_flag)
            self.saved_flag = flag
            batched_bias[b_idx] = current_bias

        batched_bias_np = np.array(batched_bias)
        batched_bias = torch.Tensor(batched_bias_np).to(scores.device)

        scores = self._bias_logits(scores=scores, batched_bias=batched_bias)
        return scores


class DualGuard:
    """Top-level class for DualGuard algorithm."""

    def __init__(self, arg_config, algorithm_config: str, transformers_config: TransformersConfig, *args,
                 **kwargs) -> None:
        """
            Initialize the DualGuard algorithm.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        self.config = DualGuardConfig(arg_config, algorithm_config, transformers_config)
        self.utils = DualGuardUtils(self.config)
        self.logits_processor = DualGuardLogitsProcessor(self.config, self.utils)

    def generate_watermarked_text(self, prompt: str, *args, **kwargs):
        """Generate watermarked text."""

        # encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(
            self.config.device)

        prompt_length = encoded_prompt["input_ids"].shape[-1]
        self.logits_processor.init_prompt(prompt_length)

        # Configure generate_with_watermark
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            **self.config.gen_kwargs
        )

        # generate watermarked text
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
        # decode
        watermarked_text = \
            self.config.generation_tokenizer.batch_decode(encoded_watermarked_text, skip_special_tokens=True)[0]
        return watermarked_text

    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """Detect watermark in the input text."""

        token_id = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False).to(
            self.config.device)["input_ids"].squeeze(0)
        all_value, all_value_pos, all_value_neg = [], [], []

        watermark_prefix = self.config.watermark_prefix
        saved_flag = True
        id_list = [token_id[i:i + watermark_prefix] for i in range(len(token_id) - watermark_prefix + 1)]
        context_sent_list = self.config.generation_tokenizer.batch_decode(id_list, skip_special_tokens=True)

        context_embedding = self.utils.get_embedding(context_sent_list)
        output_pos_list, output_neg_list = self.utils.transform_model(context_embedding)

        pos_V = np.stack(output_pos_list.cpu().detach().numpy(), axis=0)  # (N, D)
        pos_scaled_V = self.utils.scale_vectors_batch(pos_V)
        pos_scaled_V = pos_scaled_V[:, self.utils.mapping]
        pos_scaled_list = [pos_scaled_V[i] for i in range(len(pos_scaled_V))]

        neg_V = np.stack(output_neg_list.cpu().detach().numpy(), axis=0)  # (N, D)
        neg_scaled_V = self.utils.scale_vectors_batch(neg_V)
        neg_scaled_V = neg_scaled_V[:, self.utils.mapping]
        neg_scaled_list = [neg_scaled_V[i] for i in range(len(neg_scaled_V))]

        for idx in range(watermark_prefix, token_id.shape[0]):
            token_len = token_id[:idx].shape[0]
            if token_len % self.config.prefix_length == 0:
                _, _, flag = self.utils.get_flag(token_id[:idx])
            else:
                flag = saved_flag

            saved_flag = flag
            if flag:
                similarity_array = pos_scaled_list[idx - watermark_prefix]
            else:
                similarity_array = neg_scaled_list[idx - watermark_prefix]
            all_value.append(float(similarity_array[token_id[idx]]))

        if len(all_value) != 0:
            z_score = np.mean(all_value).tolist()
        else:
            z_score = 0

        # Determine if the z_score indicates a watermark
        is_watermarked = z_score > self.config.z_threshold

        # Return results based on the return_dict flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score}
        else:
            return (is_watermarked, z_score)

    def detect_spoof(self, text: str, *args, **kwargs):
        token_id = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False).to(
            self.config.device)["input_ids"].squeeze(0)
        pos_logits, neg_logits = [], []

        watermark_prefix = self.config.watermark_prefix

        for idx in range(watermark_prefix, token_id.shape[0]):
            token_len = token_id[:idx].shape[0]
            if token_len % self.config.prefix_length == 0:
                output_pos, output_neg, flag = self.utils.get_flag(token_id[:idx])
                pos_logits.append(output_pos)
                neg_logits.append(output_neg)
        return {"pos_logits": pos_logits, "neg_logits": neg_logits}

    def trace_spoof(self, text: str, *args, **kwargs):
        token_id = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False).to(
            self.config.device)["input_ids"].squeeze(0)
        pos_hit_num, pos_num = 0, 0
        neg_hit_num, neg_num = 0, 0
        watermark_prefix = self.config.watermark_prefix
        saved_flag = True
        id_list = [token_id[i:i + watermark_prefix] for i in range(len(token_id) - watermark_prefix + 1)]
        context_sent_list = self.config.generation_tokenizer.batch_decode(id_list, skip_special_tokens=True)
        context_embedding = self.utils.get_embedding(context_sent_list)
        output_pos_list, output_neg_list = self.utils.transform_model(context_embedding)

        for idx in range(watermark_prefix, token_id.shape[0]):
            token_len = token_id[:idx].shape[0]
            if token_len % self.config.prefix_length == 0:
                _, _, flag = self.utils.get_flag(token_id[:idx])
            else:
                flag = saved_flag
            saved_flag = flag

            if flag is False:
                neg_num += 1
            else:
                pos_num += 1
            output_pos = output_pos_list[idx - watermark_prefix].cpu().detach().numpy()
            output_neg = output_neg_list[idx - watermark_prefix].cpu().detach().numpy()
            if flag:
                similarity_array = self.utils.scale_vector(output_pos)[self.utils.mapping]
            else:
                similarity_array = self.utils.scale_vector(output_neg)[self.utils.mapping]
            if flag is False and float(similarity_array[token_id[idx]]) > 0:
                neg_hit_num += 1
            if flag is True and float(similarity_array[token_id[idx]]) > 0:
                pos_hit_num += 1

        if neg_num == 0:
            neg_hit_ratio = 0
        else:
            neg_hit_ratio = neg_hit_num / neg_num
        if pos_num == 0:
            pos_hit_ratio = 0
        else:
            pos_hit_ratio = pos_hit_num / pos_num

        return {"neg_hit_ratio": neg_hit_ratio, "pos_hit_ratio": pos_hit_ratio}

    def generate_unwatermarked_text(self, prompt: str, *args, **kwargs) -> str:
        # Configure generate_without_watermark
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            **self.config.gen_kwargs
        )

        # encode prompt
        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(
            self.config.device)

        # generate unwatermarked_text text
        encoded_unwatermarked_text = generate_with_watermark(**encoded_prompt)

        # decode
        unwatermarked_text = \
            self.config.generation_tokenizer.batch_decode(encoded_unwatermarked_text, skip_special_tokens=True)[0]

        return unwatermarked_text
