import torch
from openai import OpenAI
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertForMaskedLM, BertTokenizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet
import random


class AttackByGpt:
    def __init__(self, llm_name="gpt-4.1-nano", max_new_tokens=4096):
        self.model_name = llm_name
        print("attack model_name", self.model_name)

        api_key, base_url = None, None
        self.max_new_tokens = max_new_tokens
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def para_paraphrase_prompt(self, text):
        return ("Task Description:\n"
                "Rewrite the following text while preserving the original meaning. Ensure that the output meets the following criteria:\n"
                "1. Modification Scope: Restrict modification to the sentence level, rewriting them whenever feasible.\n"
                "2. Modification Requirement: Rewrite the sentences by employing different vocabulary and syntactic constructions to the greatest extent possible, while preserving logical coherence and consistency.\n"
                "3. Preserve Meaning: Rewrite should convey the same core idea and without omitting or distorting information.\n"
                "4. Fluency and Grammar: Rewrite must be natural, grammatically correct, and well-structured.\n"
                "5. Appropriate Length: Maintain a similar length unless a slight adjustment improves clarity.\n"
                "6. Consistency with Context: Retain the original tone and formality (e.g., academic, casual, professional).\n"
                "7. Minimal Redundancy: Avoid unnecessary repetition while keeping essential details.\n"
                "8. Retains Nuances: Preserve connotations, implied meanings, and idiomatic expressions where appropriate.\n"
                "Just provide the text after rewriting, without any introductory or concluding phrases.\n"
                "Original Text:\n{}".format(text))

    def para_malicious_prompt(self, text):
        return ("Task Description:\n"
                "Maliciously rewrite the following text to express more negative, pessimistic, harmful, hateful, or other negative content. Ensure that the output meets the following criteria:\n"
                "1. Modification Scope: Restrict modification to the sentence level, rewriting them whenever feasible.\n"
                "2. Modification Requirement: Maliciously rewrite the sentences by employing different vocabulary and syntactic constructions to the greatest extent possible, and ensure that the modified text has obvious negative and pessimistic content.\n"
                "3. Fluency and Grammar: Malicious rewrite must be natural, grammatically correct, and well-structured.\n"
                "4. Appropriate Length: Maintain a similar length unless a slight adjustment improves clarity.\n"
                "5. Consistency with Context: Retain the original tone and formality (e.g., academic, casual, professional).\n"
                "6. Minimal Redundancy: Avoid unnecessary repetition while keeping essential details.\n"
                "7. Retains Nuances: Preserve connotations, implied meanings, and idiomatic expressions where appropriate.\n"
                "Just provide the text after malicious rewriting, without any introductory or concluding phrases.\n"
                "Original Text:\n{}".format(text))

    def paraphrase(self, text, level):
        input_content = self.para_paraphrase_prompt(text)
        print("paraphrase prompt", input_content)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=self.max_new_tokens,
                messages=[
                    {"role": "system", "content": "Your are a helpful assistant to rewrite the text."},
                    {"role": "user", "content": input_content},
                ],
                temperature=0,
            )

            result = response.choices[0].message.content.strip()
        except Exception as e:
            print("e", e)
            print("inference except")
            result = ""
        return result

    def malicious_modify(self, text, level):
        input_content = self.para_malicious_prompt(text)
        print("malicious_modify prompt", input_content)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=self.max_new_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": input_content},
                ],
                temperature=0,
            )

            result = response.choices[0].message.content.strip()
        except Exception as e:
            print("e", e)
            print("inference except")
            result = ""
        return result

    def rephrase(self, text):
        prompt = "Rewrite the following paragraph:\n "
        input_content = prompt + text
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=self.max_new_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": input_content},
                ],
                temperature=0,
            )
            print("response", response)
            result = response.choices[0].message.content.strip()
        except Exception as e:
            print("e", e)
            print("inference except")
            result = ""
        return result

    def translate_prompt(self, text, source_language, target_language):
        return "Translate the following {} text to {}:\n{}".format(source_language, target_language, text)

    def translate(self, text):
        input_content = self.translate_prompt(text, "English", "Chinese")
        print("translate before prompt", input_content)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=self.max_new_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": input_content},
                ],
                temperature=0,
            )
            print("response", response)
            result = response.choices[0].message.content.strip()
            input_content = self.translate_prompt(result, "Chinese", "English")
            print("translate after prompt", input_content)
            response = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=self.max_new_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": input_content},
                ],
                temperature=0,
            )
            print("response", response)
            result = response.choices[0].message.content.strip()
        except Exception as e:
            print("e", e)
            print("inference except")
            result = ""
        return result


class DipperParaphraser(object):
    def __init__(self, model="dipper-paraphraser-xxl", verbose=True):
        time1 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained('t5-v1_1-xxl')
        self.model = T5ForConditionalGeneration.from_pretrained(model, device_map='auto')
        if verbose:
            print(f"{model} model loaded in {time.time() - time1}")
        self.model.eval()

    def paraphrase(self, input_text, lex_diversity, order_diversity, prefix="", sent_interval=3, **kwargs):
        """Paraphrase a text using the DIPPER model.

        Args:
            input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
            lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            **kwargs: Additional keyword arguments like top_p, top_k, max_length.
        """
        assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        # import ipdb; ipdb.set_trace()
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""

        for sent_idx in range(0, len(sentences), sent_interval):
            start = time.time()
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            final_input = self.tokenizer([final_input_text], return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}

            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]
            end = time.time()
            print(end - start)

        return output_text

    def paraphrase_text(self, text, lex_diversity, order_diversity):
        print(lex_diversity, order_diversity)
        return self.paraphrase(text, lex_diversity=lex_diversity, order_diversity=order_diversity, prefix="",
                               do_sample=True, top_p=0.75, top_k=None, max_length=512)


class TextEditor:
    """Base class for text editing."""

    def __init__(self) -> None:
        pass

    def edit(self, text: str, reference=None, is_watermarked=True):
        return text


class WordDeletion(TextEditor):
    """Delete words randomly from the text."""

    def __init__(self, ratio: float) -> None:
        """
            Initialize the word deletion editor.

            Parameters:
                ratio (float): The ratio of words to delete.
        """
        self.ratio = ratio

    def edit(self, text: str, reference=None, is_watermarked=True):
        """Delete words randomly from the text."""

        # Handle empty string input
        if not text:
            return text

        # Split the text into words and randomly delete each word based on the ratio
        word_list = text.split()
        edited_words = [word for word in word_list if random.random() >= self.ratio]

        # Join the words back into a single string
        deleted_text = ' '.join(edited_words)

        return deleted_text


class SynonymSubstitution(TextEditor):
    """Randomly replace words with synonyms from WordNet."""

    def __init__(self, ratio: float) -> None:
        """
            Initialize the synonym substitution editor.

            Parameters:
                ratio (float): The ratio of words to replace.
        """
        self.ratio = ratio
        # Ensure wordnet data is available
        # nltk.download('wordnet')

    def edit(self, text: str, reference=None, is_watermarked=True):
        """Randomly replace words with synonyms from WordNet."""
        words = text.split()
        num_words = len(words)

        # Dictionary to cache synonyms for words
        word_synonyms = {}

        # First pass: Identify replaceable words and cache their synonyms
        replaceable_indices = []
        for i, word in enumerate(words):
            if word not in word_synonyms:
                synonyms = [syn for syn in wordnet.synsets(word) if len(syn.lemmas()) > 1]
                word_synonyms[word] = synonyms
            if word_synonyms[word]:
                replaceable_indices.append(i)

        # Calculate the number of words to replace
        num_to_replace = min(int(self.ratio * num_words), len(replaceable_indices))

        # Randomly select words to replace
        if num_to_replace > 0:
            indices_to_replace = random.sample(replaceable_indices, num_to_replace)

            # Perform replacement
            for i in indices_to_replace:
                synonyms = word_synonyms[words[i]]
                chosen_syn = random.choice(synonyms)
                new_word = random.choice(chosen_syn.lemmas()[1:]).name().replace('_', ' ')
                words[i] = new_word

        # Join the words back into a single string
        replaced_text = ' '.join(words)

        return replaced_text


class ContextAwareSynonymSubstitution(TextEditor):
    """Randomly replace words with synonyms from WordNet based on the context."""

    def __init__(self, ratio: float, tokenizer: BertTokenizer, model: BertForMaskedLM, device='cuda') -> None:
        """
        Initialize the context-aware synonym substitution editor.

        Parameters:
            ratio (float): The ratio of words to replace.
            tokenizer (BertTokenizer): Tokenizer for BERT model.
            model (BertForMaskedLM): BERT model for masked language modeling.
            device (str): Device to run the model (e.g., 'cuda', 'cpu').
        """
        self.ratio = ratio
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def _get_synonyms_from_wordnet(self, word: str):
        """ Return a list of synonyms for the given word using WordNet. """
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        return list(synonyms)

    def edit(self, text: str, reference=None, is_watermarked=True):
        """Randomly replace words with synonyms from WordNet based on the context."""
        words = text.split()
        num_words = len(words)
        if num_words == 0:
            return text
        replaceable_indices = []

        for i, word in enumerate(words):
            if self._get_synonyms_from_wordnet(word):
                replaceable_indices.append(i)

        num_to_replace = int(min(self.ratio, len(replaceable_indices) / num_words) * num_words)
        indices_to_replace = random.sample(replaceable_indices, num_to_replace)

        real_replace = 0

        for i in indices_to_replace:
            # Create a sentence with a [MASK] token
            masked_sentence = words[:i] + ['[MASK]'] + words[i + 1:]
            masked_text = " ".join(masked_sentence)

            # Use BERT to predict the token for [MASK]
            inputs = self.tokenizer(masked_text, return_tensors='pt', padding=True, truncation=True).to(self.device)
            mask_position = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0].item()

            with torch.no_grad():
                outputs = self.model(**inputs)

            predictions = outputs.logits[0, mask_position]
            predicted_indices = torch.argsort(predictions, descending=True)
            predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_indices[0:1])
            words[i] = predicted_tokens[0]
            real_replace += 1

        replaced_text = ' '.join(words)

        return replaced_text
