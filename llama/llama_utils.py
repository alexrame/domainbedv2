import os
import torch
import numpy as np
import tqdm

from transformers import LlamaTokenizer
from transformers import pipeline
import args_utils
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

class MLMS:
    # https://stackoverflow.com/questions/70464428/how-to-calculate-perplexity-of-a-sentence-using-huggingface-masked-language-mode
    def __init__(self):
        model_name = 'cointegrated/rubert-tiny'
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.name_or_path = "cointegrated/rubert-tiny"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, sentence):
        return self.score(sentence)

    def score(self, sentence):
        tensor_input = self.tokenizer.encode(sentence, return_tensors='pt')
        repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
        mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
        masked_input = repeat_input.masked_fill(mask == 1, self.tokenizer.mask_token_id)
        labels = repeat_input.masked_fill( masked_input != self.tokenizer.mask_token_id, -100)
        with torch.inference_mode():
            loss = self.model(masked_input, labels=labels).loss
        return np.exp(loss.item())

class GPT2:
    # https://huggingface.co/docs/transformers/perplexity
    def __init__(self):
        self.device = "cuda"
        model_id = "gpt2-large"
        self.model = GPT2LMHeadModel.from_pretrained(model_id).to(self.device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    def __call__(self, sentence):
        return self.score(sentence)

    def score(self, sentence):
        encodings = self.tokenizer(sentence, return_tensors="pt")
        max_length = self.model.config.n_positions
        stride = 512
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl


class Pipelines:

    @staticmethod
    def load_pipes(sentiment_models, device):
        return [
            Pipelines.load_pipe(sentiment_model, device) for sentiment_model in sentiment_models
        ]

    @staticmethod
    def load_pipe(sentiment_model, device):
        if sentiment_model == "mlms":
            print("Load mlms")
            pipe = MLMS()
        else:
            print(f"Load sentiment model: {sentiment_model}")
            pipe = pipeline("text-classification", model=sentiment_model, device=device,
                            tokenizer=Tokenizer.load_tokenizer_name(sentiment_model))
        return pipe

class Tokenizer:

    @staticmethod
    def load_tokenizer(base_model_name):
        tokenizer_name = Tokenizer.load_tokenizer_name(base_model_name)
        tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_name,
            add_eos_token=True,
            padding_side="left",
            local_files_only=args_utils.LOCAL_FILES_ONLY
        )
        tokenizer.pad_token_id = 0
        return tokenizer

    @staticmethod
    def load_tokenizer_name(model_name):
        if model_name == "tloen/alpaca-lora-7b":
            return "decapoda-research/llama-7b-hf"
        if model_name == "alexrame/alpaca-lora-7b-merged":
            return "decapoda-research/llama-7b-hf"
        if "llama-7b" in model_name:
            return "decapoda-research/llama-7b-hf"
        if "5m_reward_model" in model_name:
            return "gpt2"
        if "sugam11/gpt2-rlhf-reward" in model_name:
            return "microsoft/DialogRPT-updown"
        return model_name




class Instructions:
    # instructions_summary = [
    #     "Generate a one-sentence summary of this post."
    #     "Read the following text and generate a 1 short sentence summary.",
    #     "Generate a fake summary modifying key elements such as names or places. All facts should be false and HALLUCINATED.",
    #     "Write a concise summary of a fact.",
    #     "Write a concise summary of the provided text.",
    #     "Create a summary of the text below", "Generate a summary.",
    #     "Write a one-sentence summary of the following news article.",
    #     "Construct a concise summary of the following article.",
    #     "Analyze the following news article and provide a brief summary.",
    #     "Write a summary of the given article in less than 100 words that accurately reflects the key points of the article."
    # ]
    # instruction_review = "Generate an helpful harmless honest movie review."
    # instruction_review = "Let's review this movie step by step."

    instruction_review = "Generate a movie review."
    instruction_summary = "Generate a one-sentence summary of this post."
    instruction_context = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

    response_split = "### Response:"
    input_split = "### Input:"

    @classmethod
    def get_prompt_review(cls):
        return cls.get_prompt_noinput(cls.instruction_review).replace("\n", " ")
        # return cls.get_prompt_context_noinput(cls.instruction_review)

    @classmethod
    def get_size_prompt_review(cls, tokenizer):
        prompt_review = cls.get_prompt_review()
        return len(tokenizer.encode(prompt_review)) - 1

    @classmethod
    def get_prompt_summary(cls, post):
        return cls.get_prompt_input(cls.instruction_summary, post)

    @classmethod
    def get_prompt_input(cls, instruction, input):
        return f"### Instruction: {instruction} ### Input: {input} ### Response: "

    # @classmethod
    # def get_prompt_context_input(cls, instruction, input):
    #     return f"{cls.instruction_context} ### Instruction: {instruction} ### Input: {input} ### Response: "

    @staticmethod
    def get_prompt_noinput(instruction):
        return f"### Instruction: {instruction} ### Response: "

    # @classmethod
    # def get_prompt_context_noinput(cls, instruction):
    #     return f"{cls.instruction_context} ### Instruction: {instruction} ### Response: "

    @staticmethod
    def get_input(query):
        after_input = ". ".join(query.split(Instructions.input_split)[1:]).replace("\n", " ").strip()
        # if after_input == "":
        #     after_input = ". ".join(query.split(Instructions.instruction_split)[1:]).replace("\n"," ").strip()
        #     assert after_input != ""
        before_response = after_input.split(Instructions.response_split)[0]
        return before_response

    @staticmethod
    def get_response(response):
        return ". ".join(response.split(Instructions.response_split)[1:]).replace("\n", " ").strip()
