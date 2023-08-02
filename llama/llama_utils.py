import os
from transformers import LlamaTokenizer
from transformers import pipeline
import args_utils



class Pipelines:

    @staticmethod
    def load_pipes(sentiment_models, device):
        return [
            Pipelines.load_pipe(sentiment_model, device) for sentiment_model in sentiment_models
        ]

    @staticmethod
    def load_pipe(sentiment_model, device):
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
