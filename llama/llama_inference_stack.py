import torch
import os

import numpy as np
import llama_utils, llama_ppo_stack, inference_utils, args_utils

assert torch.cuda.is_available()
device = 0 if torch.cuda.is_available() else "cpu"

class PredictorStack(inference_utils.Predictor):

    def get_rewards(self, texts):

        queries_responses = [
            (llama_utils.Instructions.get_input(text), llama_utils.Instructions.get_response(text))
            for text in texts
        ]
        rewards = [
            [
                sentiment_pipe(
                    llama_ppo_stack.transform_text_stack(
                        sentiment_pipe=sentiment_pipe, post=query, response=response
                    ), **self.sent_kwargs
                ) for sentiment_pipe in self.sentiment_pipes
            ] for query, response in queries_responses
        ]

        rewards = [self.transform_reward(reward) for reward in rewards]
        return rewards


class Samples:

    @staticmethod
    def get_fake_samples_stack(bs):

        list_posts = [
            'For example, User adds this "iamsmelly.com". And if I add an href to this, the link would be www.mywebsite.com/iamsmelly.com Is there a way to make it absolute if its not prepended by an http:// ? Or should I revert to jQuery for t',
            'I have a `<div id="content">`. I want to load the content from <http://vietduc24h.com> into my `div`: ``` <html> <head> <script type="text/javascript"> $(document).ready(function() { $("#content").attr("src","http://vietduc24h.com"); }) </script> </head> <body> <div id="content"></div> </body> </html ``` I dont want to use an iframe. How can I do this?',
        ]
        list_responses = [
            "As stated in the docs The Support Library 26.0 provide",
            "Thierry Henry is a footballer",
        ]
        list_texts = [
            llama_utils.Instructions.get_prompt_noinput(instruction=post) + response
            for post, response in zip(list_posts, list_responses)
        ]

        batch = [np.array(tokenizer.encode(text), dtype=np.int32)[:-1] for text in list_texts][:bs]
        return batch

    @staticmethod
    def get_samples_stack(dataset_name, bs=16):
        ds = llama_ppo_stack.build_dataset(
            dataset_name=dataset_name, tokenizer=tokenizer, split="validation"
        )

        ds.set_format("pandas")
        # df_batch = ds[:].sample(bs)
        df_batch = ds[:bs]
        query_tensors = df_batch['input_ids'].tolist()
        # [print(i) for i in df_batch["query"][:3]]
        return query_tensors


if __name__ == "__main__":
    if os.environ.get("MERGE", "0") == "0":
        default_args = args_utils.DefaultArgsInferenceStack()
    else:
        raise ValueError()

    script_args = args_utils.get_args_inference(default_args=default_args)
    args_utils.set_seed(script_args.seed)
    print(args_utils.Naming.str_dict(script_args.__dict__))

    # 1. load dataset and tokenizers
    tokenizer = llama_utils.Tokenizer.load_tokenizer(script_args.base_model_name)
    if script_args.dataset_name == "samples":
        query_tensors = Samples.get_fake_samples_stack(bs=script_args.num_samples)
    else:
        query_tensors = Samples.get_samples_stack(
            dataset_name=script_args.dataset_name, bs=script_args.num_samples
        )
    print("First query:", query_tensors[0])
    print("First decoded query:", tokenizer.decode(query_tensors[0]))

    # 2. load models
    sentiment_pipes = llama_utils.Pipelines.load_pipes(script_args.sentiment_models, device=device)
    base_model = inference_utils.Loader.load_base_model(script_args.base_model_name)

    # 3. inference for wa
    predictor = PredictorStack(
        sentiment_pipes=sentiment_pipes,
        tokenizer=tokenizer,
        output_max_length=script_args.output_max_length,
        device=device,
    )
    resultscomputer = inference_utils.ResultsComputer(
        predictor=predictor,
        base_model=base_model,
        query_tensors=query_tensors,
        verbose=script_args.verbose
    )
    inference_utils.get_results_rewards(
        resultscomputer, peft_names=script_args.peft_names, every=script_args.every
    )
