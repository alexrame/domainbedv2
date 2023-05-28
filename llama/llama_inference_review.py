import os
import torch

import numpy as np
import  llama_utils, llama_ppo_review, inference_utils, args_utils

device = 0 if torch.cuda.is_available() else "cpu"


class PredictorReview(inference_utils.Predictor):

    def get_rewards(self, texts):

        responses = [llama_utils.Instructions.get_response(q) for q in texts]
        rewards = [
            [
                sentiment_pipe(
                    llama_ppo_review.transform_text_review(
                        sentiment_pipe=sentiment_pipe,
                        response=response,
                    ), **self.sent_kwargs
                ) for sentiment_pipe in self.sentiment_pipes
            ] for response in responses
        ]

        rewards = [self.transform_reward(reward) for reward in rewards]
        return rewards


class Samples:

    @staticmethod
    def get_fake_samples_review(bs):
        prompt_review = llama_utils.Instructions.get_prompt_review()
        list_texts = [
            prompt_review + "We really hated the horrible hint towards this badaboum.",
            prompt_review + "We really enjoyed the slight hint towards this badaboum."
        ]
        size_prompt_review = llama_utils.Instructions.get_size_prompt_review(tokenizer)
        batch = [
            np.array(tokenizer.encode(text), dtype=np.int32)[:size_prompt_review + 8]
            for text in list_texts
        ][:bs]
        return batch

    @staticmethod
    def get_samples_review(dataset_name=None, bs=16):
        ds = llama_ppo_review.build_dataset(
            tokenizer=tokenizer, dataset_name=dataset_name, split="test"
        )

        ds.set_format("pandas")
        # df_batch = ds[:].sample(bs)
        df_batch = ds[:bs]
        query_tensors = df_batch['input_ids'].tolist()
        return query_tensors


if __name__ == "__main__":
    if os.environ.get("MERGE", "0") == "0":
        default_args = args_utils.DefaultArgsInference()
    else:
        default_args = args_utils.DefaultArgsInferenceMerged()
    script_args = args_utils.get_args_inference(
        default_args=default_args
    )
    args_utils.set_seed(script_args.seed)
    print(args_utils.Naming.str_dict(script_args.__dict__))

    # 1. load dataset and tokenizers
    tokenizer = llama_utils.Tokenizer.load_tokenizer(script_args.base_model_name)
    if script_args.dataset_name == "samples":
        query_tensors = Samples.get_fake_samples_review(bs=script_args.num_samples)
    else:
        query_tensors = Samples.get_samples_review(
            dataset_name=script_args.dataset_name,
            bs=script_args.num_samples)
    print("First query:", query_tensors[0])
    print("First decoded query:", tokenizer.decode(query_tensors[0]))

    # 2. load models
    base_model = inference_utils.Loader.load_base_model(script_args.base_model_name)
    sentiment_pipes = llama_utils.Pipelines.load_pipes(script_args.sentiment_models, device=device)


    # 3. inference for wa
    predictor = PredictorReview(
        sentiment_pipes=sentiment_pipes,
        tokenizer=tokenizer,
        output_max_length=script_args.output_max_length,
        device=device
    )
    resultscomputer = inference_utils.ResultsComputer(
        predictor=predictor,
        base_model=base_model,
        query_tensors=query_tensors,
        verbose=script_args.verbose
    )
    inference_utils.get_results_rewards(
        resultscomputer,
        peft_names=script_args.peft_names,
        every=script_args.every)
