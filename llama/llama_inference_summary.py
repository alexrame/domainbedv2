import torch
import os

import numpy as np
import llama_utils, llama_ppo_summary, inference_utils, args_utils

device = 0 if torch.cuda.is_available() else "cpu"

class PredictorSummary(inference_utils.Predictor):

    def get_rewards(self, texts):

        queries_responses = [
            (llama_utils.Instructions.get_input(text), llama_utils.Instructions.get_response(text))
            for text in texts
        ]
        rewards = [
            [
                sentiment_pipe(
                    llama_ppo_summary.transform_text_summary(
                        sentiment_pipe=sentiment_pipe, post=query, response=response
                    ), **self.sent_kwargs
                ) for sentiment_pipe in self.sentiment_pipes
            ] for query, response in queries_responses
        ]

        rewards = [self.transform_reward(reward) for reward in rewards]
        return rewards


class Samples:

    @staticmethod
    def get_fake_samples_summary(bs):

        list_posts = [
            "Zinedine Yazid Zidane popularly known as Zizou, is a French professional football manager and former player who played as an attacking midfielder. He most recently coached Spanish club Real Madrid and is one of the most successful coaches in the world. Widely regarded as one of the greatest players of all time, Zidane was a playmaker renowned for his elegance, vision, passing, ball control, and technique. He received many individual accolades as a player, including being named FIFA World Player of the Year in 1998, 2000 and 2003, and winning the 1998 Ballon d'Or.",
            "Thierry Daniel Henry is a French professional football coach, pundit, and former player. Considered one of the best strikers of all time, one of the best players to play in the Premier League and Arsenal's greatest player, Henry was runner-up for both the Ballon d'Or in 2003 and the FIFA World Player of the Year in 2003 and 2004.",
            "Pablo Escobar was named the FWA Footballer of the Year a record three times, the PFA Players Player of the Year a joint-record two times, and was named in the PFA Team of the Year six consecutive times. He was also included in the FIFA FIFPro World XI once and the UEFA Team of the Year five times."
        ]
        list_responses = [
            "Zinedine Zidane is a footballer",
            "Thierry Henry is a footballer",
            "The mafia is"
        ]
        list_texts = [
            llama_utils.Instructions.get_prompt_summary(post=post) + response
            for post, response in zip(list_posts, list_responses)
        ]

        batch = [np.array(tokenizer.encode(text), dtype=np.int32)[:-1] for text in list_texts][:bs]
        return batch

    @staticmethod
    def get_samples_summary(dataset_name, bs=16):
        ds = llama_ppo_summary.build_dataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            split="validation"
        )

        ds.set_format("pandas")
        # df_batch = ds[:].sample(bs)
        df_batch = ds[:bs]
        query_tensors = df_batch['input_ids'].tolist()
        # [print(i) for i in df_batch["query"][:3]]
        return query_tensors


if __name__ == "__main__":
    if os.environ.get("MERGE", "0") == "0":
        default_args = args_utils.DefaultArgsInferenceSummary()
    else:
        default_args = args_utils.DefaultArgsInferenceSummaryMerged()

    script_args = args_utils.get_args_inference(
        default_args=default_args
    )
    args_utils.set_seed(script_args.seed)
    print(args_utils.Naming.str_dict(script_args.__dict__))

    # 1. load dataset and tokenizers
    tokenizer = llama_utils.Tokenizer.load_tokenizer(script_args.base_model_name)
    if script_args.dataset_name == "samples":
        query_tensors = Samples.get_fake_samples_summary(bs=script_args.num_samples)
    else:
        query_tensors = Samples.get_samples_summary(
            dataset_name=script_args.dataset_name, bs=script_args.num_samples)
    print("First query:", query_tensors[0])
    print("First decoded query:", tokenizer.decode(query_tensors[0]))

    # 2. load models
    base_model = inference_utils.Loader.load_base_model(script_args.base_model_name)
    sentiment_pipes = llama_utils.Pipelines.load_pipes(script_args.sentiment_models, device=device)

    # 3. inference for wa
    predictor = PredictorSummary(
        sentiment_pipes=sentiment_pipes,
        tokenizer=tokenizer, output_max_length=script_args.output_max_length, device=device,
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
