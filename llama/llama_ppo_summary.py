import os
import numpy as np
import torch
from datasets import load_dataset

from trl import PPOConfig, PPOTrainer
from trl.core import LengthSampler

import llama_utils, ppo_utils, args_utils

MIN_SIZE = 100
MAX_SIZE_NEWS = os.environ.get("MAX_SIZE_NEWS", 1500)

def build_dataset(dataset_name, *args, **kwargs):
    os.environ['HF_DATASETS_OFFLINE'] = "1"
    if dataset_name == "news":
        dataset = _build_news_dataset(*args, **kwargs)
    else:
        dataset = _build_openai_dataset(*args, **kwargs)
    print(f"Loaded dataset {dataset_name}:", dataset)
    return dataset


def _build_news_dataset(
    tokenizer,
    split="train",
):
    """
    Args:
        dataset_name (`str`): "argilla/news-summary"
    """
    split = {"train": "test", "validation": "train"}[split]
    ds = load_dataset("argilla/news-summary", name="comparisons", split=split)
    ds_filtered = ds.filter(
        lambda x: x["text"] is not None and MIN_SIZE < len(x["text"]) < MAX_SIZE_NEWS and x["id"] is
        not None,
        batched=False
    )
    def remove_duplicate(duplicated_dataset):
        initial_list = duplicated_dataset.map(lambda x: {"id": x['id']})
        _ , unique_indices = np.unique(initial_list["id"], return_index=True, axis=0)
        filtered_dataset = duplicated_dataset.select(unique_indices.tolist())
        return filtered_dataset

    ds_deduplicated = remove_duplicate(ds_filtered)
    input_size_sampler = LengthSampler(
        2,
        8
    )

    def tokenize(sample):
        info_post = "-".join(sample["text"].replace("\n", " ").split("(Reuters) -")[1:]).strip()
        prompt_summary = llama_utils.Instructions.get_prompt_summary(post=info_post)
        size_prompt_summary = len(tokenizer.encode(prompt_summary)) - 1
        input_size = size_prompt_summary + input_size_sampler()
        choice = 0 # select the best summary
        response = sample["prediction"][choice]["text"].replace("\n", " ").replace(".", ",")
        sample["input_ids"] = tokenizer.encode(prompt_summary + response)[:input_size]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds_mapped = ds_deduplicated.map(tokenize, batched=False, load_from_cache_file=False)
    ds_mapped.set_format(type="torch")
    return ds_mapped


def _build_openai_dataset(tokenizer, split="train", max_size=1200):
    ds = load_dataset("openai/summarize_from_feedback", name="comparisons", split=split)
    ds = ds.filter(
        lambda x: x["info"]["post"] is not None and MIN_SIZE < len(x["info"]["post"]) < max_size and
        x['info']["id"] is not None,
        batched=False
    )

    def remove_duplicate(duplicated_dataset):
        initial_list = duplicated_dataset.map(lambda x: {"id": x['info']["id"]})
        _ , unique_indices = np.unique(initial_list["id"], return_index=True, axis=0)
        filtered_dataset = duplicated_dataset.select(unique_indices.tolist())
        return filtered_dataset

    ds = remove_duplicate(ds)

    input_size_sampler = LengthSampler(
        2,
        8
    )

    def tokenize(sample):
        info_post = sample["info"]["post"].replace("\n", " ")
        prompt_summary = llama_utils.Instructions.get_prompt_summary(post=info_post)
        size_prompt_summary = len(tokenizer.encode(prompt_summary)) - 1
        input_size = size_prompt_summary + input_size_sampler()
        choice = sample["choice"] # select the best summary
        response = sample["summaries"][choice]["text"].replace("\n", " ").replace(".", ",")
        sample["input_ids"] = tokenizer.encode(prompt_summary + response)[:input_size]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False, load_from_cache_file=False)
    ds.set_format(type="torch")
    return ds


def transform_text_summary(sentiment_pipe, post, response):
    response = response.split(".")[0] + "." # selecting only first sentence
    if sentiment_pipe.model.name_or_path.startswith("CogComp/bart-faithful-summary-detector"):
        return response + sentiment_pipe.tokenizer.eos_token + sentiment_pipe.tokenizer.eos_token + post
    if sentiment_pipe.model.name_or_path.startswith("Tristan/gpt2_reward_summarization"):
        return response + " " + sentiment_pipe.tokenizer.bos_token + " " + post
        # truncation=True)
    if sentiment_pipe.model.name_or_path.startswith("valurank/distilbert-quality"):
        return response
    raise ValueError(sentiment_pipe)


if __name__ == "__main__":
    if os.environ.get("MERGE", "0") == "0":
        default_args = args_utils.DefaultArgsPPO()
    else:
        default_args = args_utils.DefaultArgsPPOMerged()
    script_args = args_utils.get_args_ppo(default_args=default_args)

    script_args.task = "summary"
    for sentiment_model in script_args.sentiment_models:
        assert sentiment_model in args_utils.DefaultArgs.sentiment_models_summary
    os.environ["WANDB_DIR"] = os.path.join(args_utils.FOLDER_EXPE, "wandb")
    script_args.script_args_name = args_utils.Naming.get_name(script_args)
    os.environ["WANDB_NAME"] = script_args.script_args_name
    print(args_utils.Naming.str_dict(script_args.__dict__))

    base_model = ppo_utils.Loader.load_base_model(script_args.base_model_name)
    model = ppo_utils.Loader.load_peft_model(base_model, peft_name=script_args.peft_name)


    config = PPOConfig(
        model_name=script_args.base_model_name,
        init_kl_coef=script_args.init_kl_coef,
        adap_kl_ctrl=script_args.adap_kl_ctrl,
        learning_rate=script_args.learning_rate,
        log_with=script_args.log_with if script_args.log_with != "" else None,
        batch_size=script_args.batch_size,
        mini_batch_size=script_args.mini_batch_size,
        optimize_cuda_cache=True,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        seed=script_args.seed
    )
    # set seed before initializing value head for deterministic eval
    args_utils.set_seed(script_args.seed)

    tokenizer = llama_utils.Tokenizer.load_tokenizer(script_args.base_model_name)
    ppo_utils.Loader.print_trainable_parameters(model)

    # We retrieve the dataloader by calling the `build_dataset` function.
    dataset = build_dataset(
        dataset_name=script_args.dataset_name,
        tokenizer=tokenizer,
        split="train")

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate
    )
    ppo_utils.Loader.load_optimizer(
        optimizer,
        peft_name=script_args.peft_name,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    warmup_scheduler = ppo_utils.get_scheduler(optimizer, script_args.warmup_steps)

    ref_model = ppo_utils.Loader.load_ref_model(script_args)
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer
    )

    # We then build the sentiment analysis pipeline, passing the model name and the
    # sentiment analysis pipeline arguments. Let's also make sure to set the device
    # to the same device as the PPOTrainer.
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

    runner = ppo_utils.RunnerInputQuery(
        ppo_trainer=ppo_trainer,
        tokenizer=tokenizer,
        device=device,
        script_args=script_args,
        transform_text_method=transform_text_summary,
        warmup_scheduler=warmup_scheduler
    )
    for sentiment_pipe in runner.sentiment_pipes:
        sentiment_pipe.tokenizer.pad_token_id = model.config.eos_token_id
    runner.train_ppo(model, num_epochs=script_args.num_epochs)
