import os
import numpy as np
import torch
from datasets import load_dataset, load_from_disk

from trl import PPOConfig, PPOTrainer
from trl.core import LengthSampler

import llama_utils, ppo_utils, args_utils

MIN_SIZE = 5


def build_dataset(dataset_name, *args, **kwargs):
    os.environ['HF_DATASETS_OFFLINE'] = "1"
    dataset = _build_rlhf_dataset(*args, **kwargs)
    print(f"Loaded dataset {dataset_name}:", dataset)
    return dataset


def _build_rlhf_dataset(tokenizer, split="train", max_size=100):
    split = {"train": "train", "validation": "test"}[split]

    if args_utils.LOCAL_FILES_ONLY:
        if split == "train":
            ds = load_from_disk("/gpfsdswork/projects/rech/edr/utr15kn/dataplace/hh_data")
        else:
            ds = load_from_disk("/gpfsdswork/projects/rech/edr/utr15kn/dataplace/data/huggingface/hh_data")
            ds = ds[split]
    else:
        ds = load_dataset("Anthropic/hh-rlhf", name="comparisons", split=split)

    ds_filtered = ds.filter(
        lambda x: x["chosen"] is not None and MIN_SIZE <
        len(x["chosen"].split("Assistant: ")[0]) < max_size,
        batched=False
    )

    input_size_sampler = LengthSampler(3, 9)

    def tokenize(sample):
        text = sample["chosen"].replace("\n", " ")
        instruction = text.split("Assistant: ")[0].split("Human: ")[1]
        prompt = llama_utils.Instructions.get_prompt_noinput(instruction=instruction,)
        response = text.split("Assistant: ")[1]
        size_prompt = len(tokenizer.encode(prompt)) - 1
        input_size = size_prompt + input_size_sampler()
        sample["input_ids"] = tokenizer.encode(prompt + response)[:input_size][:-1]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds_mapped = ds_filtered.map(tokenize, batched=False, load_from_cache_file=False)
    ds_mapped.set_format(type="torch")
    return ds_mapped


def transform_text_assistant(sentiment_pipe, post, response):
    if sentiment_pipe.model.name_or_path.startswith("OpenAssistant/"):
        return post + sentiment_pipe.tokenizer.sep_token + response
    if sentiment_pipe.model.name_or_path.startswith("sugam11/gpt2-rlhf-reward"):
        return "Human: " + post + " Assistant: " + response
    if sentiment_pipe.model.name_or_path.startswith("theblackcat102"):
        return post + sentiment_pipe.tokenizer.sep_token + response
    if sentiment_pipe.model.name_or_path.startswith("ChaiML/"):
        return "User: " + post + "\nBot: " + response
    raise ValueError(sentiment_pipe)

if __name__ == "__main__":
    if os.environ.get("MERGE", "0") == "0":
        default_args = args_utils.DefaultArgsPPO()
    else:
        default_args = args_utils.DefaultArgsPPOMerged()

    script_args = args_utils.get_args_ppo(default_args=default_args)

    script_args.task = "assistant"
    for sentiment_model in script_args.sentiment_models:
        assert sentiment_model in args_utils.DefaultArgs.sentiment_models_assistant
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
        dataset_name=script_args.dataset_name, tokenizer=tokenizer, split="train"
    )

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate
    )
    ppo_utils.Loader.load_optimizer(
        optimizer, peft_name=script_args.peft_name,
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
        transform_text_method=transform_text_assistant,
        warmup_scheduler=warmup_scheduler
    )

    runner.train_ppo(model, num_epochs=script_args.num_epochs)
