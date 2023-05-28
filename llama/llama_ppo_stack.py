import os
import numpy as np
import torch
from datasets import load_dataset, load_from_disk

from trl import PPOConfig, PPOTrainer
from trl.core import LengthSampler

import llama_utils, ppo_utils, args_utils
# see this https://github.com/lvwerra/trl/blob/main/examples/stack_llama/scripts/rl_training.py
MIN_SIZE = 5


def build_dataset(dataset_name, *args, **kwargs):
    os.environ['HF_DATASETS_OFFLINE'] = "1"
    dataset = _build_stack_dataset(dataset_name=dataset_name, *args, **kwargs)
    print(f"Loaded dataset {dataset_name}:", dataset)
    return dataset


def _build_stack_dataset(
    dataset_name,
    tokenizer,
    split="train",
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    split = {"train": "train", "validation": "test"}[split]
    if dataset_name == "uniq":
        save_name = "stack_data_sampled_uniq"
    else:
        save_name = "stack_data_sampled"
    print("Loading dataset:", save_name)

    if args_utils.LOCAL_FILES_ONLY:
        ds = load_from_disk("/gpfsdswork/projects/rech/edr/utr15kn/dataplace/data/huggingface/" + save_name)
        ds = ds[split]
    else:
        ds = load_from_disk("/data/rame/data/huggingface/" + save_name)
        ds = ds[split]
        # ds = load_dataset("lvwerra/stack-exchange-paired"),
        #     cache_dir="/data/rame/data/huggingface")
        # def remove_duplicate(duplicated_dataset):
        #     initial_list = duplicated_dataset.map(lambda x: {"id": x['qid']})
        #     _ , unique_indices = np.unique(initial_list["qid"], return_index=True, axis=0)
        #     filtered_dataset = duplicated_dataset.select(unique_indices.tolist())
        #     return filtered_dataset
        # dtrainf_deduplicated = remove_duplicate(dtrainf)
        # dtrainf = ds["train"].filter(lambda x: len(x["question"]) < 300, batched=False)
        # dtrainfs = dtrainf.select(range(40000))
        # dtestf = ds["test"].filter(lambda x: len(x["question"]) < 300, batched=False)
        # dtestfs = dtestf.select(range(1000))
        # import datasets
        # dd = datasets.DatasetDict({"train":dtrainfs,"test":dtestfs})
        # dd.save_to_disk("/gpfsdswork/projects/rech/edr/utr15kn/dataplace/data/huggingface/stack_data_sampled")

    input_size_sampler = LengthSampler(3, 9)

    def tokenize(sample):
        instruction = sample["question"].replace("\n", " ")
        prompt = llama_utils.Instructions.get_prompt_noinput(instruction=instruction,)
        size_prompt = len(tokenizer.encode(prompt)) - 1
        response = sample["response_k"].replace("\n", " ")
        input_size = size_prompt + input_size_sampler()
        sample["input_ids"] = tokenizer.encode(prompt + response)[:input_size][:-1]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds_mapped = ds.map(tokenize, batched=False, load_from_cache_file=False)
    ds_filtered = ds_mapped.filter(lambda x: len(x["input_ids"]) < 512, batched=False)
    ds_filtered.set_format(type="torch")
    return ds_filtered


def transform_text_stack(sentiment_pipe, post, response):
    if sentiment_pipe.model.name_or_path.startswith("OpenAssistant/"):
        return post + sentiment_pipe.tokenizer.sep_token + response
    if sentiment_pipe.model.name_or_path in ["trl-lib/llama-7b-se-rm-peft", "edbeeching/gpt2_stack-exchange-paired_rmts__10000_2e-05_hub"]:
        return "Question: " + post + "\n\nAnswer: " + response
    #truncation=True)
    if sentiment_pipe.model.name_or_path.startswith("sugam11/gpt2-rlhf-reward"):
        return "Human: " + post + " Assistant: " + response
    if sentiment_pipe.model.name_or_path.startswith("theblackcat102"):
        return post + sentiment_pipe.tokenizer.sep_token + response
    raise ValueError(sentiment_pipe)



if __name__ == "__main__":
    if os.environ.get("MERGE", "0") == "0":
        default_args = args_utils.DefaultArgsPPO()
    else:
        default_args = args_utils.DefaultArgsPPOMerged()
    script_args = args_utils.get_args_ppo(default_args=default_args)

    script_args.task = "stack"
    for sentiment_model in script_args.sentiment_models:
        assert sentiment_model in args_utils.DefaultArgs.sentiment_models_stack
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
        transform_text_method=transform_text_stack,
        warmup_scheduler=warmup_scheduler
    )

    for sentiment_pipe in runner.sentiment_pipes:
        if "llama" in sentiment_pipe.model.name_or_path:
            DEFAULT_PAD_TOKEN = "[PAD]"
            DEFAULT_EOS_TOKEN = "</s>"
            DEFAULT_BOS_TOKEN = "</s>"
            DEFAULT_UNK_TOKEN = "</s>"
            sentiment_pipe.tokenizer.add_special_tokens(
                {
                    "eos_token": DEFAULT_EOS_TOKEN,
                    "bos_token": DEFAULT_BOS_TOKEN,
                    "unk_token": DEFAULT_UNK_TOKEN,
                    "pad_token": DEFAULT_PAD_TOKEN,
                }
            )
        elif "gpt2" in sentiment_pipe.model.name_or_path:
            sentiment_pipe.tokenizer.pad_token = tokenizer.eos_token
    runner.train_ppo(model, num_epochs=script_args.num_epochs)
