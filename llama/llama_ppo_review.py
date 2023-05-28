# python3 llama_ppo_imdb.py --sentiment_model OpenAssistant/reward-model-deberta-v3-base --score_goal 0 --output_max_length 32 --prompt medium --ref_model newpeft
# python3 llama_ppo_imdb.py --sentiment_model OpenAssistant/reward-model-electra-large-discriminator --score_goal 0 --output_max_length 32 --prompt medium --ref_model newpeft
# python3 llama_ppo_imdb.py --sentiment_model OpenAssistant/reward-model-electra-large-discriminator --score_goal 0 --output_max_length 32 --prompt medium --ref_model none --base_model_name alexrame/alpaca-lora-7b-merged --peft_name lora
# python3 llama_ppo_summary.py --sentiment_model CogComp/bart-faithful-summary-detector --score_goal "1-0" --output_max_length 32 --ref_model newpeft
# python3 llama_ppo_summary.py --sentiment_model CogComp/bart-faithful-summary-detector --score_goal "1-0" --output_max_length 32 --ref_model newpeft --batch_size 12

import os
import torch
from datasets import load_dataset
# ds = load_dataset("imdb")
from trl import PPOConfig, PPOTrainer
from trl.core import LengthSampler

import llama_utils, ppo_utils, args_utils



def build_dataset(
    tokenizer, dataset_name="imdb", split="train"
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
    # load imdb with datasets
    os.environ['HF_DATASETS_OFFLINE'] = "1"
    if dataset_name == "default":
        dataset_name = "imdb"
    assert dataset_name == "imdb"

    ds = load_dataset(dataset_name, split=split)
    ds = ds.filter(lambda x: len(x["text"]) > 200, batched=False)
    size_prompt_review = llama_utils.Instructions.get_size_prompt_review(tokenizer)

    input_size_sampler = LengthSampler(
        size_prompt_review + 2,
        size_prompt_review + 8
    )

    def tokenize(sample):
        input_size = input_size_sampler()
        sample["input_ids"] = tokenizer.encode(
            llama_utils.Instructions.get_prompt_review() + sample["text"])[:input_size]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False, load_from_cache_file=False)
    ds.set_format(type="torch")
    return ds

def transform_text_review(sentiment_pipe, response):
    if sentiment_pipe.model.name_or_path.startswith("OpenAssistant/"):
        if os.environ.get("FIXED", "0") == "0":
            return sentiment_pipe.tokenizer.cls_token + llama_utils.Instructions.instruction_review + sentiment_pipe.tokenizer.sep_token + response
        else:
            return llama_utils.Instructions.instruction_review + sentiment_pipe.tokenizer.sep_token + response
    if sentiment_pipe.model.name_or_path.startswith("sugam11/gpt2-rlhf-reward"):
        return "Human: " + llama_utils.Instructions.instruction_review + " Assistant: " + response
    if sentiment_pipe.model.name_or_path.startswith("theblackcat102"):
        return llama_utils.Instructions.instruction_review + sentiment_pipe.tokenizer.sep_token + response
    if sentiment_pipe.model.name_or_path.startswith("ChaiML/"):
        return "User: " + llama_utils.Instructions.instruction_review + "\nBot: " + response
    return response


if __name__ == "__main__":
    if os.environ.get("MERGE", "0") == "0":
        default_args = args_utils.DefaultArgsPPO()
    else:
        default_args = args_utils.DefaultArgsPPOMerged()
    script_args = args_utils.get_args_ppo(default_args=default_args)
    script_args.task = "review"

    for sentiment_model in script_args.sentiment_models:
        assert sentiment_model in args_utils.DefaultArgs.sentiment_models_review
    print(args_utils.Naming.str_dict(script_args.__dict__))

    base_model = ppo_utils.Loader.load_base_model(script_args.base_model_name)
    model = ppo_utils.Loader.load_peft_model(base_model, peft_name=script_args.peft_name)

    os.environ["WANDB_DIR"] = os.path.join(args_utils.FOLDER_EXPE, "wandb")
    script_args.script_args_name = args_utils.Naming.get_name(script_args)
    os.environ["WANDB_NAME"] = script_args.script_args_name

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

    runner = ppo_utils.Runner(
        ppo_trainer=ppo_trainer,
        tokenizer=tokenizer,
        device=device,
        script_args=script_args,
        transform_text_method=transform_text_review,
        warmup_scheduler=warmup_scheduler
    )
    runner.train_ppo(model, num_epochs=script_args.num_epochs)
