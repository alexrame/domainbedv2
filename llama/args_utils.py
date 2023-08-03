import os
import argparse
import re
from datetime import datetime
from trl import set_seed

LOCAL_FILES_ONLY = __file__.startswith("/gpfs")
if LOCAL_FILES_ONLY:
    FOLDER_EXPE = "/gpfswork/rech/edr/utr15kn/dataplace/experiments/"
else:
    FOLDER_EXPE = "/data/rame/experiments"
LOAD_IN_8BIT = True
print("Loading from files only", LOCAL_FILES_ONLY)


class Naming:
    @staticmethod
    def str_dict(dict_args):
        """
        function that print args dictionaries in a beautiful way
        """
        str_out = "\n" + "#" * 40
        col_width = max(len(str(word)) for word in dict_args) + 2
        for arg in sorted(list(dict_args.keys())):
            if arg.startswith("__"):
                continue
            else:
                str_print = str(dict_args[arg])
                str_out += "\n" + "".join([str(arg).ljust(col_width), str_print])
        str_out += "\n" + "#" * 40 + "\n"
        return str_out

    @staticmethod
    def get_name(script_args):
        name = script_args.base_model_name.split("/")[-1] + "-ppo-" + script_args.task

        if script_args.dataset_name != "default":
            name += f"-d{script_args.dataset_name}"

        for sentiment_model in script_args.sentiment_models:
            short_sent = Naming.get_name_model(sentiment_model)
            name += f"-{short_sent}"

        score_goal = script_args.score_goal.replace("_", "").replace("x", "")
        name += f"-g{score_goal}"

        if script_args.learning_rate != DefaultArgsPPO.learning_rate:
            name += f"-lr{script_args.learning_rate}"

        if script_args.init_kl_coef != DefaultArgsPPO.init_kl_coef:
            init_kl_coef = str(script_args.init_kl_coef).split(".")[-1]
            name += f"-kl{init_kl_coef}"

        if script_args.peft_name not in [None, "None", "none", DefaultArgs.peft_name]:
            short_peft_name = script_args.peft_name.split("/")[-1]
            name += f"-p{short_peft_name}"

        name += "-" + datetime.now().strftime("%m-%d-%s")
        return name[:92]

    @staticmethod
    def get_name_model(name):
        list_sentiment_suffix = re.split("-|_", name.split('/')[-1])
        # short_sent = "-".join(list_sentiment_suffix[:2])
        # if len(list_sentiment_suffix) > 2:
        list_sentiment_suffix = [t for t in list_sentiment_suffix if t]
        short_sent = "".join([t[0] for t in list_sentiment_suffix])
        return short_sent


class DefaultArgs:
    seed = 0
    output_max_length = 32
    base_model_name = "decapoda-research/llama-7b-hf"
    # databricks/dolly-v1-6b
    # alexrame/alpaca-lora-7b-merged

    peft_name = "tloen/alpaca-lora-7b"

    if os.environ.get("DEBUG", "0") == "1":
        sentiment_models_review = ["lvwerra/distilbert-imdb"]
    elif True:
        sentiment_models_review = [
            "OpenAssistant/reward-model-deberta-v3-large-v2",
            "OpenAssistant/reward-model-deberta-v3-base",
            "OpenAssistant/reward-model-electra-large-discriminator",
            "theblackcat102/reward-model-deberta-v3-base-v2"
        ]
    else:
        # old setup
        sentiment_models_review = [
            "lvwerra/distilbert-imdb",
            "distilbert-base-uncased-finetuned-sst-2-english",
            "martin-ha/toxic-comment-model",
            "valurank/distilbert-quality",
            "OpenAssistant/reward-model-deberta-v3-large-v2",
            "OpenAssistant/reward-model-deberta-v3-base",
            "OpenAssistant/reward-model-electra-large-discriminator",
            "sugam11/gpt2-rlhf-reward",
            "unitary/toxic-bert", # 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
            # "pedropei/sentence-level-certainty",
            # "ChaiML/gpt2_base_retry_and_continue_5m_reward_model",
            # "ChaiML/gpt2_base_retry_and_continue_12m_reward_model",
            # "ChaiML/3plus_stars_gpt2_reward"
        ]
    # pipeline("text-classification", model="Tristan/gpt2_reward_summarization", device="cpu")
    # pipeline("text-classification", model="CogComp/bart-faithful-summary-detector", device="cpu")
    sentiment_models_summary = [
        "Tristan/gpt2_reward_summarization",
        "CogComp/bart-faithful-summary-detector",  #  True, label mapping: "0" -> "Hallucinated" "1" -> "Faithful"
        # "valurank/distilbert-quality",

    ]

    if os.environ.get("XXCAT", "0") == "1":
        sentiment_models_assistant = ["theblackcat102/deberta-v2-xxlarge-rm"]
    else:
        sentiment_models_assistant = ["OpenAssistant/reward-model-deberta-v3-large-v2"]
    sentiment_models_assistant.extend([
        "OpenAssistant/reward-model-deberta-v3-base",
        "OpenAssistant/reward-model-electra-large-discriminator",
        "theblackcat102/reward-model-deberta-v3-base-v2",
    ])

    sentiment_models_stack = [
        "edbeeching/gpt2_stack-exchange-paired_rmts__10000_2e-05_hub",
        "OpenAssistant/reward-model-deberta-v3-base",
        "OpenAssistant/reward-model-electra-large-discriminator",
        "OpenAssistant/reward-model-deberta-v3-large-v2",
        "theblackcat102/reward-model-deberta-v3-base-v2",
        # "edbeeching/gpt2_stack-exchange-paired_rmts_1000",
        # "trl-lib/llama-7b-se-rm-peft",
        # "kashif/llama-7b_stack-exchange_RM_peft-adapter-merged",
    ]


class DefaultArgs13b:
    base_model_name = "decapoda-research/llama-13b-hf"
    peft_name = "chansung/alpaca-lora-13b"


class DefaultArgsMerged:
    base_model_name = "alexrame/alpaca-lora-7b-merged"
    ref_model = "none"
    peft_name = "lora"



class DefaultArgsInference:
    seed = DefaultArgs.seed
    base_model_name = DefaultArgs.base_model_name
    sentiment_models = DefaultArgs.sentiment_models_review
    dataset_name = "default"
    peft_names = [DefaultArgs.peft_name]

    num_samples = 200
    every = 0
    verbose = 0
    output_max_length = DefaultArgs.output_max_length


class DefaultArgsInferenceMerged(DefaultArgsInference):
    base_model_name = DefaultArgsMerged.base_model_name
    peft_names = ["todo"]


class DefaultArgsInferenceSummary(DefaultArgsInference):
    sentiment_models = DefaultArgs.sentiment_models_summary

class DefaultArgsInferenceStack(DefaultArgsInference):
    sentiment_models = DefaultArgs.sentiment_models_stack

class DefaultArgsInferenceAssistant(DefaultArgsInference):
    sentiment_models = DefaultArgs.sentiment_models_assistant

class DefaultArgsInferenceSummaryMerged(DefaultArgsInferenceSummary):
    base_model_name = DefaultArgsMerged.base_model_name
    peft_names = ["todo"]


def get_peft_name(peft_name):
    if peft_name in [None, "None"]:
        return None
    if "/" not in peft_name:
        return "alexrame/" + peft_name
    if "epoch" in peft_name.split("/")[-1]:
        return  os.path.join(FOLDER_EXPE, peft_name)
    return peft_name

def get_args_inference(default_args):
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument(
        '--sentiment_models', type=str, nargs='+', default=default_args.sentiment_models
    )
    parser.add_argument('--base_model_name', type=str, default=default_args.base_model_name)
    parser.add_argument('--dataset_name', type=str, default=default_args.dataset_name)
    parser.add_argument('--peft_names', type=str, nargs='+', default=default_args.peft_names)
    parser.add_argument('--num_samples', type=int, default=default_args.num_samples)
    parser.add_argument('--every', type=int, default=default_args.every)
    parser.add_argument('--verbose', type=int, default=default_args.verbose)
    parser.add_argument('--seed', type=int, default=default_args.seed)
    parser.add_argument('--output_max_length', type=int, default=default_args.output_max_length)
    args = parser.parse_args()

    args.peft_names = [
        get_peft_name(peft_name) for peft_name in args.peft_names]
    args.myhost = os.uname()[1]

    return args

class DefaultArgsPPO:
    seed = DefaultArgs.seed
    sentiment_models = [DefaultArgs.sentiment_models_review][:1]

    base_model_name = DefaultArgs.base_model_name
    peft_name = DefaultArgs.peft_name
    ref_model = "newpeft"
    dataset_name = "default"

    output_min_length = DefaultArgs.output_max_length//2
    output_max_length = DefaultArgs.output_max_length
    score_goal = "0"
    log_with = ""
    mini_batch_size = 4
    gradient_accumulation_steps = 1
    batch_size = 128
    warmup_steps = 0
    learning_rate = 1.41e-5
    init_kl_coef = 0.2
    adap_kl_ctrl = 1
    num_epochs = 1


class DefaultArgsPPO13b(DefaultArgsPPO):
    base_model_name = DefaultArgs13b.base_model_name
    peft_name = DefaultArgs13b.peft_name


class DefaultArgsPPOMerged(DefaultArgsPPO):
    base_model_name = DefaultArgsMerged.base_model_name
    peft_name = DefaultArgsMerged.peft_name
    ref_model = DefaultArgsMerged.ref_model


def get_args_ppo(default_args):
    parser = argparse.ArgumentParser(description='Inference')

    parser.add_argument('--seed', type=int, default=default_args.seed)
    parser.add_argument('--sentiment_models', type=str, nargs='+', default=default_args.sentiment_models)
    parser.add_argument('--dataset_name', type=str, default=default_args.dataset_name)
    parser.add_argument('--base_model_name', type=str, default=default_args.base_model_name)
    parser.add_argument('--peft_name', type=str, default=default_args.peft_name)
    parser.add_argument('--ref_model', type=str, default=default_args.ref_model)


    parser.add_argument('--output_min_length', type=int, default=default_args.output_min_length)
    parser.add_argument('--output_max_length', type=int, default=default_args.output_max_length)
    parser.add_argument('--log_with', type=str, default="wandb")
    parser.add_argument('--score_goal', type=str, default=default_args.score_goal)

    parser.add_argument(
        '--mini_batch_size',
        type=int,
        default=default_args.mini_batch_size,
        help="Can be None or copy or alpaca"
    )

    parser.add_argument('--gradient_accumulation_steps', type=int, default=default_args.gradient_accumulation_steps)
    parser.add_argument('--batch_size', type=int, default=default_args.batch_size)
    parser.add_argument('--num_epochs', type=int, default=default_args.num_epochs)
    parser.add_argument('--learning_rate', type=float, default=default_args.learning_rate)
    parser.add_argument('--warmup_steps', type=float, default=default_args.warmup_steps)
    parser.add_argument('--init_kl_coef', type=float, default=default_args.init_kl_coef)
    parser.add_argument('--adap_kl_ctrl', type=int, default=default_args.adap_kl_ctrl)
    args = parser.parse_args()

    if "merged" in args.base_model_name:
        assert args.ref_model == "none"

    args.myhost = os.uname()[1]
    args.default_peft_name = DefaultArgs.peft_name if (args.base_model_name == DefaultArgs.base_model_name) else DefaultArgs13b.peft_name

    return args
