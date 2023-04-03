
    sentiment_models_review = [
        "lvwerra/distilbert-imdb",
        "distilbert-base-uncased-finetuned-sst-2-english",
        "martin-ha/toxic-comment-model",
        "valurank/distilbert-quality",
        "OpenAssistant/reward-model-deberta-v3-large-v2",
        "OpenAssistant/reward-model-deberta-v3-base",
        "OpenAssistant/reward-model-electra-large-discriminator",
        "sugam11/gpt2-rlhf-reward",
    ]

    sentiment_models_summary = [
        "Tristan/gpt2_reward_summarization",
        "CogComp/bart-faithful-summary-detector"  #  True, label mapping: "0" -> "Hallucinated" "1" -> "Faithful"
    ]


# running

## summary news
python3 llama_ppo_summary.py --sentiment_model CogComp/bart-faithful-summary-detector --score_goal "1-0" --dataset_name news


## captioning

python train.py --optim_type radam --seed 775533 --sched_type custom_warmup_anneal --warmup 1 --anneal_coeff 1.0 --lr 5e-6 --batch_size 18 --num_accum 2 --is_end_to_end True --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/ --body_save_path ${DATA_DIR}/saves/rf_model-002.pth --num_epochs 8 --save_path ${DATA_DIR}/saves/e2ebleumeteorbs18lr5e-6/ --reinforce bleu,meteor


# tolaunch

FIXED=0 python3 llama_ppo_review.py --sentiment_models OpenAssistant/reward-model-deberta-v3-base OpenAssistant/reward-model-electra-large-discriminator --score_goal 0x1_0x1

python3 llama_ppo_summary.py --sentiment_model Tristan/gpt2_reward_summarization --score_goal 0 --dataset_name news


python3 llama_ppo_review.py --sentiment_model martin-ha/toxic-comment-model --score_goal 0
python3 llama_ppo_review.py --sentiment_model martin-ha/toxic-comment-model --score_goal 1
python3 llama_ppo_review.py --sentiment_model lvwerra/distilbert-imdb --score_goal "1-0"
python3 llama_ppo_review.py --sentiment_model valurank/distilbert-quality --score_goal "2-0"

python3 llama_ppo_review.py --sentiment_model valurank/distilbert-quality --score_goal "2-0"

python3 llama_ppo_review.py --sentiment_model martin-ha/toxic-comment-model --score_goal 1

# summary news
python3 llama_ppo_summary.py --sentiment_model Tristan/gpt2_reward_summarization --score_goal 0 --dataset_name news


# summary

# tocode
## code rlhf

dataset: Anthropic/hh-rlhf
assistants as rewards

