# captioning

** running
     JOBID PARTITION                           NAME     USER ST       TIME  NODES NODELIST(REASON)
    570694   gpu_p13     jz_inf_fts_rougebleu.slurm  utr15kn  R       1:03      1 r7i0n0
    570784    gpu_p4           fts_rougebleu4.slurm  utr15kn  R       0:02      1 jean-zay-a103
    570774    gpu_p4            fts_bleubleu4.slurm  utr15kn  R       0:52      1 jean-zay-a103
/data/rame/logs_experiments_notebook/captioning/e2e_bleumeteor.py

# review

python3 llama_inference_review.py --num_samples 200 --peft_names tloen/alpaca-lora-7b /data/rame/experiments/nlp/llama-7b-hf-ppo-imdb-reward-model-dvb-g0-pmedium-03-30-1680170161/epoch191/

python3 llama_inference_review.py --num_samples 200 --peft_names "/data/rame/experiments/nlp/llama-7b-hf-ppo-imdb-reward-model-dvb-g0-pmedium-03-30-1680170161/epoch*"

python3 llama_inference_review.py --num_samples 200 --peft_names tloen/alpaca-lora-7b /data/rame/experiments/nlp/llama-7b-hf-ppo-imdb-reward-model-eld-g0-03-30-1680173234/epoch191/

python3 llama_inference_review.py --num_samples 200 --peft_names /data/rame/experiments/nlp/llama-7b-hf-ppo-imdb-reward-model-dvb-g0-pmedium-03-30-1680170161/epoch191/ /data/rame/experiments/nlp/llama-7b-hf-ppo-imdb-reward-model-eld-g0-03-30-1680173234/epoch191/

# summary news

** running

python3 llama_ppo_summary.py --sentiment_model CogComp/bart-faithful-summary-detector --score_goal "1-0" --dataset_name news


** todo


# summary

** doing

python3 llama_ppo_summary.py --sentiment_models CogComp/bart-faithful-summary-detector Tristan/gpt2_reward_summarization --score_goal '1-0x1_0x1' --init_kl_coef 0.05

# toxic

** doing

python3 llama_ppo_review.py --sentiment_model unitary/toxic-bert --score_goal 0


python3 llama_inference_review.py --num_samples 200 --peft_names /data/rame/experiments/nlp/llama-7b-hf-ppo-review-tcm-g0-04-02-1680463194/epoch91/ /data/rame/experiments/nlp/llama-7b-hf-ppo-review-tb-g-0-04-03-1680546735/epoch91/

** todo

python3 llama_inference_review.py --num_samples 200 --peft_names /data/rame/experiments/nlp/llama-7b-hf-ppo-review-tcm-g0-04-02-1680463194/epoch191/ /data/rame/experiments/nlp/llama-7b-hf-ppo-review-tb-g-0-04-03-1680546735/epoch191/

# positiv

** doing

python3 llama_inference_review.py --num_samples 30 --peft_names /data/rame/experiments/nlp/llama-7b-hf-ppo-review-g1-0-04-02-1680428653/epoch91/ /data/rame/experiments/nlp/llama-7b-hf-ppo-review-dbufs2e-g1-0-04-03-1680545803/epoch91/

# tocode

## code rlhf

dataset: Anthropic/hh-rlhf
assistants as rewards



# tolaunch



python3 llama_ppo_summary.py --sentiment_model Tristan/gpt2_reward_summarization --score_goal 0 --dataset_name news --init_kl_coef 0.05 --num_epochs 3
llama-7b-hf-ppo-summary-g-g0-dnews-04-03-1680542051 at: https://wandb.ai/alexrame/trl/runs/cr4apvyl


wandb: 🚀 View run llama-7b-hf-ppo-summary-bfsd-g1-0-lr5e-06-dnews-04-04-1680608847 at: https://wandb.ai/alexrame/trl/runs/asgkzwmz
wandb: Synced 6 W&B file(s), 54 media file(s), 54 artifact file(s) and 0 other file(s)
wandb: Find logs at: /data/rame/experiments/wandb/wandb/run-20230404_134805-asgkzwmz/logs
(nlp) rame@zz:~/trl/examples/llama$ python3 llama_ppo_summary.py --sentiment_model CogComp/bart-faithful-summary-detector --score_goal "1-0" --dataset_name news --init_kl_coef 0.05 --num_epochs 1 --output_min_length 31 --learning_rate 5e-6
