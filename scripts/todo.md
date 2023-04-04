# captioning

** running
     JOBID PARTITION                           NAME     USER ST       TIME  NODES NODELIST(REASON)
    570694   gpu_p13     jz_inf_fts_rougebleu.slurm  utr15kn  R       1:03      1 r7i0n0
    570784    gpu_p4           fts_rougebleu4.slurm  utr15kn  R       0:02      1 jean-zay-a103
    570774    gpu_p4            fts_bleubleu4.slurm  utr15kn  R       0:52      1 jean-zay-a103

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

