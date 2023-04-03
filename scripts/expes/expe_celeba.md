
# Expes

# folders





init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba01_lp_0901


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --output_dir=/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba_ermwnr_lp_0901 --command_launcher multi_gpu --datasets CelebA_Blond --algorithms MA --train_env 0 --n_hparams 20 --n_trials 1 --train_only_classifier 0 --save_model_every_checkpoint 1 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba01_lp_0901


/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba2_ermwnr_lp_0901



# old




WHICHMODEL=step INCLUDE_UPTO=3 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset CelebA_Blond --test_env 2 --train_envs 0 1 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba2_ermwnr_lp_0901 --trial_seed 0 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba01_lp_0901 0 --topk 0
WHICHMODEL=step INCLUDE_UPTO=3 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset CelebA_Blond --test_env 2 --train_envs 0 1 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba2_ermwnr_lp_0901 --trial_seed 0 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba01_lp_0901 0 --topk 0
WHICHMODEL=step INCLUDE_UPTO=3 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset CelebA_Blond --test_env 2 --train_envs 0 1 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba2_ermwnr_lp_0901 --trial_seed 0 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba01_lp_0901 0 --topk 0
WHICHMODEL=step INCLUDE_UPTO=3 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset CelebA_Blond --test_env 2 --train_envs 0 1 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba2_ermwnr_lp_0901 --trial_seed 0 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba01_lp_0901 0 --topk 0
WHICHMODEL=step INCLUDE_UPTO=3 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset CelebA_Blond --test_env 2 --train_envs 0 1 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba2_ermwnr_lp_0901 --trial_seed 0 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba01_lp_0901 0 --topk 0
WHICHMODEL=step INCLUDE_UPTO=3 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset CelebA_Blond --test_env 2 --train_envs 0 1 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba2_ermwnr_lp_0901 --trial_seed 0 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba01_lp_0901 0 --topk 0
WHICHMODEL=step INCLUDE_UPTO=3 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset CelebA_Blond --test_env 2 --train_envs 0 1 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba2_ermwnr_lp_0901 --trial_seed 0 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba01_lp_0901 0 --topk 0
WHICHMODEL=step INCLUDE_UPTO=3 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset CelebA_Blond --test_env 2 --train_envs 0 1 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba2_ermwnr_lp_0901 --trial_seed 0 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba01_lp_0901 0 --topk 0
WHICHMODEL=step INCLUDE_UPTO=3 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset CelebA_Blond --test_env 2 --train_envs 0 1 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba2_ermwnr_lp_0901 --trial_seed 0 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba01_lp_0901 0 --topk 0


MODEL_SELECTION=train WHICHMODEL=best INCLUDE_UPTO=3 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 2 --train_envs 0 1 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba2_ermwnr_lp_0901 --trial_seed 0 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba01_lp_0901 0 --topk 0

MODEL_SELECTION=train WHICHMODEL=best INCLUDE_UPTO=3 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 2 --train_envs 0 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba_ermwnr_lp_0901 --trial_seed 0 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba01_lp_0901 0 --topk 0

MODEL_SELECTION=train WHICHMODEL=best INCLUDE_UPTO=3 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 2 --train_envs 1 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba_ermwnr_lp_0901 --trial_seed 0 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba01_lp_0901 0 --topk 0



WHICHMODEL=best INCLUDE_UPTO=3 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 2 --train_envs 0 1 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba2_ermwnr_lp_0901 --trial_seed 0 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba01_lp_0901 0 --topk 0

WHICHMODEL=best INCLUDE_UPTO=3 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 2 --train_envs 0 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba_ermwnr_lp_0901 --trial_seed 0 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba01_lp_0901 0 --topk 0

WHICHMODEL=best INCLUDE_UPTO=3 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 2 --train_envs 1 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba_ermwnr_lp_0901 --trial_seed 0 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba01_lp_0901 0 --topk 0


WHICHMODEL=best INCLUDE_UPTO=3 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 2 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba_ermwnr_lp_0901 --trial_seed 0 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba01_lp_0901 0 --topk 0

WHICHMODEL=best INCLUDE_UPTO=3 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 2 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba2_ermwnr_lp_0901 /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba_ermwnr_lp_0901 --trial_seed 0 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba01_lp_0901 0 --topk 0


WHICHMODEL=step200 INCLUDE_UPTO=3 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 2 --train_envs 0 1 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba2_ermwnr_lp_0901 --trial_seed 0 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba01_lp_0901 0 --topk 0
WHICHMODEL=step400 INCLUDE_UPTO=3 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 2 --train_envs 0 1 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba2_ermwnr_lp_0901 --trial_seed 0 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba01_lp_0901 0 --topk 0
WHICHMODEL=step600 INCLUDE_UPTO=3 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 2 --train_envs 0 1 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba2_ermwnr_lp_0901 --trial_seed 0 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba01_lp_0901 0 --topk 0
WHICHMODEL=step800 INCLUDE_UPTO=3 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 2 --train_envs 0 1 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba2_ermwnr_lp_0901 --trial_seed 0 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba01_lp_0901 0 --topk 0
WHICHMODEL=step1000 INCLUDE_UPTO=3 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 2 --train_envs 0 1 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba2_ermwnr_lp_0901 --trial_seed 0 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba01_lp_0901 0 --topk 0
WHICHMODEL=step1200 INCLUDE_UPTO=3 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 2 --train_envs 0 1 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba2_ermwnr_lp_0901 --trial_seed 0 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba01_lp_0901 0 --topk 0
WHICHMODEL=step1400 INCLUDE_UPTO=3 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 2 --train_envs 0 1 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba2_ermwnr_lp_0901 --trial_seed 0 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba01_lp_0901 0 --topk 0
WHICHMODEL=step1600 INCLUDE_UPTO=3 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 2 --train_envs 0 1 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba2_ermwnr_lp_0901 --trial_seed 0 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba01_lp_0901 0 --topk 0
WHICHMODEL=step1800 INCLUDE_UPTO=3 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 2 --train_envs 0 1 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/celeba2_ermwnr_lp_0901 --trial_seed 0 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba01_lp_0901 0 --topk 0
