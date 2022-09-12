# train only on env0

DATA=/Users/alexandrerame/Dataplace python3 -m domainbed.scripts.train --data_dir /Users/alexandrerame/Dataplace/data --algorithm ERMask --dataset ColoredRotatedMNIST --test_envs 0 2 3 4 --path_for_init /Users/alexandrerame/Dataplace/experiments/domainbed/sorbonne/ColoredRotatedMNISTClean_ERMte0234_best.pkl

CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.train --data_dir /data/rame/data/domainbed --algorithm ERM --dataset ColoredRotatedMNIST --test_envs 1 2 3 --steps 0 --train_only_classifier 0


# classifier
DATA=/Users/alexandrerame/Dataplace MODEL_SELECTION=train WHICHMODEL=best INCLUDEVAL_UPTO=5 python3 -m domainbed.scripts.diwa --dataset ColoredRotatedMNISTClean --test_env 4  --output_dir no --trial_seed 0 --data_dir /Users/alexandrerame/Dataplace/data --what netm --checkpoints /Users/alexandrerame/Dataplace/experiments/domainbed/sorbonne/ColoredRotatedMNISTClean_ERMte0234_best.pkl 1 /Users/alexandrerame/Dataplace/experiments/domainbed/sorbonne/ColoredRotatedMNISTClean_ERMte0134_best.pkl 1

# masking
DATA=/Users/alexandrerame/Dataplace MODEL_SELECTION=train WHICHMODEL=best INCLUDEVAL_UPTO=5 python3 -m domainbed.scripts.diwa --dataset ColoredRotatedMNISTClean --test_env 4 --output_dir no --trial_seed 0 --data_dir /Users/alexandrerame/Dataplace/data --what netm cla mask --checkpoints /Users/alexandrerame/Dataplace/experiments/domainbed/sorbonne/ColoredRotatedMNISTClean_ERMaskte0234_best.pkl 1 /Users/alexandrerame/Dataplace/experiments/domainbed/sorbonne/ColoredRotatedMNISTClean_ERMaskte0134_best.pkl 1

MLFLOWEXPES_VERSION=nomlflow CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --output_dir=/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home1_erm_saveall_lpd023_0905 --command_launcher multi_gpu --datasets OfficeHome --algorithms ERM --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home_lpd023_0903 --n_hparams 20 --n_trials 1 --train_only_classifier 0 --test_env 1 --save_model_every_checkpoint 1


MLFLOWEXPES_VERSION=nomlflow HP=L CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.sweep launch --data_dir /data/rame/data/domainbed --output_dir=/data/rame/experiments/domainbed/crmnistc_erm0_s220911 --command_launcher multi_gpu --algorithms ERM SD --datasets ColoredRotatedMNISTClean --n_hparams 20 --n_trials 1 --train_env 0 --train_only_classifier 0




# folders

## ERM1 20k Hp=L wd

--checkpoints /data/rame/experiments/domainbed/singleruns/ColoredRotatedMNISTClean/ColoredRotatedMNISTClean_ERMte1234_bat64_cla0_dat1_lre001_non0_res0_res0_weie0001_11185256/model_best.pkl 1


## SD1 20k Hp=L

/data/rame/experiments/domainbed/singleruns/ColoredRotatedMNISTClean/ColoredRotatedMNISTClean_SDte1234_bat64_cla0_dat1_lre001_non0_res0_res0_sde1_wei0_11190916/model_best.pkl




MLFLOWEXPES_VERSION=nomlflow HP=L CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.sweep launch --data_dir /data/rame/data/domainbed --output_dir=/data/rame/experiments/domainbed/crmnistc_erm3_isd0_s220912 --command_launcher multi_gpu --algorithms ERM --datasets ColoredRotatedMNISTClean --n_hparams 20 --n_trials 1 --train_env 3 --train_only_classifier mask --steps 20001 --path_for_init /data/rame/experiments/domainbed/singleruns/ColoredRotatedMNISTClean/ColoredRotatedMNISTClean_SDte1234_bat64_cla0_dat1_lre001_non0_res0_res0_sde1_wei0_11190916/model_best.pkl



# 0->1
/data/rame/experiments/domainbed/singleruns/ColoredRotatedMNISTClean/ColoredRotatedMNISTClean_ERMte0234_bat64_cla0_dat1_lre001_non0_res0_res0_wei0_12144836/model_best.pkl
# 0->2
/data/rame/experiments/domainbed/singleruns/ColoredRotatedMNISTClean/ColoredRotatedMNISTClean_ERMte0134_bat64_cla0_dat1_lre001_non0_res0_res0_wei0_12144753/model_best.pkl
# 0->3
/data/rame/experiments/domainbed/singleruns/ColoredRotatedMNISTClean/ColoredRotatedMNISTClean_SDte1234_bat64_cla0_dat1_lre001_non0_res0_res0_sde1_wei0_11190916/model_best.pkl
