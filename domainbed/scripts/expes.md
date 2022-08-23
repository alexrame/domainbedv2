# current

    326397   gpu_p13       home0_erm_alp_0822.slurm  utr15kn PD       0:00      1 (Resources)
    326398   gpu_p13           home0_erm_0822.slurm  utr15kn PD       0:00      1 (Priority)
    326431   gpu_p13 enshomeontest8_ermwnr_saveall_  utr15kn PD       0:00      1 (Priority)

0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 4999
# toluanch

CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.train --algorithm ERM --dataset OfficeHome --test_env 0 --init_step --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_0822 --steps -1 --seed 822
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --output_dir=/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0_ermll_saveall_si_0822 --command_launcher multi_gpu --datasets OfficeHome --algorithms ERM --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_0822 --n_hparams 20 --n_trials 1 --train_only_classifier 1 --test_env 0 --save_model_every_checkpoint 1
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --output_dir=/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0_erm_alp_0822 --command_launcher multi_gpu --datasets OfficeHome --algorithms ERM --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0_ermll_saveall_si_0822/{hash}/model.pkl --n_hparams 20 --n_trials 1 --train_only_classifier 0 --test_env 0


INCLUDE_TRAIN=1 WHICHMODEL=step10 HOLDOUT=0.8 INDOMAIN=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest8_ermwnr_saveall_lpd_0822/ --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed





# backup
## WA peforms worse than ERM


## impact HP = L

## last layer training

## Save diwa weights and last layer retraining

* with erm
* with dro
* with vrex
* with fishr



CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset CelebA_Blond --test_env 2 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/celeba2_ma0320_0704 --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba2_0704_lp_diwa0



CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shhps0406home --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_0406_lp_diwa0



CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --output_dir=/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0_mallr_0707 --command_launcher multi_gpu --datasets OfficeHome --algorithms MA --test_env 0 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_0406_lp_diwa{trial_seed} --n_hparams 20 --n_trials 3 --skip_confirmation --train_only_classifier 1



KEYACC=Accuracies/acc_net CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --what netm
KEYACC=Accuracies/acc_net CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home --trial_seed 1 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --what netm
KEYACC=Accuracies/acc_net CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home --trial_seed 2 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --what netm


CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --data_dir /data/rame/data/domainbed/  --output_dir /data/rame/experiments/domainbed/celeba2_0320_ma_0704 --dataset CelebA_Blond --test_env 2 --trial_seed 0 --what netm


MAXM=5 KEYACC=Accuracies/acc_net CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --what var
MAXM=5 KEYACC=Accuracies/acc_net CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home --trial_seed 1 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --what var
MAXM=5 KEYACC=Accuracies/acc_net CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home --trial_seed 2 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --what var



VARM=0 MAXM=5 KEYACC=Accuracies/acc_net CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --what var
VARM=0.5 MAXM=5 KEYACC=Accuracies/acc_net CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --what var
VARM=1 MAXM=5 KEYACC=Accuracies/acc_net CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --what var
VARM=2 MAXM=5 KEYACC=Accuracies/acc_net CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --what var
VARM=4 MAXM=5 KEYACC=Accuracies/acc_net CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --what var
VARM=8 MAXM=5 KEYACC=Accuracies/acc_net CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --what var
VARM=16 MAXM=5 KEYACC=Accuracies/acc_net CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --what var
VARM=32 MAXM=5 KEYACC=Accuracies/acc_net CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --what var


CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.train --data_dir /data/rame/data/domainbed/ --dataset ColoredMNIST --algorithm DARE --test_env 2 --train_only_classifier 1



RESET_CLASSIFIER=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset CelebA_Blond --test_env 2 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/celeba2_ma0320_0704 --trial_seed 0 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba2_0704_lpreset_diwa0
RESET_CLASSIFIER=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset CelebA_Blond --test_env 2 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/celeba2_ma0320_0704 --trial_seed 1 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba2_0704_lpreset_diwa1
RESET_CLASSIFIER=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset CelebA_Blond --test_env 2 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/celeba2_ma0320_0704 --trial_seed 2 --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/celeba2_0704_lpreset_diwa2



KEYACC=Accuracies/acc_net CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --what netm --path_for_save_weight /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_0722_lp_diwa2
KEYACC=Accuracies/acc_net CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home --trial_seed 1 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --what netm --path_for_save_weight /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_0722_lp_diwa2
KEYACC=Accuracies/acc_net CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home --trial_seed 2 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --what netm --path_for_save_weight /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_0722_lp_diwa2




INCLUDE_TRAIN=1 HOLDOUT=0.2 KEEP_ALL_ENV=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest_ermwnr_lpd_0731 --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed
INCLUDE_TRAIN=1 HOLDOUT=0.2 KEEP_ALL_ENV=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest_ermwnr_lpd_0731 --trial_seed 1 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed
INCLUDE_TRAIN=1 HOLDOUT=0.2 KEEP_ALL_ENV=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest_ermwnr_lpd_0731 --trial_seed 2 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed

INCLUDE_TRAIN=1 HOLDOUT=0.2 KEEP_ALL_ENV=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest_ermwnr_lpe_0731 --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed
INCLUDE_TRAIN=1 HOLDOUT=0.2 KEEP_ALL_ENV=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest_ermwnr_lpe_0731 --trial_seed 1 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed
INCLUDE_TRAIN=1 HOLDOUT=0.2 KEEP_ALL_ENV=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest_ermwnr_lpe_0731 --trial_seed 2 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed

INCLUDE_TRAIN=1 HOLDOUT=0.2 KEEP_ALL_ENV=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest_ermwnr_0731 --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed
INCLUDE_TRAIN=1 HOLDOUT=0.2 KEEP_ALL_ENV=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest_ermwnr_0731 --trial_seed 1 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed
INCLUDE_TRAIN=1 HOLDOUT=0.2 KEEP_ALL_ENV=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest_ermwnr_0731 --trial_seed 2 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed



/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/

/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/


