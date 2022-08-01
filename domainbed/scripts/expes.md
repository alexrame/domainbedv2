# debug

"env0_in_acc": 0.23892893923789907
"env0_out_acc": 0.2618556701030928
"env1_in_acc": 0.2571592210767468
"env1_out_acc": 0.23367697594501718
"env2_in_acc": 0.3761261261261261
"env2_out_acc": 0.35851183765501693
"env3_in_acc": 0.36058519793459554
"env3_out_acc": 0.338691159586682,


# current
   1748886   gpu_p13 enshomeontest8_erm_lpdr_0731.s  utr15kn PD       0:00      1 (None)
   1748864   gpu_p13 enshomeontest8_ermwnr_lp_0731.  utr15kn  R       0:53      1 r12i4n8
   1748858   gpu_p13 enshomeontest_ermwnr_lp_0731.s  utr15kn  R       1:13      1 r11i4n7
   1748859   gpu_p13   enshomeontest8_lp_0727.slurm  utr15kn  R       1:13      1 r11i4n7
   1748855   gpu_p13    enshomeontest_lp_0727.slurm  utr15kn  R       1:27      1 r11i0n7
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


