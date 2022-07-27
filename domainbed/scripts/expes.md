# current
(bias) [utr15kn@jean-zay1: home0707]$ scid
     JOBID PARTITION                           NAME     USER ST       TIME  NODES NODELIST(REASON)
   1515657   gpu_p13 home0_dareswapllr_lpdr_0722.sl  utr15kn PD       0:00      1 (Priority)
   1514156   gpu_p13 home0_fishrllr_lpdr_0722.slurm  utr15kn  R    3:03:09      1 r12i1n3
   1514153   gpu_p13    home0_ermllr_lpd_0722.slurm  utr15kn  R    3:03:31      1 r11i0n0
   1514148   gpu_p13   home0_ermllr_lpdr_0722.slurm  utr15kn  R    3:03:55      1 r10i7n5
   1515589   gpu_p13  home0_darellr_lpdr_0722.slurm  utr15kn  R       8:42      1 r11i4n7

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
