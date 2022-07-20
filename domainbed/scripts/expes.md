# CelebA

     JOBID PARTITION                           NAME     USER ST       TIME  NODES NODELIST(REASON)
    971512   gpu_p13   celeba2_llrdrohpl_0704.slurm  utr15kn PD       0:00      1 (Resources)
    971128   gpu_p13     celeba2_llma203_0704.slurm  utr15kn  R       9:25      1 r12i3n2
    971236   gpu_p13      celeba2_llrdro_0704.slurm  utr15kn  R       9:25      1 r13i5n4
    971334   gpu_p13    celeba2_llrfishr_0704.slurm  utr15kn  R       9:25      1 r14i3n3
    971471   gpu_p13        celeba2_gdro_0704.slurm  utr15kn  R       8:13      1 r10i1n6
    970358   gpu_p13    celeba2_ma203hpl_0704.slurm  utr15kn  R      24:56      1 r11i4n8
    970295   gpu_p13      celeba2_llr203_0704.slurm  utr15kn  R      25:56      1 r12i2n8


    971128   gpu_p13     celeba2_llma203_0704.slurm  utr15kn  R    3:47:30      1 r12i3n2
    971471   gpu_p13        celeba2_gdro_0704.slurm  utr15kn  R    3:46:18      1 r10i1n6
    970358   gpu_p13    celeba2_ma203hpl_0704.slurm  utr15kn  R    4:03:01      1 r11i4n8
    970295   gpu_p13      celeba2_llr203_0704.slurm  utr15kn  R    4:04:01      1 r12i2n8
    971878   gpu_p13     celeba2_llmahpl_0704.slurm  utr15kn  R    1:17:42      1 r10i5n7


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


