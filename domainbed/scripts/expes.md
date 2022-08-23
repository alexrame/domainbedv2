



# current

     JOBID PARTITION                           NAME     USER ST       TIME  NODES NODELIST(REASON)
    356250   gpu_p13       home0_erm_alp_0822.slurm  utr15kn PD       0:00      1 (Priority)
    353629   gpu_p13           home0_erm_0823.slurm  utr15kn  R      49:59      1 r11i6n8
    355887   gpu_p13 enshomeontest8_ermwnr_saveall_  utr15kn  R      30:14      1 r10i1n2
    355883   gpu_p13 enshomeontest8_ermwnr_saveall_  utr15kn  R      30:53      1 r10i7n1
    355884   gpu_p13 enshomeontest8_ermwnr_saveall_  utr15kn  R      30:53      1 r11i0n1
    355896   gpu_p13 enshomeontest8_ermwnr_saveall_  utr15kn  R      29:39      1 r10i0n2
    356100   gpu_p13 enshomeontest8_ermllr_saveall_  utr15kn  R      21:06      1 r10i7n4
    356101   gpu_p13 enshomeontest8_ermllr_saveall_  utr15kn  R      21:06      1 r11i0n8
    356113   gpu_p13 enshomeontest8_ermllr_saveall_  utr15kn  R      20:06      1 r10i2n6
    356115   gpu_p13 enshomeontest8_ermllr_saveall_  utr15kn  R      20:06      1 r10i5n5
    356597   gpu_p13      enshome_erm_lp_0406.slurm  utr15kn  R       0:09      1 r7i7n7
    356606   gpu_p13 enshomeontest8_ermwnr_saveall_  utr15kn  R       0:09      1 r10i5n1

# toluanch

CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.train --algorithm ERM --dataset OfficeHome --test_env 0 --init_step --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_0822 --steps -1 --seed 822
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --output_dir=/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0_ermll_saveall_si_0822 --command_launcher multi_gpu --datasets OfficeHome --algorithms ERM --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_0822 --n_hparams 20 --n_trials 1 --train_only_classifier 1 --test_env 0 --save_model_every_checkpoint 1
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --output_dir=/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0_erm_alp_0822 --command_launcher multi_gpu --datasets OfficeHome --algorithms ERM --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0_ermll_saveall_si_0822/{hash}/model.pkl --n_hparams 20 --n_trials 1 --train_only_classifier 0 --test_env 0


INCLUDE_TRAIN=1 WHICHMODEL=step10 HOLDOUT=0.8 INDOMAIN=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest8_ermwnr_saveall_lpd_0822/ --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed





#
MODEL_SELECTION=oracle WHICHMODEL=step100 INCLUDE_TRAIN=1 HOLDOUT=0.8 INDOMAIN=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest8_ermwnr_saveall_lpd_0822/ --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_0406_lp_diwa0 0
MODEL_SELECTION=oracle WHICHMODEL=step100 INCLUDE_TRAIN=1 HOLDOUT=0.8 INDOMAIN=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest8_ermwnr_saveall_lpd_0822/ --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_0406_lp_diwa0 1
MODEL_SELECTION=oracle WHICHMODEL=step100 INCLUDE_TRAIN=1 HOLDOUT=0.8 INDOMAIN=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest8_ermwnr_saveall_lpd_0822/ --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_0406_lp_diwa0 5
MODEL_SELECTION=oracle WHICHMODEL=step100 INCLUDE_TRAIN=1 HOLDOUT=0.8 INDOMAIN=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest8_ermwnr_saveall_lpd_0822/ --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_0406_lp_diwa0 20
MODEL_SELECTION=oracle WHICHMODEL=step100 INCLUDE_TRAIN=1 HOLDOUT=0.8 INDOMAIN=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest8_ermwnr_saveall_lpd_0822/ --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_0406_lp_diwa0 1000000




home0ontest8_ermwnr_saveall_lpd_0822


INCLUDE_TRAIN=1 MODEL_SELECTION=oracle KEYACC=out_Accuracies/acc_net CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_lp_0401 1
INCLUDE_TRAIN=1 MODEL_SELECTION=oracle KEYACC=out_Accuracies/acc_net CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_lp_0401 5
INCLUDE_TRAIN=1 MODEL_SELECTION=oracle KEYACC=out_Accuracies/acc_net CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_lp_0401 10
INCLUDE_TRAIN=1 MODEL_SELECTION=oracle KEYACC=out_Accuracies/acc_net CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_lp_0401 15
INCLUDE_TRAIN=1 MODEL_SELECTION=oracle KEYACC=out_Accuracies/acc_net CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_lp_0401 20
INCLUDE_TRAIN=1 MODEL_SELECTION=oracle KEYACC=out_Accuracies/acc_net CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_lp_0401 25
INCLUDE_TRAIN=1 MODEL_SELECTION=oracle KEYACC=out_Accuracies/acc_net CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_lp_0401 30
INCLUDE_TRAIN=1 MODEL_SELECTION=oracle KEYACC=out_Accuracies/acc_net CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_lp_0401 40


