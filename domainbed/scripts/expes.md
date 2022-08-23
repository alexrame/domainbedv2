



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





#
MODEL_SELECTION=oracle WHICHMODEL=step100 INCLUDE_TRAIN=1 HOLDOUT=0.8 INDOMAIN=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest8_ermwnr_saveall_lpd_0822/ --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_0406_lp_diwa0 0
MODEL_SELECTION=oracle WHICHMODEL=step100 INCLUDE_TRAIN=1 HOLDOUT=0.8 INDOMAIN=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest8_ermwnr_saveall_lpd_0822/ --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_0406_lp_diwa0 1
MODEL_SELECTION=oracle WHICHMODEL=step100 INCLUDE_TRAIN=1 HOLDOUT=0.8 INDOMAIN=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest8_ermwnr_saveall_lpd_0822/ --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_0406_lp_diwa0 5
MODEL_SELECTION=oracle WHICHMODEL=step100 INCLUDE_TRAIN=1 HOLDOUT=0.8 INDOMAIN=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest8_ermwnr_saveall_lpd_0822/ --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_0406_lp_diwa0 20
MODEL_SELECTION=oracle WHICHMODEL=step100 INCLUDE_TRAIN=1 HOLDOUT=0.8 INDOMAIN=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest8_ermwnr_saveall_lpd_0822/ --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_0406_lp_diwa0 1000000




home0ontest8_ermwnr_saveall_lpd_0822
