# train only on env0

DATA=/Users/alexandrerame/Dataplace python3 -m domainbed.scripts.train --data_dir /Users/alexandrerame/Dataplace/data --algorithm ERMask --dataset ColoredRotatedMNIST --test_envs 0 2 3 4 --path_for_init /Users/alexandrerame/Dataplace/experiments/domainbed/sorbonne/ColoredRotatedMNISTClean_ERMte0234_best.pkl

CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.train --data_dir /data/rame/data/domainbed --algorithm ERM --dataset ColoredRotatedMNIST --test_envs 1 2 3 --steps 0 --train_only_classifier 0



for EPOCH in best
do
        MODEL_SELECTION=train WHICHMODEL=step$EPOCH INCLUDEVAL_UPTO=4 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset ColoredRotatedMNISTClean --test_env 4  --output_dir no --trial_seed 0 --data_dir /data/rame/data/domainbed  0 --what cla netm --checkpoints /data/rame/experiments/domainbed/singleruns/ColoredRotatedMNISTClean/ColoredRotatedMNISTClean_ERMte0234_bat32_cla0_dat1_lr5e-05_non0_res0_res0_wei0_09164650/model_best.pkl 0 /data/rame/experiments/domainbed/singleruns/ColoredRotatedMNISTClean/ColoredRotatedMNISTClean_ERMte0134_bat32_cla0_dat1_lr5e-05_non0_res0_res0_wei0_09164620/model_best.pkl 0
done


# standard protocol for sweep

CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.train --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --algorithm ERM --dataset PACS --test_envs 0 --path_for_save /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/pacs_0906 --steps -1 --seed 906 --train_only_classifier 1
CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.train --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --algorithm ERM --dataset PACS --test_envs 1 2 3 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/pacs_0906 --path_for_save /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/pacs_lp0_906 --seed 906 --train_only_classifier 1
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --output_dir=/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/pacs_erm0wnr_lp0_0906 --command_launcher multi_gpu --datasets PACS --algorithms ERM --train_env 0 --n_hparams 20 --n_trials 1 --train_only_classifier 0 --steps 5000 --save_model_every_checkpoint 1 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/pacs_lp0_906



for EPOCH in 100 200 400 600 800 1000 1200 1400 1600 1800 2000 2200 2400 2600 2800 3000 3200 3400 3600 3800 4000 4200 4400 4600 4800 best
do
        MODEL_SELECTION=train WHICHMODEL=step$EPOCH INCLUDEVAL_UPTO=4 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0  --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0_ma_saveall_lp_0824 --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0_ermll_saveall_si_0822/3d4174ccb9f69a3d872124137129aa1f/model.pkl 0 --what cla
done

DATA=/Users/alexandrerame/Dataplace MODEL_SELECTION=train WHICHMODEL=best INCLUDEVAL_UPTO=5 python3 -m domainbed.scripts.diwa --dataset ColoredRotatedMNISTClean --test_env 4  --output_dir no --trial_seed 0 --data_dir /Users/alexandrerame/Dataplace/data --what netm --checkpoints /Users/alexandrerame/Dataplace/experiments/domainbed/sorbonne/ColoredRotatedMNISTClean_ERMte0234_best.pkl 1 /Users/alexandrerame/Dataplace/experiments/domainbed/sorbonne/ColoredRotatedMNISTClean_ERMte0134_best.pkl 1

