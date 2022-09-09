# train only on env0

CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.train --data_dir /data/rame/data/domainbed --algorithm ERM --dataset ColoredRotatedMNIST --test_envs 1 2 3 --steps 0 --train_only_classifier 0


# standard protocol for sweep

CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.train --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --algorithm ERM --dataset PACS --test_envs 0 --path_for_save /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/pacs_0906 --steps -1 --seed 906 --train_only_classifier 1
CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.train --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --algorithm ERM --dataset PACS --test_envs 1 2 3 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/pacs_0906 --path_for_save /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/pacs_lp0_906 --seed 906 --train_only_classifier 1
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --output_dir=/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/pacs_erm0wnr_lp0_0906 --command_launcher multi_gpu --datasets PACS --algorithms ERM --train_env 0 --n_hparams 20 --n_trials 1 --train_only_classifier 0 --steps 5000 --save_model_every_checkpoint 1 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/pacs_lp0_906
