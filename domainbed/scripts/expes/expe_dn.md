

CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.train --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --algorithm ERM --dataset DomainNet --test_envs -1 --path_for_save /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/dn_erm012345_0921.pkl --steps -1 --train_only_classifier 1

CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.train --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --algorithm ERM --dataset DomainNet --test_envs -1 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/dn_erm012345_0921.pkl --path_for_save /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/dn_erm012345_lp_0921.pkl --train_only_classifier 1

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --output_dir=/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/dn_erm012345_lp_0921 --command_launcher multi_gpu --datasets DomainNet --algorithms ERM --test_envs -1 --n_hparams 20 --n_trials 1 --train_only_classifier 0 --save_model_every_checkpoint 0 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/dn_erm012345_lp_0921.pkl
