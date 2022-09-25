

CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.train --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --algorithm ERM --dataset DomainNet --test_envs -1 --path_for_save /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/dn_erm012345_0921.pkl --steps -1 --train_only_classifier 1

CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.train --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --algorithm ERM --dataset DomainNet --test_envs -1 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/dn_erm012345_0921.pkl --path_for_save /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/dn_erm012345_lp_0921.pkl --train_only_classifier 1

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --output_dir=/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/dn_erm012345_lp_0921 --command_launcher multi_gpu --datasets DomainNet --algorithms ERM --test_envs -1 --n_hparams 20 --n_trials 1 --train_only_classifier 0 --save_model_every_checkpoint 0 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/dn_erm012345_lp_0921.pkl


python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_lp_0401_65 0 --what netm


SAVE_ONLY_FEATURES=1 MODEL_SELECTION=oracle WHICHMODEL=stepbest INCLUDEVAL_UPTO=6 KEYACC=out_Accuracies/acc_net CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --data_dir /data/rame/data/domainbed/ --dataset DomainNet --test_envs 1 --output_dir /data/rame/ermdomainnet0425 --trial_seed 0 --checkpoints /data/rame/init_diwa/inits/domainnet1_lp_0424_345 0 --path_for_init /data/rame/inits/transfer/dn1_erm02345_lp_0921.pkl --what feats



dn1_erm02345_lp_r0_0921.pkl
