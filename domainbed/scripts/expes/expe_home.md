
Random init: /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_0822

SLP123: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0_ermll_saveall_si_0822
LP123: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0_ermll_saveall_si_0822/3d4174ccb9f69a3d872124137129aa1f/model.pkl

FT123alp: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0_erm_alp_0822
FT123: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0_ma_saveall_lp_0824

FTPerd123:/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home_ermwnr_lp_0901






TART_CHKPT_FREQ=10 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --output_dir=/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest8_ermwnr_saveall_lp_0827 --command_launcher multi_gpu --datasets OfficeHome --algorithms MA --train_env 0 --n_hparams 20 --n_trials 1 --train_only_classifier 0 --holdout_fraction 0.8 --steps 5000 --save_model_every_checkpoint 1 --path_for_init /gpfsdswork/projects/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest841_ermllr_saveall_0827/db66347321bed464bcf99408194a620a/model.pkl

