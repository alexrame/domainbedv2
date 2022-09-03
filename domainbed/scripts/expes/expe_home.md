
Random init: /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_0822

SLP123: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0_ermll_saveall_si_0822
LP123: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0_ermll_saveall_si_0822/3d4174ccb9f69a3d872124137129aa1f/model.pkl

FT123alp: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0_erm_alp_0822
FT123: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0_ma_saveall_lp_0824

FTPerd123:/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home_ermwnr_lp_0901






TART_CHKPT_FREQ=10 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --output_dir=/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest8_ermwnr_saveall_lp_0827 --command_launcher multi_gpu --datasets OfficeHome --algorithms MA --train_env 0 --n_hparams 20 --n_trials 1 --train_only_classifier 0 --holdout_fraction 0.8 --steps 5000 --save_model_every_checkpoint 1 --path_for_init /gpfsdswork/projects/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest841_ermllr_saveall_0827/db66347321bed464bcf99408194a620a/model.pkl

# OfficeHome

## analyze performance of DIWA on one domain

### top1

enshome0_erm1_lp_0901_r0_top1.slurm
enshome0_erm2_lp_0901_r0_top1.slurm
enshome0_erm3_lp_0901_r0_top1.slurm

### m=20
enshome0_erm1_lp_0901_r0.slurm
enshome0_erm2_lp_0901_r0.slurm
enshome0_erm3_lp_0901_r0.slurm

### m=20 with ma
enshome0_erm1_lp_0901_r0_ma.slurm


## compare avg of 3 vs ERM 3 domains





### model selection
- oracle
enshome0_ermt123_lp_0901_r0_top1_oracle.slurm_909082.out
- training
enshome0_ermt123_lp_0901_r0_top1_backup.slurm_908762.out
- random
enshome0_ermt123_lp_0901_r0_top-1.slurm_918818.out

=> conclusion; Embarrassingly Parallel Training
better or similar results than ERM

### moving average ?
enshome0_ermt123_lp_0901_r0_ma_top1.slurm


## compare avg of 20 * 3 runs vs


enshome0_erm123_lp_0822_r0.slurm_890762.out

vs.

enshome0_ermt123_lp_0901_r0.slurm_892403.out
### retraining
home0h8_ermwnr_lp_0901.slurm
home0h8_ermwnr_0901.slurm
home0h8_ermwnr_lp0_0901.slurm

by modifying enshomeontest8_ermwnr_saveall_lpd_0827_robust0.slurm


### all combined

enshome0_erm123t123_lp_0901_r0.slurm
enshome0_erm123t123_lp_0901_r20.slurm
### combined with robust fine tuning ?

enshome0_erm123_lp_0822_r20.slurm
vs.
enshome0_ermt123_lp_0901_r60.slurm



### training efficiency
analyze home1_ermwnrbs96_lp_0901.slurm


# CelebA

Analyze ratio across domains
# Open questions
## need for linear probing ?

##
