# Expes
/ ana
@ pending
+ doing
! todo
& later
? think about


# OfficeHome

## analyze performance of DIWA on one domain

### top1

enshome0_erm1_lp_0901_r0_top1.slurm /
enshome0_erm2_lp_0901_r0_top1.slurm /
enshome0_erm3_lp_0901_r0_top1.slurm /
enshome0_erm123_lp_0901_r0_top1.slurm /

### top1 with ma

todo ?
### top1 with robust

enshome0_erm123_lp_0901_r20_top1.slurm doing
todo ?

### m=20
enshome0_erm1_lp_0901_r0.slurm /
enshome0_erm2_lp_0901_r0.slurm /
enshome0_erm3_lp_0901_r0.slurm /


### m=20 with robust
enshome0_erm0_lp_0901_r20.slurm /
enshome0_erm1_lp_0901_r20.slurm /
enshome0_erm2_lp_0901_r20.slurm /

### m=20 with ma
enshome0_erm1_lp_0901_r0_ma.slurm doing

## compare avg of 3 vs ERM 3 domains

### model selection
- oracle
enshome0_ermt123_lp_0901_r0_top1_oracle.slurm_909082.out /
enshome0_ermt123_lp_0901_r0_top1_backup.slurm_908762.out /
enshome0_ermt123_lp_0901_r0_top-1.slurm_918818.out /

### robust
enshome0_ermt123_lp_0901_r20_top1.slurm doing

### moving average ?
enshome0_ermt123_lp_0901_r0_ma_top1.slurm doing


## compare avg of 20 * 3 runs vs diwa

enshome0_erm123_lp_0822_r0.slurm_890762.out /
enshome0_ermt123_lp_0901_r0.slurm_892403.out /
enshome0_ermt123_lp_0901_r0_top-7.slurm /


### combined with robust fine tuning ?

enshome0_erm123_lp_0822_r20.slurm done
vs.
enshome0_ermt123_lp_0901_r20.slurm done
vs.
enshome0_ermt123_lp_0901_r20_top-7.slurm done

## all combined

enshome0_erm123t123_lp_0901_r0.slurm +
enshome0_erm123t123_lp_0901_r20.slurm +

enshome0_erm123t123_lp_0901_r0_top1.slurm +
enshome0_erm123t123_lp_0901_r20_top1.slurm +


## retraining

enshome0h8_ermwnr_0901.slurm running
## need for linear probing ?


930026   gpu_p13 enshome0_ermt012_lp012_0901_r0  utr15kn PD       0:00      1 /
929851   gpu_p13 enshome0_ermt013_lp013_0901_r0  utr15kn  R       5:58      1 /
929782   gpu_p13 enshome0_ermt023_lp023_0901_r0  utr15kn  R       8:53      1 /
929743   gpu_p13 enshome0_ermt123_lp123_0901_r0  utr15kn  R       9:53      1 /
929926   gpu_p13 enshome0_ermt123_lp123_0901_r2  utr15kn  R       2:52      1 /

## training efficiency
enshome0_ermt123bs96_lp_0901_r0 todo
enshome0_ermt123bs96_lp_0901_r20 todo

### maybe try this for another domain ?

# Folders
Random init: /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_0822

SLP123: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0_ermll_saveall_si_0822
LP123: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0_ermll_saveall_si_0822/3d4174ccb9f69a3d872124137129aa1f/model.pkl

FT123alp: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0_erm_alp_0822
FT123: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0_ma_saveall_lp_0824

FTPerd123:/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home_ermwnr_lp_0901




TART_CHKPT_FREQ=10 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --output_dir=/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest8_ermwnr_saveall_lp_0827 --command_launcher multi_gpu --datasets OfficeHome --algorithms MA --train_env 0 --n_hparams 20 --n_trials 1 --train_only_classifier 0 --holdout_fraction 0.8 --steps 5000 --save_model_every_checkpoint 1 --path_for_init /gpfsdswork/projects/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest841_ermllr_saveall_0827/db66347321bed464bcf99408194a620a/model.pkl



home0h8_ermwnr_lp_0901.slurm
/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/homeh8_ermwnr_lp_0901

home0h8_ermwnr_0901.slurm
/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/homeh8_ermwnr_0901

home0h8_ermwnr_lp0_0901.slurm
/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/homeh8_ermwnr_lp0_0901
/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home_ermwnrbs96_lp_0901
