✓ done
+ doing
! todo
& later
? think about


# OfficeHome

## analyze performance of DIWA on one domain

### top1

enshome0_erm1_lp_0901_r0_top1.slurm
enshome0_erm2_lp_0901_r0_top1.slurm
enshome0_erm3_lp_0901_r0_top1.slurm
vs.
enshome0_erm123_lp_0901_r0_top1.slurm

### m=20
enshome0_erm1_lp_0901_r0.slurm
enshome0_erm2_lp_0901_r0.slurm
enshome0_erm3_lp_0901_r0.slurm

### m=20 with ma
enshome0_erm1_lp_0901_r0_ma.slurm

### m=20 with robust
enshome0_erm2_lp_0901_r20.slurm

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


## compare avg of 20 * 3 runs vs diwa


enshome0_erm123_lp_0822_r0.slurm_890762.out

vs.

enshome0_ermt123_lp_0901_r0.slurm_892403.out

vs.
enshome0_ermt123_lp_0901_r0_top-7.slurm doing


### combined with robust fine tuning ?

enshome0_erm123_lp_0822_r20.slurm
vs.
enshome0_ermt123_lp_0901_r20.slurm doing

## all combined

enshome0_erm123t123_lp_0901_r0.slurm doing
enshome0_erm123t123_lp_0901_r80.slurm doing

enshome0_erm123t123_lp_0901_r0_top1.slurm doing
enshome0_erm123t123_lp_0901_r4_top1.slurm doing


## retraining
home0h8_ermwnr_lp_0901.slurm
home0h8_ermwnr_0901.slurm
home0h8_ermwnr_lp0_0901.slurm

by modifying enshomeontest8_ermwnr_saveall_lpd_0827_robust0.slurm


## training efficiency
analyze home1_ermwnrbs96_lp_0901.slurm


# CelebA

Analyze ratio across domains
# Open questions
## need for linear probing ?

##
