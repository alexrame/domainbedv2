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

### top1 with robust

enshome0_erm123_lp_0901_r20_top1.slurm /

### m=20
enshome0_erm1_lp_0901_r0.slurm /
enshome0_erm2_lp_0901_r0.slurm /
enshome0_erm3_lp_0901_r0.slurm /


### m=20 with robust
enshome0_erm0_lp_0901_r20.slurm /
enshome0_erm1_lp_0901_r20.slurm /
enshome0_erm2_lp_0901_r20.slurm /

### m=20 with ma
enshome0_erm1_lp_0901_r0_ma.slurm /

## compare avg of 3 vs ERM 3 domains

### model selection
- oracle
enshome0_ermt123_lp_0901_r0_top1_oracle.slurm_909082.out /
enshome0_ermt123_lp_0901_r0_top1_backup.slurm_908762.out /
enshome0_ermt123_lp_0901_r0_top-1.slurm_918818.out /

### robust
enshome0_ermt123_lp_0901_r20_top1.slurm /

### moving average ?
enshome0_ermt123_lp_0901_r0_ma_top1.slurm /


## compare avg of 20 * 3 runs vs diwa

enshome0_erm123_lp_0822_r0.slurm_890762.out /
enshome0_ermt123_lp_0901_r0.slurm_892403.out /
enshome0_ermt123_lp_0901_r0_top-7.slurm /


### combined with robust fine tuning ?

enshome0_erm123_lp_0822_r20.slurm /
enshome0_ermt123_lp_0901_r20.slurm /
enshome0_ermt123_lp_0901_r20_top-7.slurm /

## all combined

enshome0_erm123t123_lp_0901_r0.slurm /
enshome0_erm123t123_lp_0901_r20.slurm /

enshome0_erm123t123_lp_0901_r0_top1.slurm /
enshome0_erm123t123_lp_0901_r20_top1.slurm /


## retraining

enshome0h8_ermwnr_0901.slurm /
## need for linear probing ?


930026   gpu_p13 enshome0_ermt012_lp012_0901_r0  utr15kn PD       0:00      1 /
929851   gpu_p13 enshome0_ermt013_lp013_0901_r0  utr15kn  R       5:58      1 /
929782   gpu_p13 enshome0_ermt023_lp023_0901_r0  utr15kn  R       8:53      1 /
929743   gpu_p13 enshome0_ermt123_lp123_0901_r0  utr15kn  R       9:53      1 /
929926   gpu_p13 enshome0_ermt123_lp123_0901_r2  utr15kn  R       2:52      1 /

## training efficiency
enshome0_ermt123bs96_lp_0901_r0 /
enshome0_ermt123bs96_lp_0901_r20 /

## home1 (contrairement au nom indiqué)
enshome0_ermt023_lp023_0901_r0
enshome0_erm023t023_lp023_0901
enshome0_erm023t023_lp023_0901
enshome0_ermt023_lp023_0901_r2
enshome0_erm023_lp_0901_r20


# transfer abilities across domains
     JOBID PARTITION                           NAME     USER ST       TIME  NODES NODELIST(REASON)
   1013625   gpu_p13 enshome_ermt3_lp3_0901_r0_top1  utr15kn PD       0:00      1 (None)
   1013624   gpu_p13 enshome_ermt3_lp3_0901_r0.slur  utr15kn  R       0:09      1 r12i1n7
   1013619   gpu_p13 enshome_ermt2_lp2_0901_r0.slur  utr15kn  R       1:11      1 r12i3n3
   1013617   gpu_p13 enshome_ermt2_lp2_0901_r0_top1  utr15kn  R       1:16      1 r12i1n5
   1013600   gpu_p13 enshome_ermt1_lp1_0901_r0_top1  utr15kn  R       2:01      1 r11i6n1
   1013601   gpu_p13 enshome_ermt1_lp1_0901_r0.slur  utr15kn  R       2:01      1 r11i6n8
   1013560   gpu_p13 enshome_ermt0_lp0_0901_r0_top1  utr15kn  R       5:12      1 r10i1n1
   1013562   gpu_p13 enshome_ermt0_lp0_0901_r0.slur  utr15kn  R       5:12      1 r10i3n7


# rich features construction by specialization

   1014113   gpu_p13 home1_erm023_saveall_lpd023top  utr15kn  R       1:00      1 r10i4n5
   1014108   gpu_p13 home1_erm023_saveall_lpd023_09  utr15kn  R       1:30      1 r10i1n5


TART_CHKPT_FREQ=10 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --output_dir=/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest8_ermwnr_saveall_lp_0827 --command_launcher multi_gpu --datasets OfficeHome --algorithms MA --train_env 0 --n_hparams 20 --n_trials 1 --train_only_classifier 0 --holdout_fraction 0.8 --steps 5000 --save_model_every_checkpoint 1 --path_for_init /gpfsdswork/projects/rech/edr/utr15kn/dataplace/experiments/domainbed/home0ontest841_ermllr_saveall_0827/db66347321bed464bcf99408194a620a/model.pkl



home0h8_ermwnr_lp_0901.slurm
/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/homeh8_ermwnr_lp_0901

home0h8_ermwnr_0901.slurm
/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/homeh8_ermwnr_0901

home0h8_ermwnr_lp0_0901.slurm
/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/homeh8_ermwnr_lp0_0901
/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home_ermwnrbs96_lp_0901


# Folders
Random init: /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_0822

SLP123: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0_ermll_saveall_si_0822
LP123: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0_ermll_saveall_si_0822/3d4174ccb9f69a3d872124137129aa1f/model.pkl

FT123alp: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0_erm_alp_0822
FT123: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/home0_ma_saveall_lp_0824

FTPerd123:/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home_ermwnr_lp_0901

# updatable ml
Random init: /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_0822

## 0
/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home_erm0wnr_lp0_0903
from /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/home_lp0_903

## place for sbatch
/gpfswork/rech/edr/utr15kn/slurmconfig/homeperd0901/infforpacs
/gpfswork/rech/edr/utr15kn/slurmconfig/pacsspe0906

# doing
     JOBID PARTITION                           NAME     USER ST       TIME  NODES NODELIST(REASON)
   1326534   gpu_p13 pacs3_erm012wnr_ihome_lp_0915.  utr15kn PD       0:00      1 (Priority)
   1326641   gpu_p13 pacs3_erm012wnr_ihomer40_lp_09  utr15kn PD       0:00      1 (Priority)


SAVE_ONLY_FEATURES=1 MODEL_SELECTION=train WHICHMODEL=step$EPOCH INCLUDEVAL_UPTO=4 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env -1 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home_ma0123_lp_0916 --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/home_lp0123_916 20 --topk 0
# --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/specm/home_lp0123_r20_903.pkl --what feats



[0.8061855670, 0.7686139748, 0.9098083427, 0.8610792193]          21                    1.0000000000          stepbest
[0.8000000000, 0.7411225659, 0.9007891770, 0.8438576349]          21                    2.0000000000          stepbest
0.7917525773          0.7995418099          0.9255918828          0.8668197474          20

env_0_out_acc         env_1_out_acc         env_2_out_acc         env_3_out_acc         length                maxm                  robust                which
0.6989690722          0.7605956472          0.8996617813          0.8495981630          21                    3                     2.0000000000          stepbest

/gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/specm/home_erm0123_lp_0916_r40.pkl
/gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/specm/home_erm0123_lp_0916_r20.pkl
/gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/specm/home_erm0123_lp_0916_r0.pkl



LOAD_ONLY_FEATURES=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.train --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --algorithm ERM --dataset PACS --test_envs 3 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/specm/home_erm0123_lp_0916_r0.pkl --path_for_save /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/lp/pacs3_erm012wn_ihome0123r0_lp_0916.pkl --train_only_classifier 1
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --output_dir=/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/pacs3_erm012wn_ihome0123r0_lp_0916 --command_launcher multi_gpu --datasets PACS --algorithms ERM --test_envs 3 --n_hparams 20 --n_trials 1 --train_only_classifier 0 --save_model_every_checkpoint 1 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/lp/pacs3_erm012wn_ihome0123r0_lp_0916.pkl



     JOBID PARTITION                           NAME     USER ST       TIME  NODES NODELIST(REASON)
   1348141   gpu_p13 pacs3_erm012wn_ihome0123r40_lp  utr15kn  R       0:47      1 r11i2n2
   1348140   gpu_p13 pacs3_erm012wn_ihome0123r20_lp  utr15kn  R       1:07      1 r11i1n7
   1348126   gpu_p13 pacs3_erm012wn_ihome0123r0_lp_  utr15kn  R       3:39      1 r10i7n2



# from pacs

        SAVE_ONLY_FEATURES=1 MODEL_SELECTION=oracle WHICHMODEL=step$EPOCH INCLUDEVAL_UPTO=4 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r20_lp_0916 --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/lp/home0_erm123wn_ipacs0123r20_lp_0916.pkl 20 --topk 0

/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r0_lp_0916
/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r20_lp_0916




/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r40_lp_0916

## r0

0.5777548919          0.5752577320          0.7133447881          0.6323024055          0.8336148649          0.7891770011          0.8235800344          0.7703788749          82.3892893924         0.8635875392          7.8502902985          5000                  0.1862661314


Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r0_lp_0916/a2c5c880c527ecf14fb698bdfafaeeec/model_step100.pkl  with score:  0.622680412371134
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r0_lp_0916/5feb7a8830c4dc47fde650c9670dc4b5/model_step100.pkl  with score:  0.6206185567010309
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r0_lp_0916/f4b81dec88fce22108d7d9f6cc272a76/model_step100.pkl  with score:  0.6144329896907217
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r0_lp_0916/f39eb3901d347275ba5b3e3ca1f33d5a/model_step100.pkl  with score:  0.6144329896907217
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r0_lp_0916/4cb317864c5773132a6c52eb289d4a3c/model_step100.pkl  with score:  0.6082474226804123
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r0_lp_0916/04f031b99aedd570971e3d3ec0faadd8/model_step100.pkl  with score:  0.6082474226804123
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r0_lp_0916/0f6e5cac72a5565b81cd1048f5987b15/model_step100.pkl  with score:  0.6082474226804123
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r0_lp_0916/0201b7e3f603e1f92087a41592ac69cb/model_step100.pkl  with score:  0.6082474226804123
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r0_lp_0916/2e299acb2043dd288c5686c9b0c55e4b/model_step100.pkl  with score:  0.6061855670103092
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r0_lp_0916/092611e6caa483cd65daa5791ce20418/model_step100.pkl  with score:  0.6041237113402061
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r0_lp_0916/0c3d738b6ade4fcb3dedb477dccd9e84/model_step100.pkl  with score:  0.6041237113402061
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r0_lp_0916/bbd08958542690418394ddc48f49ca5d/model_step100.pkl  with score:  0.6020618556701031
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r0_lp_0916/f19375e0cf49c050d25b7132b1c9a4d1/model_step100.pkl  with score:  0.6020618556701031
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r0_lp_0916/9e19a5bae326586139415d3a771596f8/model_step100.pkl  with score:  0.6
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r0_lp_0916/7e2ffb3cbf6c316c5d5cf90cf3d12490/model_step100.pkl  with score:  0.5958762886597938
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r0_lp_0916/186bc8836a0f8d5d4a4ec48c355035f4/model_step100.pkl  with score:  0.5958762886597938
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r0_lp_0916/af98a533b35aa3d5374aa76cb7bc2a5a/model_step100.pkl  with score:  0.5876288659793815
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r0_lp_0916/34bd9b576dea7e8d403bebab87bedefb/model_step100.pkl  with score:  0.5835051546391753
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r0_lp_0916/3d4174ccb9f69a3d872124137129aa1f/model_step100.pkl  with score:  0.5711340206185567
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r0_lp_0916/3ddac555654ec118d35d45b072c168fc/model_step100.pkl  with score:  0.5690721649484536

## r20
0.6513903193          0.6556701031          0.7485681558          0.6632302405          0.8856981982          0.8466741826          0.8711990820          0.8128587830          82.3892893924         0.6680733785          7.8502902985          5000                  0.1873966360


Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r20_lp_0916/0c3d738b6ade4fcb3dedb477dccd9e84/model_best.pkl  with score:  0.6556701030927835
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r20_lp_0916/0f6e5cac72a5565b81cd1048f5987b15/model_best.pkl  with score:  0.6515463917525773
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r20_lp_0916/4cb317864c5773132a6c52eb289d4a3c/model_best.pkl  with score:  0.6474226804123712
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r20_lp_0916/5feb7a8830c4dc47fde650c9670dc4b5/model_best.pkl  with score:  0.643298969072165
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r20_lp_0916/f39eb3901d347275ba5b3e3ca1f33d5a/model_best.pkl  with score:  0.6412371134020619
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r20_lp_0916/186bc8836a0f8d5d4a4ec48c355035f4/model_best.pkl  with score:  0.6371134020618556
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r20_lp_0916/bbd08958542690418394ddc48f49ca5d/model_best.pkl  with score:  0.6350515463917525
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r20_lp_0916/04f031b99aedd570971e3d3ec0faadd8/model_best.pkl  with score:  0.6288659793814433
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r20_lp_0916/7e2ffb3cbf6c316c5d5cf90cf3d12490/model_best.pkl  with score:  0.6206185567010309
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r20_lp_0916/a2c5c880c527ecf14fb698bdfafaeeec/model_best.pkl  with score:  0.6164948453608248
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r20_lp_0916/f19375e0cf49c050d25b7132b1c9a4d1/model_best.pkl  with score:  0.6164948453608248
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r20_lp_0916/092611e6caa483cd65daa5791ce20418/model_best.pkl  with score:  0.6082474226804123
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r20_lp_0916/2e299acb2043dd288c5686c9b0c55e4b/model_best.pkl  with score:  0.6082474226804123
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r20_lp_0916/3d4174ccb9f69a3d872124137129aa1f/model_best.pkl  with score:  0.6061855670103092
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r20_lp_0916/f4b81dec88fce22108d7d9f6cc272a76/model_best.pkl  with score:  0.6041237113402061
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r20_lp_0916/af98a533b35aa3d5374aa76cb7bc2a5a/model_best.pkl  with score:  0.6
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r20_lp_0916/34bd9b576dea7e8d403bebab87bedefb/model_best.pkl  with score:  0.5979381443298969
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r20_lp_0916/0201b7e3f603e1f92087a41592ac69cb/model_best.pkl  with score:  0.5979381443298969
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r20_lp_0916/3ddac555654ec118d35d45b072c168fc/model_best.pkl  with score:  0.5835051546391753
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r20_lp_0916/9e19a5bae326586139415d3a771596f8/model_best.pkl  with score:  0.5690721649484536
## r40
0.6549948507          0.6618556701          0.7491408935          0.6632302405          0.8921734234          0.8602029312          0.8763625932          0.8243398393          82.3892893924         0.6444481599          7.8502902985          5000                  0.1874449897


Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r40_lp_0916/0f6e5cac72a5565b81cd1048f5987b15/model_best.pkl  with score:  0.6597938144329897
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r40_lp_0916/a2c5c880c527ecf14fb698bdfafaeeec/model_best.pkl  with score:  0.6556701030927835
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r40_lp_0916/5feb7a8830c4dc47fde650c9670dc4b5/model_best.pkl  with score:  0.6412371134020619
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r40_lp_0916/0c3d738b6ade4fcb3dedb477dccd9e84/model_best.pkl  with score:  0.6412371134020619
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r40_lp_0916/4cb317864c5773132a6c52eb289d4a3c/model_best.pkl  with score:  0.6391752577319587
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r40_lp_0916/f4b81dec88fce22108d7d9f6cc272a76/model_best.pkl  with score:  0.6371134020618556
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r40_lp_0916/092611e6caa483cd65daa5791ce20418/model_best.pkl  with score:  0.6309278350515464
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r40_lp_0916/f19375e0cf49c050d25b7132b1c9a4d1/model_best.pkl  with score:  0.6288659793814433
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r40_lp_0916/bbd08958542690418394ddc48f49ca5d/model_best.pkl  with score:  0.622680412371134
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r40_lp_0916/04f031b99aedd570971e3d3ec0faadd8/model_best.pkl  with score:  0.622680412371134
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r40_lp_0916/3ddac555654ec118d35d45b072c168fc/model_best.pkl  with score:  0.6185567010309279
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r40_lp_0916/f39eb3901d347275ba5b3e3ca1f33d5a/model_best.pkl  with score:  0.6144329896907217
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r40_lp_0916/2e299acb2043dd288c5686c9b0c55e4b/model_best.pkl  with score:  0.6103092783505155
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r40_lp_0916/7e2ffb3cbf6c316c5d5cf90cf3d12490/model_best.pkl  with score:  0.6082474226804123
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r40_lp_0916/af98a533b35aa3d5374aa76cb7bc2a5a/model_best.pkl  with score:  0.5979381443298969
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r40_lp_0916/34bd9b576dea7e8d403bebab87bedefb/model_best.pkl  with score:  0.5979381443298969
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r40_lp_0916/3d4174ccb9f69a3d872124137129aa1f/model_best.pkl  with score:  0.5958762886597938
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r40_lp_0916/186bc8836a0f8d5d4a4ec48c355035f4/model_best.pkl  with score:  0.5938144329896907
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r40_lp_0916/0201b7e3f603e1f92087a41592ac69cb/model_best.pkl  with score:  0.5855670103092784
Found:  /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r40_lp_0916/9e19a5bae326586139415d3a771596f8/model_best.pkl  with score:  0.5835051546391753


## imagenet


Found:  /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home/a237f897ccee3c9df18e6f887b914424/best/model_with_weights.pkl  with score:  0.6639175257731958
Found:  /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home/984b8230b5f4e350ea92eae6589202be/best/model_with_weights.pkl  with score:  0.6577319587628866
Found:  /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home/c4993823eb4730e98c4998f45dfbe5df/best/model_with_weights.pkl  with score:  0.6556701030927835
Found:  /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home/60518992a4672a2de8e9e9a7c03e5138/best/model_with_weights.pkl  with score:  0.6494845360824743
Found:  /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home/686198d1e138f25a342d5bf01436aa19/best/model_with_weights.pkl  with score:  0.6494845360824743
Found:  /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home/d40df28539ffb883795745a3f1aff7fe/best/model_with_weights.pkl  with score:  0.6494845360824743
Found:  /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home/99c90984592360b3a073ddab3f21a4dc/best/model_with_weights.pkl  with score:  0.6474226804123712
Found:  /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home/b2b931e32e0e51d3e78fa1fef54edb09/best/model_with_weights.pkl  with score:  0.643298969072165
Found:  /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home/b098d7a1520796c0c3d6d1dfb8e06478/best/model_with_weights.pkl  with score:  0.643298969072165
Found:  /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home/e08d0957d79d1afa45fd4333830740b6/best/model_with_weights.pkl  with score:  0.6371134020618556
Found:  /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home/ac8da225513c7e0b9efbf3db45ce5031/best/model_with_weights.pkl  with score:  0.6185567010309279
Found:  /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home/5fc2c3a3472f026667b02e35c43b9fbe/best/model_with_weights.pkl  with score:  0.6164948453608248
Found:  /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home/bf0bf660c2e84ea34425f0800231bfc2/best/model_with_weights.pkl  with score:  0.6144329896907217
Found:  /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home/59a0d0e21090d3cd06c9325a1e761fc3/best/model_with_weights.pkl  with score:  0.6123711340206186
Found:  /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home/b224b31bc97424b14c98c9d6d01c881a/best/model_with_weights.pkl  with score:  0.5979381443298969
Found:  /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home/883e19f430593cbeff6f073eb09ec8d5/best/model_with_weights.pkl  with score:  0.5958762886597938
Found:  /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home/9a616a675bee8aca51b5131bc4edab27/best/model_with_weights.pkl  with score:  0.5917525773195876
Found:  /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home/15b4615bcac2574163ac571885244a03/best/model_with_weights.pkl  with score:  0.5917525773195876
Found:  /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home/3f1bc74e064b8a77c610798bb2141ba8/best/model_with_weights.pkl  with score:  0.5896907216494846
Found:  /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home/78e38cf8fe4c1c176e06280eef6b99cc/best/model_with_weights.pkl  with score:  0.5793814432989691



#


enshome0_erm123wn_ipacs0123r0_lp_0916.slurm_1434327.out
enshome0_erm123wn_ipacs0123r0_lp_0916_r20.slurm_1447018.out
enshome0_erm123wn_ipacs0123r20_lp_0916.slurm_1434330.out
enshome0_erm123wn_ipacs0123r20_lp_0916_r20.slurm_1446968.out
enshome0_erm123wn_ipacs0123r40_lp_0916.slurm_1434335.out
enshome0_erm123wn_ipacs0123r40_lp_0916_r20.slurm_1446210.out
enshome0_erm123_lp_0824_r0.slurm_1445856.out
enshome0_erm123_lp_0824_r20.slurm_1447069.out


#

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --output_dir=/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123fewwn_ipacs0123r0_lp_0916 --command_launcher multi_gpu --datasets OfficeHome --algorithms ERM --test_envs 0 --n_hparams 20 --n_trials 1 --train_only_classifier 0 --save_model_every_checkpoint 1 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/lp/home0_erm123fewwn_ipacs0123r0_lp_0916.pkl  --holdout_fraction 0.8



CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --output_dir=/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_idn1erm0921r40_lp_0916 --command_launcher multi_gpu --datasets OfficeHome --algorithms ERM --test_envs 0 --n_hparams 20 --n_trials 1 --train_only_classifier 0 --save_model_every_checkpoint 1 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/lp/home0_erm123wn_idn1erm0921r40_lp_0916.pkl



env_0_in_acc          env_0_out_acc         env_1_in_acc          env_1_out_acc         env_2_in_acc          env_2_out_acc         env_3_in_acc          env_3_out_acc         length                maxm                  robust                step
0.9922680412          0.8103092784          0.9541547278          0.7938144330          0.9957746479          0.9210822999          0.9942611191          0.8645235362          20                    3                     0.0000000000          best
env_0_in_acc          env_0_out_acc         env_1_in_acc          env_1_out_acc         env_2_in_acc          env_2_out_acc         env_3_in_acc          env_3_out_acc         length                maxm                  robust                step
0.9871134021          0.8020618557          0.9383954155          0.7605956472          0.9915492958          0.8940248027          0.9870875179          0.8323765786          1                     3                     0.0000000000          best
0.9896907216          0.7546391753          0.9555873926          0.7651775487          0.9887323944          0.9064261556          0.9827833572          0.8404133180          1                     3                     0.0000000000          best


sbatch -A gtw@v100 enshome_erm0123wn_idn1erm0921r0top1_lp_0916_r0.slurm
sbatch -A gtw@v100 enshome_erm0123wn_idn1erm0921r0top1_lp_0916_r20.slurm
sbatch -A gtw@v100 enshome_erm0123wn_idn1erm0921r20top1_lp_0916_r0.slurm
sbatch -A gtw@v100 enshome_erm0123wn_idn1erm0921r20top1_lp_0916_r20.slurm

sbatch -A gtw@v100 enspacs_erm0123wn_idn1erm0921r0top1_lp_0916_r0.slurm
sbatch -A gtw@v100 enspacs_erm0123wn_idn1erm0921r0top1_lp_0916_r20.slurm
sbatch -A gtw@v100 enspacs_erm0123wn_idn1erm0921r20top1_lp_0916_r0.slurm
sbatch -A gtw@v100 enspacs_erm0123wn_idn1erm0921r20top1_lp_0916_r20.slurm


# toanalyze


/gpfsdswork/projects/rech/edr/utr15kn/slurmconfig/homeperd0901/trainfromdn/inf1/runs/enshome1_erm023_lp_r0824_r20fdn1erm0921r0.slurm_1842743.out
/gpfsdswork/projects/rech/edr/utr15kn/slurmconfig/homeperd0901/trainfromdn/inf1/runs/enshome1_erm023_lp_r0824_r20fdn1erm0921r20.slurm_1843366.out




# MODEL_SELECTION=train WHICHMODEL=stepbest INCLUDEVAL_UPTO=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0  --output_dir /data/rame/experiments/domainbed/home0_ma_lp_0824 --trial_seed 0 --data_dir /data/rame/data/domainbed --checkpoints /data/rame/data/domainbed/inits/model_home0_ermll_saveall_si_0822.pkl 0 featurizer --what addfeats --topk 1 --weight_selection train

# MODEL_SELECTION=train WHICHMODEL=stepbest CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0  --output_dir /data/rame/experiments/domainbed/home0_ma_lp_0824 --trial_seed 0 --data_dir /data/rame/data/domainbed --checkpoints /data/rame/data/domainbed/inits/model_home0_ermll_saveall_si_0822.pkl -5 featurizer --what addfeats --topk 1 --weight_selection train --hparams '{"suploss": 0, "entloss": 1., "bdiloss": 1.}'

# tta


acc_cla               df_0_0                df_0_1                df_0_2                df_1_0                df_1_1                df_1_2                df_2_0                df_2_1                df_2_2                env_0_out_acc_cla     env_1_out_acc_cla     env_2_out_acc_cla     length                robust                step                  testenv               topk
0.6390605686          0.0011117256          0.00045739525         0.00043065738         0.001068139           0.00041127444         0.00038388933         0.0010371739          0.0003933505          0.0003692967          0.6515463918          0.7709049255          0.9098083427          1                     0.000_featurizer      best                  0                     1
acc_cla               df_0_0                df_0_1                df_0_2                df_1_0                df_1_1                df_1_2                df_2_0                df_2_1                df_2_2                env_0_out_acc_cla     env_1_out_acc_cla     env_2_out_acc_cla     length                robust                step                  testenv               topk
0.6691388546          0.0017730497          0.00062730943         0.00012460387         0.0017611707          0.0006086793          9.7616714e-05         0.0017047494          0.00055893627         5.6140914e-05         0.6989690722          0.7445589920          0.8895152198          2                     1.000_featurizer      best                  0                     1
