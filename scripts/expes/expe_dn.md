

CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.train --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --algorithm ERM --dataset ${dataset} --test_envs -1 --path_for_save /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/dn_erm012345_0921.pkl --steps -1 --train_only_classifier 1

CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.train --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --algorithm ERM --dataset DomainNet --test_envs -1 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/dn_erm012345_0921.pkl --path_for_save /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/dn_erm012345_lp_0921.pkl --train_only_classifier 1

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --data_dir /gpfsscratch/rech/edr/utr15kn/data/domainbed/ --output_dir=/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/dn_erm012345_lp_0921 --command_launcher multi_gpu --datasets DomainNet --algorithms ERM --test_envs -1 --n_hparams 20 --n_trials 1 --train_only_classifier 0 --save_model_every_checkpoint 0 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/dn_erm012345_lp_0921.pkl


python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0423home --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_lp_0401_65 0 --what netm


SAVE_ONLY_FEATURES=1 MODEL_SELECTION=oracle WHICHMODEL=stepbest INCLUDEVAL_UPTO=6 KEYACC=out_Accuracies/acc_net CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --data_dir /data/rame/data/domainbed/ --dataset DomainNet --test_envs 1 --output_dir /data/rame/ermdomainnet0425 --trial_seed 0 --checkpoints /data/rame/init_diwa/inits/domainnet1_lp_0424_345 0 --path_for_init /data/rame/inits/transfer/dn1_erm02345_lp_0921.pkl --what feats



dn1_erm02345_lp_r0_0921.pkl

# Val
## DiWA


acc                   env_0_out_acc         env_1_out_acc         env_2_out_acc         env_3_out_acc         env_4_out_acc         env_5_out_acc         length                robust                which
0.2311210154          0.7850389610          0.2305978103          0.6961876427          0.6561739130          0.7909450982          0.6807956600          20                    0.0000000000          stepbest
acc                   env_0_out_acc         env_1_out_acc         env_2_out_acc         env_3_out_acc         env_4_out_acc         env_5_out_acc         length                robust                which
0.2025772696          0.7159480519          0.1998837322          0.6539126825          0.4931884058          0.7709676487          0.5977576854          21                    1.0000000000          stepbest
acc                   env_0_out_acc         env_1_out_acc         env_2_out_acc         env_3_out_acc         env_4_out_acc         env_5_out_acc         length                robust                which
0.1846914059          0.6635844156          0.1862222653          0.6226388985          0.3897391304          0.7555292145          0.5440867993          21                    2.0000000000          stepbest


## Top1
acc                   env_0_out_acc         env_1_out_acc         env_2_out_acc         env_3_out_acc         env_4_out_acc         env_5_out_acc         length                robust                which
0.2175758163          0.7571948052          0.2182928011          0.6705182315          0.6244927536          0.7666888317          0.6587341772          1                     0.0000000000          stepbest
acc                   env_0_out_acc         env_1_out_acc         env_2_out_acc         env_3_out_acc         env_4_out_acc         env_5_out_acc         length                robust                which
0.2012789458          0.7108571429          0.2000775119          0.6454023386          0.4843768116          0.7668044754          0.5958770344          2                     1.0000000000          stepbest




# Train

acc                   env_0_in_acc          env_0_out_acc         env_1_in_acc          env_1_out_acc         env_2_in_acc          env_2_out_acc         env_3_in_acc          env_3_out_acc         env_4_in_acc          env_4_out_acc         env_5_in_acc          env_5_out_acc         length                robust                step
0.2311210154          0.9020779221          0.7850389610          0.2300145349          0.2305978103          0.8053104999          0.6961876427          0.6928623188          0.6561739130          0.8319178924          0.7909450982          0.7921338156          0.6807956600          20                    0.0000000000          best

0.2175758163          0.8663636364          0.7571948052          0.2186288760          0.2182928011          0.7659574468          0.6705182315          0.6587681159          0.6244927536          0.8040909255          0.7666888317          0.7520795660          0.6587341772          1                     0.0000000000          best

0.2012789458          0.7733766234          0.7108571429          0.2046996124          0.2000775119          0.6987545407          0.6454023386          0.4982608696          0.4843768116          0.7922012215          0.7668044754          0.6518987342          0.5958770344          2                     1.0000000000          best



# lp dn

data_dir=/private/home/alexandrerame/dataplace/data/domainbed
dataset=DomainNet

test_envs="0 1 2 3 4 5"
train_env=

srun /private/home/alexandrerame/.conda/envs/pytorch/bin/python3 /private/home/alexandrerame/domainbedv2/domainbed/scripts/train.py --data_dir ${data_dir} --algorithm ERM --dataset ${dataset} --test_envs ${test_envs} --path_for_init ${data_dir}/inits/dn/dn_rand_0926.pkl --path_for_save ${data_dir}/inits/dn/dn_lp${train_env}_0926.pkl --train_only_classifier 1

# sweep dn

data_dir=/private/home/alexandrerame/dataplace/data/domainbed
expe_dir=/private/home/alexandrerame/dataplace/experiments/domainbed
dataset=DomainNet

test_envs="0 1 2 3 4 5"
train_env=

srun /private/home/alexandrerame/.conda/envs/pytorch/bin/python3 /private/home/alexandrerame/domainbedv2/domainbed/scripts/sweep.py launch --n_hparams 20 --n_trials 1 --save_model_every_checkpoint 0 --data_dir ${data_dir} --command_launcher multi_gpu --datasets ${dataset} --algorithms ERM --path_for_init ${data_dir}/inits/dn/dn_lp${train_env}_0926.pkl --output_dir ${expe_dir}/dn/dn_erm_lp${train_env}_0926 --test_envs ${test_envs}

# diwa for dn

export INCLUDETSV_UPTO=6
export MODEL_SELECTION=train
export SAVE_ONLY_FEATURES=1
export WHICHMODEL=stepbest

data_dir=/private/home/alexandrerame/dataplace/data/domainbed
expe_dir=/private/home/alexandrerame/dataplace/experiments/domainbed
dataset=DomainNet

test_envs="0 1 2 3 4 5"
train_env=
robust_ft=0
topk=0

srun /private/home/alexandrerame/.conda/envs/pytorch/bin/python3 /private/home/alexandrerame/domainbedv2/domainbed/scripts/diwa.py --trial_seed 0 --data_dir ${data_dir} --dataset ${dataset} --output_dir ${expe_dir}/dn/dn_erm_lp${train_env}_0926 --checkpoints ${data_dir}/inits/dn/dn_lp${train_env}_0926.pkl ${robust_ft} --path_for_init ${data_dir}/inits/dn/transfer/dn_erm_lp${train_env}_r${robust_ft}_t${topk}_0926.pkl --what feats --test_env -1 --topk ${topk}

# lp home from dn
export LOAD_ONLY_FEATURES=1

data_dir=/private/home/alexandrerame/dataplace/data/domainbed
expe_dir=/private/home/alexandrerame/dataplace/experiments/domainbed
dataset=OfficeHome

train_env_transfer=0
robust_ft_transfer=0
topk_transfer=0
test_envs=

srun /private/home/alexandrerame/.conda/envs/pytorch/bin/python3 /private/home/alexandrerame/domainbedv2/domainbed/scripts/train.py --data_dir ${data_dir} --algorithm ERM --dataset ${dataset} --test_envs ${test_envs} --path_for_init ${data_dir}/inits/dn/transfer/dn_erm_lp${train_env_transfer}_r${robust_ft_transfer}_t${topk_transfer}_0926.pkl --path_for_save ${data_dir}/inits/home/fromdn/home${test_envs}_lp_idn${train_env_transfer}r${robust_ft_transfer}t${topk_transfer}_0926.pkl --train_only_classifier 1

# sweep home

export LOAD_ONLY_FEATURES=1

data_dir=/private/home/alexandrerame/dataplace/data/domainbed
expe_dir=/private/home/alexandrerame/dataplace/experiments/domainbed
dataset=OfficeHome

train_env_transfer=0
robust_ft_transfer=0
topk_transfer=0
test_envs=

srun /private/home/alexandrerame/.conda/envs/pytorch/bin/python3 /private/home/alexandrerame/domainbedv2/domainbed/scripts/sweep.py launch --n_hparams 20 --n_trials 1 --save_model_every_checkpoint 1 --data_dir ${data_dir} --command_launcher multi_gpu --datasets ${dataset} --algorithms ERM --path_for_init ${data_dir}/inits/home/fromdn/home${test_envs}_lp_idn${train_env_transfer}r${robust_ft_transfer}t${topk_transfer}_0926.pkl --output_dir ${expe_dir}/home/home_erm_lp${train_env}_idn${train_env_transfer}r${robust_ft_transfer}t${topk_transfer}_0926 --test_envs ${test_envs}

# diwa for home

export INCLUDETSV_UPTO=4
export MODEL_SELECTION=train
export WHICHMODEL=stepbest

data_dir=/private/home/alexandrerame/dataplace/data/domainbed
expe_dir=/private/home/alexandrerame/dataplace/experiments/domainbed
dataset=DomainNet

train_env_transfer=0
robust_ft_transfer=0
topk_transfer=0
test_envs=
robust_ft=0
topk=0

srun /private/home/alexandrerame/.conda/envs/pytorch/bin/python3 /private/home/alexandrerame/domainbedv2/domainbed/scripts/diwa.py --trial_seed 0 --data_dir ${data_dir} --dataset ${dataset} --output_dir ${expe_dir}/home/home_erm_lp${train_env}_idn${train_env_transfer}r${robust_ft_transfer}t${topk_transfer}_0926 --checkpoints ${data_dir}/inits/home/fromdn/home${test_envs}_lp_idn${train_env_transfer}r${robust_ft_transfer}t${topk_transfer}_0926.pkl ${robust_ft} --test_env ${test_envs} --topk ${topk}
