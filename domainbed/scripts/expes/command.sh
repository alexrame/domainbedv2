ssh utr15kn@jean-zay.idris.fr
sed -i -- 's/ModelRecycling/ModelRatatouille/g' *.slurm


grep "&&" | jq '(.acc|tostring) +" " + (.acc_conf|tostring) +" "+ .topk + " " + (.length|tostring) + " " + (.dirs|tostring)'



grep "&&" | cut -f2- -d" "| cut -f1 -d")"
cat infhomecorrupt01_0120.slurm_2223741.out | grep "&&" | jq '(.length|tostring) + " " + (.env_0_in_acc|tostring) + " " + (.env_1_in_acc|tostring) + " " + (.env_2_in_acc|tostring) + " " + (.env_2_in_acc_ens|tostring) + " " +  (.dirs|tostring)'

grep "&&" | jq '(.acc|tostring) +" "+ (.acc_conf|tostring) +" "+ .topk + " " + (.length|tostring)'

grep "&&" | jq '(.acc|tostring) +" " + (.acc_conf|tostring) +" "+ .topk + " " + (.length|tostring) + " " + (.dirs|tostring)'

grep printres | tail -n 10 | jq '(.env0_out_acc|tostring) + " " + (.step|tostring)'

sbatch -A gtw@v100


/private/home/alexandrerame/.conda/envs/pytorch/bin/python3 /private/home/alexandrerame/domainbedv2/domainbed/scripts/train.py --data_dir ${data_dir} --dataset ${dataset} --algorithm ERM --path_for_init ${data_dir}/inits/home/home0_lp_0926.pkl --output_dir ${expe_dir}/homecorrupt/train_homecorrupt1_erm2_lp_0120 --test_envs 0 1 --hparams '{"corrupt_prop": 0.1, "resnet_dropout": 0.0, "data_augmentation": 0}' --steps 20000 --save_model_every_checkpoint 1000


for FILE in enspacs1_erm023wn_idn1erm0921r0_lp_0916_r0.slurm_1801193.out enspacs1_erm023wn_idn1erm0921r20_lp_0916_r0.slurm_1801255.out enspacs1_erm023wn_idn1erm0921r40_lp_0916_r0.slurm_1801272.out enspacs1_erm023wn_idn1erm0921r0_lp_0916_r20.slurm_1801043.out enspacs1_erm023wn_idn1erm0921r20_lp_0916_r20.slurm_1801275.out enspacs1_erm023wn_idn1erm0921r40_lp_0916_r20.slurm_1801326.out
do
        echo $FILE
        echo $FILE >> pacs1dn.py
        cat $FILE | grep printres >> pacs1dn.py
done


# list top hparams
KEYSORT=train TESTOUT=1 python3 -m domainbed.scripts.list_top_hparams --input_dir /private/home/alexandrerame/dataplace/experiments/domainbed/home/home0_erm0p8_ermftop0_lpdn1_1026/ --dataset OfficeHome --test_env 1 --algorithm ERM

# dev gpus
srun --gpus-per-node=1 --partition=learnlab --time=10:00:00 --cpus-per-task 10 --constraint=volta32gb --pty /bin/bash -l

# full pipeline from DomainNet to OfficeHome

## lp dn

data_dir=/private/home/alexandrerame/dataplace/data/domainbed
dataset=DomainNet

test_envs="0 1 2 3 4 5"

train_env=

srun /private/home/alexandrerame/.conda/envs/pytorch/bin/python3 /private/home/alexandrerame/domainbedv2/domainbed/scripts/train.py --data_dir ${data_dir} --algorithm ERM --dataset ${dataset} --test_envs ${test_envs} --path_for_init ${data_dir}/inits/dn/dn_rand_0926.pkl --path_for_save ${data_dir}/inits/dn/dn_lp${train_env}_0926.pkl --train_only_classifier 1

## sweep dn

data_dir=/private/home/alexandrerame/dataplace/data/domainbed
expe_dir=/private/home/alexandrerame/dataplace/experiments/domainbed
dataset=DomainNet

test_envs="0 1 2 3 4 5"
train_env=

srun /private/home/alexandrerame/.conda/envs/pytorch/bin/python3 /private/home/alexandrerame/domainbedv2/domainbed/scripts/sweep.py launch --n_hparams 20 --save_model_every_checkpoint 0 --data_dir ${data_dir} --datasets ${dataset} --algorithms ERM --path_for_init ${data_dir}/inits/dn/dn_lp${train_env}_0926.pkl --output_dir ${expe_dir}/dn/dn_erm${train_env}_lp${train_env}_0926 --test_envs ${test_envs}

## diwa for dn

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

srun /private/home/alexandrerame/.conda/envs/pytorch/bin/python3 /private/home/alexandrerame/domainbedv2/domainbed/scripts/diwa.py --trial_seed 0 --data_dir ${data_dir} --dataset ${dataset} --output_dir ${expe_dir}/dn/dn_erm${train_env}_lp${train_env}_0926 --checkpoints ${data_dir}/inits/dn/dn_lp${train_env}_0926.pkl ${robust_ft} --path_for_init ${data_dir}/inits/dn/transfer/dn_erm${train_env}_lp${train_env}_r${robust_ft}_t${topk}_0926.pkl --what feats --test_env -1 --topk ${topk}

## lp home from dn
export LOAD_ONLY_FEATURES=1

data_dir=/private/home/alexandrerame/dataplace/data/domainbed
expe_dir=/private/home/alexandrerame/dataplace/experiments/domainbed
dataset=OfficeHome

train_env_transfer=0
robust_ft_transfer=0
topk_transfer=0
test_env="todo"

srun /private/home/alexandrerame/.conda/envs/pytorch/bin/python3 /private/home/alexandrerame/domainbedv2/domainbed/scripts/train.py --data_dir ${data_dir} --algorithm ERM --dataset ${dataset} --test_envs ${test_env} --path_for_init ${data_dir}/inits/dn/transfer/dn_erm${train_env_transfer}_lp${train_env_transfer}_r${robust_ft_transfer}_t${topk_transfer}_0926.pkl --path_for_save ${data_dir}/inits/home/fromdn/home${test_env}_lp_idn${train_env_transfer}r${robust_ft_transfer}t${topk_transfer}_0926.pkl --train_only_classifier 1

## sweep home

export LOAD_ONLY_FEATURES=1

data_dir=/private/home/alexandrerame/dataplace/data/domainbed
expe_dir=/private/home/alexandrerame/dataplace/experiments/domainbed
dataset=OfficeHome

train_env_transfer=0
robust_ft_transfer=0
topk_transfer=0
test_env="todo"

srun /private/home/alexandrerame/.conda/envs/pytorch/bin/python3 /private/home/alexandrerame/domainbedv2/domainbed/scripts/sweep.py launch --n_hparams 20 --save_model_every_checkpoint 1 --data_dir ${data_dir} --datasets ${dataset} --algorithms ERM --path_for_init ${data_dir}/inits/home/fromdn/home${test_env}_lp_idn${train_env_transfer}r${robust_ft_transfer}t${topk_transfer}_0926.pkl --output_dir ${expe_dir}/home/home_erm_lp${train_env}_idn${train_env_transfer}r${robust_ft_transfer}t${topk_transfer}_0926 --test_envs ${test_env}

## diwa for home

export INCLUDETSV_UPTO=4
export MODEL_SELECTION=train
export WHICHMODEL=stepbest

data_dir=/private/home/alexandrerame/dataplace/data/domainbed
expe_dir=/private/home/alexandrerame/dataplace/experiments/domainbed
dataset=DomainNet

train_env_transfer=0
robust_ft_transfer=0
topk_transfer=0
test_env="todo"
robust_ft=0
topk=0

srun /private/home/alexandrerame/.conda/envs/pytorch/bin/python3 /private/home/alexandrerame/domainbedv2/domainbed/scripts/diwa.py --trial_seed 0 --data_dir ${data_dir} --dataset ${dataset} --output_dir ${expe_dir}/home/home_erm_lp${train_env}_idn${train_env_transfer}r${robust_ft_transfer}t${topk_transfer}_0926 --checkpoints ${data_dir}/inits/home/fromdn/home${test_env}_lp_idn${train_env_transfer}r${robust_ft_transfer}t${topk_transfer}_0926.pkl ${robust_ft} --test_env ${test_env} --topk ${topk}



## rand officehome

data_dir=/private/home/alexandrerame/dataplace/data/domainbed
dataset=OfficeHome

srun /private/home/alexandrerame/.conda/envs/pytorch/bin/python3 /private/home/alexandrerame/domainbedv2/domainbed/scripts/train.py --data_dir ${data_dir} --algorithm ERM --dataset ${dataset} --test_envs -1 --path_for_init ${data_dir}/inits/home/home_rand_0926.pkl --path_for_save ${data_dir}/inits/home/home_rand_0926.pkl --train_only_classifier 1 --steps -1


## lp officehome

data_dir=/private/home/alexandrerame/dataplace/data/domainbed
dataset=OfficeHome

test_env="todo"

srun /private/home/alexandrerame/.conda/envs/pytorch/bin/python3 /private/home/alexandrerame/domainbedv2/domainbed/scripts/train.py --data_dir ${data_dir} --algorithm ERM --dataset ${dataset} --test_envs ${test_env} --path_for_save ${data_dir}/inits/home/home${test_env}_lp_0926.pkl --train_only_classifier 1

## sweep dn

data_dir=/private/home/alexandrerame/dataplace/data/domainbed
expe_dir=/private/home/alexandrerame/dataplace/experiments/domainbed
dataset=OfficeHome

test_env="todo"

srun /private/home/alexandrerame/.conda/envs/pytorch/bin/python3 /private/home/alexandrerame/domainbedv2/domainbed/scripts/sweep.py launch --n_hparams 20 --save_model_every_checkpoint 1 --data_dir ${data_dir} --datasets ${dataset} --algorithms ERM --path_for_init ${data_dir}/inits/home/home${test_env}_lp_0926.pkl --output_dir ${expe_dir}/home/home${test_env}_erm_lp_0926 --test_envs ${test_env}



## diwa for home transfer

export INCLUDETSV_UPTO=4
export MODEL_SELECTION=train
export WHICHMODEL=stepbest
export NORMALIZE=1

data_dir=/private/home/alexandrerame/dataplace/data/domainbed
expe_dir=/private/home/alexandrerame/dataplace/experiments/domainbed
dataset=DomainNet

# robust_ft_transfer=0
# topk_transfer=0

train_env_transfer=0
test_env="todo"
robusttransfer=0
topk=0

srun /private/home/alexandrerame/.conda/envs/pytorch/bin/python3 /private/home/alexandrerame/domainbedv2/domainbed/scripts/diwa.py --trial_seed 0 --dataset ${dataset} --test_env ${test_env} --topk ${topk} --data_dir ${data_dir} --output_dir ${expe_dir}/home/home${test_env}_erm_lp_0926 --checkpoints ${data_dir}/inits/home/home${test_env}_lp_0926.pkl 0 network ${data_dir}/inits/dn/transfer/dn_erm${train_env_transfer}_lp${train_env_transfer}_r0_t0_0926.pkl ${robusttransfer} featurizeronly --weighting 1/20 --what feats cla

