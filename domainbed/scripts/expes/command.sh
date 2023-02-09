


ssh utr15kn@jean-zay.idris.fr
sed -i -- 's/topk \-1/topk \-2/g' inf2homecorrupt*.slurm
sed -i -- 's/1-_-4/1-_-2/g' *.slurm


grep "&&" | jq '(.step|tostring) + " " + (.acc|tostring) +" " + (.acc_conf|tostring) +" "+ .topk + " " + (.length|tostring) + " " + (.dirs|tostring)'

grep "&&" | jq '(.step|tostring) + " " +  (.length|tostring) + " " + (.env_0_in_acc|tostring) + " " + (.env_1_in_acc|tostring) + " " + (.env_2_in_acc|tostring)  + " " + (.topk|tostring)'

grep "&&" | jq '(.step|tostring) + " " +  (.length|tostring) + " " + (.acc|tostring) + " " + (.acc_ens|tostring) + " " + (.env_1_in_acc|tostring) + " " + (.env_1_in_acc_ens|tostring) + " " + (.env_2_in_acc|tostring) + " " + (.env_2_in_acc_ens|tostring) + " " + (.env_1_out_acc|tostring) + " " + (.env_1_out_acc_ens|tostring) + " " + (.env_2_out_acc|tostring) + " " + (.env_2_out_acc_ens|tostring) + " " + (.topk|tostring)'



{"acc": 0.661722290894108, "acc_ens": 0.6501854140914709, "dirs": "homecorrupt1_erm2_lp_0120,homecorrupt0_erm2_lp_0120", "dirslen": 2, "env_1_in_acc": 0.9742038216560509, "env_1_in_acc_ens": 0.9729299363057324, "env_1_out_acc": 0.9782608695652174, "env_1_out_acc_ens": 0.9782608695652174, "env_2_in_acc": 0.2689075630252101, "env_2_in_acc_ens": 0.05042016806722689, "env_2_out_acc": 0.0, "env_2_out_acc_ens": 0.0, "length": 2, "lengthf": 2, "maxm": 0, "step": 10000, "testenv": 0, "topk": "1-_-1", "trialseed": 0, "printres": "&&"}


{"acc": 0.6683147919241862, "acc_cla0": 0.6736711990111248, "acc_cla1": 0.6641944787803873, "acc_ens": 0.6621343222084879, "acc_enscla": 0.6683147919241862, "acc_netm": 0.6312319736299958, "acc_netmax": 0.6468891635764318, "dirs": "homecorrupt1_erm2_lp_0120,homecorrupt0_erm2_lp_0120", "dirslen": 2, "divd_netm": 0.2913061392665843, "divq_netm": 0.9311051226084943, "divr_netm": 0.5318246110325319, "env_0_in_acc": 0.6652935118434603, "env_0_in_acc_cla0": 0.668898043254377, "env_0_in_acc_cla1": 0.6616889804325438, "env_0_in_acc_ens": 0.6580844490216272, "env_0_in_acc_enscla": 0.6652935118434603, "env_0_in_acc_netm": 0.6302780638516993, "env_0_in_acc_netmax": 0.6483007209062822, "env_0_in_divd_netm": 0.2909371781668383, "env_0_in_divq_netm": 0.9292713681992159, "env_0_in_divr_netm": 0.5415929203539823, "env_0_out_acc": 0.6804123711340206, "env_0_out_acc_cla0": 0.6927835051546392, "env_0_out_acc_cla1": 0.6742268041237114, "env_0_out_acc_ens": 0.6783505154639176, "env_0_out_acc_enscla": 0.6804123711340206, "env_0_out_acc_netm": 0.6350515463917525, "env_0_out_acc_netmax": 0.6412371134020619, "env_0_out_divd_netm": 0.2927835051546392, "env_0_out_divq_netm": 0.939172627682457, "env_0_out_divr_netm": 0.4929577464788732, "env_1_in_acc": 0.9745060548119822, "env_1_in_acc_cla0": 0.9741873804971319, "env_1_in_acc_cla1": 0.9745060548119822, "env_1_in_acc_ens": 0.9741873804971319, "env_1_in_acc_enscla": 0.9745060548119822, "env_1_in_acc_netm": 0.9681325685149778, "env_1_in_acc_netmax": 0.9732313575525813, "env_1_in_divd_netm": 0.020395156150414276, "env_1_in_divq_netm": 0.9892321709599933, "env_1_in_divr_netm": 1.125, "env_1_out_acc": 0.9668367346938775, "env_1_out_acc_cla0": 0.9655612244897959, "env_1_out_acc_cla1": 0.9642857142857143, "env_1_out_acc_ens": 0.9668367346938775, "env_1_out_acc_enscla": 0.9668367346938775, "env_1_out_acc_netm": 0.9642857142857142, "env_1_out_acc_netmax": 0.9693877551020408, "env_1_out_divd_netm": 0.025510204081632654, "env_1_out_divq_netm": 0.9936034115138593, "env_1_out_divr_netm": 0.7999999999999999, "env_2_in_acc": 0.028735632183908046, "env_2_in_acc_cla0": 0.02586206896551724, "env_2_in_acc_cla1": 0.02586206896551724, "env_2_in_acc_ens": 0.05172413793103448, "env_2_in_acc_enscla": 0.028735632183908046, "env_2_in_acc_netm": 0.41235632183908044, "env_2_in_acc_netmax": 0.8247126436781609, "env_2_in_divd_netm": 0.1752873563218391, "env_2_in_divq_netm": nan, "env_2_in_divr_netm": 4.7049180327868845, "env_2_out_acc": 0.011494252873563218, "env_2_out_acc_cla0": 0.0, "env_2_out_acc_cla1": 0.011494252873563218, "env_2_out_acc_ens": 0.011494252873563218, "env_2_out_acc_enscla": 0.011494252873563218, "env_2_out_acc_netm": 0.39655172413793105, "env_2_out_acc_netmax": 0.7931034482758621, "env_2_out_divd_netm": 0.20689655172413793, "env_2_out_divq_netm": nan, "env_2_out_divr_netm": 3.8333333333333335, "length": 2, "lengthf": 2, "step": 5000, "testenv": 0, "topk": "1-_-1", "trialseed": 0, "printres": "&&"}

grep "&&" | cut -f2- -d" "| cut -f1 -d")"
grep "&&" | jq '(.acc|tostring) +" "+ (.length|tostring)'
grep "&&" | jq '(.acc|tostring) +" "+ .topk + " " + (.length|tostring) + " " + (.dirs|tostring)'

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

