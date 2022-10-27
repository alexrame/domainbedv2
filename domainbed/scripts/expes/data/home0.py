# twa lambdas

h = [0.05, 0.1]
env0_in_acc = [0.72, 0.7212814645]
env0_out_acc = [0.71, 0.7272727273]

# llretrain from lambda = 1
h = [0.05, 0.1]
env0_in_acc = [0.99, 0.98]
env0_out_acc = [0.69, 0.72]


UDA_VISIBLE_DEVICES=0 /private/home/alexandrerame/.conda/envs/pytorch/bin/python3 /private/home/alexandrerame/domainbedv2/domainbed/scripts/train.py --data_dir ${data_dir} --dataset ${dataset} --algorithm TWA --path_for_init ${data_dir}/inits/home/transfer/home${test_env}_ermftop${topk}_lpdn${robusttransfer}_0926.pkl --test_envs 1 2 3 --train_only_classifier lambdas --holdout_fraction 0.5 --skip_model_save --hparams '{"lossce": 0.0, "lossent": 1.0, "lossbdi": 1.0, "batch_size": 128, "featurizers_lambdas": "2 0 0 0"}' --checkpoint_freq 1
