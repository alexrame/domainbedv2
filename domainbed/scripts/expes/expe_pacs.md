CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.train --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --algorithm ERM --dataset OfficeHome --test_envs -1 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_0822 --path_for_save /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/home_lp0123_916 --seed 916 --train_only_classifier 1
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --output_dir=/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home_ma0123_lp_0916 --command_launcher multi_gpu --datasets OfficeHome --algorithms MA --test_envs -1 --n_hparams 20 --n_trials 1 --train_only_classifier 0 --steps 5000 --save_model_every_checkpoint 0 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/home_lp0123_916


/gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/specm/pacs_erm0123_lp_0916_r0.pkl
0.9779951100          0.9786324786          0.9880239521          0.9681528662          20                    0.0000000000          stepbest
env_0_out_acc         env_1_out_acc         env_2_out_acc         env_3_out_acc         length                robust                which
0.9559902200          0.9807692308          0.9910179641          0.9566878981          1                     0.0000000000          stepbest

/gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/specm/pacs_erm0123_lp_0916_r20.pkl
0.9682151589          0.9786324786          0.9970059880          0.9414012739          21                    1.0000000000          stepbest
env_0_out_acc         env_1_out_acc         env_2_out_acc         env_3_out_acc         length                robust                which
0.9535452323          0.9636752137          0.9970059880          0.9006369427          2                     1.0000000000          stepbest


/gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/specm/pacs_erm0123_lp_0916_r40.pkl
0.9535452323          0.9529914530          0.9970059880          0.8917197452          21                    2.0000000000          stepbest
env_0_out_acc         env_1_out_acc         env_2_out_acc         env_3_out_acc         length                robust                which
0.9290953545          0.9380341880          0.9970059880          0.8343949045          2                     2.0000000000          stepbest

env_0_out_acc         env_0_out_acc_enscla  env_1_out_acc         env_1_out_acc_enscla  env_2_out_acc         env_2_out_acc_enscla  env_3_out_acc         env_3_out_acc_enscla  length                robust                which
0.9779951100          0.9779951100          0.9786324786          0.9786324786          0.9880239521          0.9880239521          0.9681528662          0.9681528662          20                    0.0000000000          stepbest



LOAD_ONLY_FEATURES=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.train --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --algorithm ERM --dataset OfficeHome --test_envs 0 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/specm/pacs_erm0123_lp_0916_r0.pkl --path_for_save /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/lp/home0_erm123wn_ipacs0123r0_lp_0916.pkl --train_only_classifier 1
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --output_dir=/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_ipacs0123r0_lp_0916 --command_launcher multi_gpu --datasets OfficeHome --algorithms ERM --test_envs 0 --n_hparams 20 --n_trials 1 --train_only_classifier 0 --save_model_every_checkpoint 1 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/lp/home0_erm123wn_ipacs0123r0_lp_0916.pkl


CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.train --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --algorithm ERM --dataset OfficeHome --test_envs -1 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home0_0822 --path_for_save /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/home_lp0123_916 --seed 916 --train_only_classifier 1
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --output_dir=/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home_ma0123_lp_0916 --command_launcher multi_gpu --datasets OfficeHome --algorithms MA --test_envs -1 --n_hparams 20 --n_trials 1 --train_only_classifier 0 --steps 5000 --save_model_every_checkpoint 0 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/home_lp0123_916



CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --output_dir=/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/ --command_launcher multi_gpu --datasets PACS --algorithms ERM --test_envs 3 --n_hparams 20 --n_trials 1 --train_only_classifier 0 --save_model_every_checkpoint 1 --path_for_init

        MAXM=3 MODEL_SELECTION=oracle WHICHMODEL=step$EPOCH INCLUDEVAL_UPTO=4 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset PACS --test_env 3 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/pacs3_erm012wn_ihome0123r0_lp_0916 --trial_seed 0 --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --checkpoints /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/lp/pacs3_erm012wn_ihome0123r0_lp_0916.pkl 0 --topk 0 --what netm


# r0
0.9280048810          0.9070904645          0.8923240938          0.8760683761          0.9917664671          0.9790419162          0.6469465649          0.6547770701          119.7604790419        0.2030716695          7.8485202789          5000                  0.1187917757
# r20
0.9188529591          0.8948655257          0.8752665245          0.8675213675          0.9940119760          0.9760479042          0.6024173028          0.6114649682          119.7604790419        0.2206916039          7.8485202789          5000                  0.1172592163
# r40
Saving new best score at step: 4900 at path: model_best.pkl
0.9151921904          0.8875305623          0.8640724947          0.8589743590          0.9917664671          0.9730538922          0.5648854962          0.5656050955          119.7604790419        0.2396117578          7.8485202789          5000                  0.1147945070
# rinf
0.8816351434          0.8630806846          0.8358208955          0.8376068376          0.9872754491          0.9850299401          0.4624681934          0.4726114650          119.7604790419        0.3207884185          7.8485202789          5000                  0.1144417500



# pacs0

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --output_dir=/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/pacs0_erm012wn_ihome0123r0_lp_0916 --command_launcher multi_gpu --datasets PACS --algorithms ERM --test_envs 0 --n_hparams 20 --n_trials 1 --train_only_classifier 0 --save_model_every_checkpoint 1 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/lp/pacs0_erm012wn_ihome0123r0_lp_0916.pkl

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --output_dir=/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/pacs_ermwnr_lp_0906 --command_launcher multi_gpu --datasets PACS --algorithms ERM --test_envs 0 --n_hparams 20 --n_trials 1 --train_only_classifier 0 --save_model_every_checkpoint 1 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/pacs0_lp_906

# officehome8

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --output_dir=/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/pacs_ermfew_lp_0906 --command_launcher multi_gpu --datasets PACS --algorithms ERM --test_env 3 --n_hparams 20 --n_trials 1 --train_only_classifier 0 --steps 5000 --save_model_every_checkpoint 1 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/pacs3_ermfew_lp_906 --holdout_fraction 0.8



OAD_ONLY_FEATURES=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.train --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --algorithm ERM --dataset PACS --test_envs 3 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/specm/home_erm0123_lp_0916_r0.pkl --path_for_save /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/lp/pacs3_erm012fewwn_ihome0123r0_lp_0916.pkl --train_only_classifier 1 --holdout_fraction 0.8
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --output_dir=/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/pacs3_erm012fewwn_ihome0123r0_lp_0916 --command_launcher multi_gpu --datasets PACS --algorithms ERM --test_envs 3 --n_hparams 20 --n_trials 1 --train_only_classifier 0 --save_model_every_checkpoint 1 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/lp/pacs3_erm012fewwn_ihome0123r0_lp_0916.pkl --holdout_fraction 0.8


/gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/lp/pacs0_erm123wn_idn1erm0921r0_lp_0916.pkl


LOAD_ONLY_FEATURES=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.train --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --algorithm ERM --dataset PACS --test_envs 0 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/specm/dn1_erm02345_lp_r0_0921.pkl --path_for_save /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/lp/pacs0_erm123wn_idn1erm0921r0_lp_0916.pkl --train_only_classifier 1
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --data_dir /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed --output_dir=/gpfsscratch/rech/edr/utr15kn/experiments/domainbed/pacs0_erm123wn_idn1erm0921r0_lp_0916 --command_launcher multi_gpu --datasets PACS --algorithms ERM --test_envs 0 --n_hparams 20 --n_trials 1 --train_only_classifier 0 --save_model_every_checkpoint 1 --path_for_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/spec/lp/pacs0_erm123wn_idn1erm0921r0_lp_0916.pkl


for FILE in enshome0_erm123wn_ipacs0123r0top1_lp_0916_r0.slurm_1587327.out enshome0_erm123wn_ipacs0123r20top1_lp_0916_r0.slurm_1587328.out enshome0_erm123wn_ipacs0123r40top1_lp_0916_r0.slurm_1587329.out
do
echo $FILE >> home0top1.py
cat $FILE | grep printres >> home0top1.py
done


enshome0_erm123fewwn_ipacs0123r40_lp_0916_r0.slurm_1587270.out enshome0_erm123fewwn_ipacs0123r40_lp_0916_r20.slurm_1587271.out enshome0_erm123fewwn_ipacs0123r20_lp_0916_r0.slurm_1587258.out enshome0_erm123fewwn_ipacs0123r20_lp_0916_r20.slurm_1587265.out enshome0_erm123fewwn_ipacs0123r0_lp_0916_r0.slurm_1587212.out enshome0_erm123fewwn_ipacs0123r0_lp_0916_r20.slurm_1587223.out

clean_labels = [
"100    0       0",
"66     33      0",
"50     50      0",
"0      100     0",
"66     0       33",
"50     0       50",
"0      0       100"
]


