

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from codeplot import plot

def plot_slopes_c():
    fig = plt.figure()
    iids = [ll["val_acc"] for ll in l]
    oods = [ll["test_acc"] for ll in l]
    plt.scatter(iids, oods)

    plt.axhline(y=0)
    plt.xlabel("IID", fontsize="x-large")
    plt.ylabel("OOD", fontsize="x-large")
    plt.legend(fontsize="x-large")
    return fig


plot.plt.rcParams["figure.figsize"] = (6, 4)
fig_testood = plot_slopes_c()


python3 -m domainbed.scripts.list_top_hparams --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra0_erm_lp_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra1_erm_lp_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra2_erm_lp_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra3_erm_lp_0926/ --dataset TerraIncognita --test_env ${test_env} --algorithm ERM >> ../slurmconfig/terra0926/results/lthp_terra${test_env}.md

python3 -m domainbed.scripts.list_top_hparams --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra0_erm_lplw_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra1_erm_lplw_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra2_erm_lplw_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra3_erm_lplw_0926/ --dataset TerraIncognita --test_env ${test_env} --algorithm MA >> ../slurmconfig/terra0926/results/lthp_terra${test_env}.md

python3 -m domainbed.scripts.list_top_hparams --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra0_twama_lpl4_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra1_twama_lpl4_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra2_twama_lpl4_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra3_twama_lpl4_0926/ --dataset TerraIncognita --test_env ${test_env} --algorithm TWAMA >> ../slurmconfig/terra0926/results/lthp_terra${test_env}.md

python3 -m domainbed.scripts.list_top_hparams --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra0_twama_lpl4w0_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra1_twama_lpl4w0_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra2_twama_lpl4w0_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra3_twama_lpl4w0_0926/ --dataset TerraIncognita --test_env ${test_env} --algorithm TWAMA >> ../slurmconfig/terra0926/results/lthp_terra${test_env}.md

# rm ../slurmconfig/terra0926/results/lthp_terra${test_env}.md

rm ../slurmconfig/terra0926/results/lthp_terra.md

for test_env in 2 3
do
python3 -m domainbed.scripts.list_top_hparams --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra0_erm_lp_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra1_erm_lp_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra2_erm_lp_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra3_erm_lp_0926/ --dataset TerraIncognita --test_env ${test_env} --key_uniq args.hpstep --algorithm ERM >> ../slurmconfig/terra0926/results/lthp_terra.md
python3 -m domainbed.scripts.list_top_hparams --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra0_erm_lplw_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra1_erm_lplw_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra2_erm_lplw_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra3_erm_lplw_0926/ --dataset TerraIncognita --test_env ${test_env} --key_uniq args.hpstep --algorithm MA >> ../slurmconfig/terra0926/results/lthp_terra.md
python3 -m domainbed.scripts.list_top_hparams --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra0_twama_lpl4w0a_i_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra1_twama_lpl4w0a_i_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra2_twama_lpl4w0a_i_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra3_twama_lpl4w0a_i_0926  --dataset TerraIncognita --test_env ${test_env} --key_uniq args.hpstep --algorithm TWAMA >> ../slurmconfig/terra0926/results/lthp_terra.md
python3 -m domainbed.scripts.list_top_hparams --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra0_twama_lpl4w0_i_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra1_twama_lpl4w0_i_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra2_twama_lpl4w0_i_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra3_twama_lpl4w0_i_0926  --dataset TerraIncognita --test_env ${test_env} --key_uniq args.hpstep --algorithm TWAMA >> ../slurmconfig/terra0926/results/lthp_terra.md
done

rm ../slurmconfig/home0926/results/lthp_home.md

for test_env in 0 1
do
python3 -m domainbed.scripts.list_top_hparams --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/home/home0_erm_lp_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/home/home1_erm_lp_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/home/home2_erm_lp_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/home/home3_erm_lp_0926/ --dataset OfficeHome --test_env ${test_env} --key_uniq args.hpstep --algorithm ERM >> ../slurmconfig/home0926/results/lthp_home01.md
python3 -m domainbed.scripts.list_top_hparams --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/home/home0_erm_lplw_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/home/home1_erm_lplw_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/home/home2_erm_lplw_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/home/home3_erm_lplw_0926/ --dataset OfficeHome --test_env ${test_env} --key_uniq args.hpstep --algorithm MA >> ../slurmconfig/home0926/results/lthp_home01.md
python3 -m domainbed.scripts.list_top_hparams --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/home/home0_twama_lpl4w0a_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/home/home1_twama_lpl4w0a_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/home/home2_twama_lpl4w0a_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/home/home3_twama_lpl4w0a_0926  --dataset OfficeHome --test_env ${test_env} --key_uniq args.hpstep --algorithm TWAMA >> ../slurmconfig/home0926/results/lthp_home01.md
python3 -m domainbed.scripts.list_top_hparams --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/home/home0_twama_lpl4w0_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/home/home1_twama_lpl4w0_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/home/home2_twama_lpl4w0_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/home/home3_twama_lpl4w0_0926  --dataset OfficeHome --test_env ${test_env} --key_uniq args.hpstep --algorithm TWAMA >> ../slurmconfig/home0926/results/lthp_home01.md
python3 -m domainbed.scripts.list_top_hparams --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/home/home0_twama_d_lpl4w0_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/home/home1_twama_d_lpl4w0_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/home/home2_twama_d_lpl4w0_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/home/home3_twama_d_lpl4w0_0926  --dataset OfficeHome --test_env ${test_env} --key_uniq args.hpstep --algorithm TWAMA >> ../slurmconfig/home0926/results/lthp_home01.md
done


python3 -m domainbed.scripts.list_top_hparams --dataset OfficeHome --test_env ${test_env} --key_uniq args.hpstep --algorithm TWAMA --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/home/home0_twama_d6_lpl4w0a_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/home/home1_twama_d6_lpl4w0a_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/home/home2_twama_d6_lpl4w0a_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/home/home3_twama_d6_lpl4w0a_0926
python3 -m domainbed.scripts.list_top_hparams --dataset OfficeHome --test_env ${test_env} --key_uniq args.hpstep --algorithm TWAMA --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/home/home0_twama_d6_lpl4w0_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/home/home1_twama_d6_lpl4w0_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/home/home2_twama_d6_lpl4w0_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/home/home3_twama_d6_lpl4w0_0926


for test_env in 0 1 2 3
do
python3 -m domainbed.scripts.list_top_hparams --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/home/home0_erm_lp_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/home/home1_erm_lp_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/home/home2_erm_lp_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/home/home3_erm_lp_0926/ --dataset OfficeHome --test_env ${test_env} --key_uniq args.hpstep --algorithm ERM  >> ../slurmconfig/home0926/results/lthp_home_d6.md
python3 -m domainbed.scripts.list_top_hparams --dataset OfficeHome --test_env ${test_env} --key_uniq args.hpstep --algorithm TWAMA --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/home/home0_twama_d6_lpl4w0a_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/home/home1_twama_d6_lpl4w0a_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/home/home2_twama_d6_lpl4w0a_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/home/home3_twama_d6_lpl4w0a_0926 >> ../slurmconfig/home0926/results/lthp_home_d6.md
python3 -m domainbed.scripts.list_top_hparams --dataset OfficeHome --test_env ${test_env} --key_uniq args.hpstep --algorithm TWAMA --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/home/home0_twama_d6_lpl4w0_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/home/home1_twama_d6_lpl4w0_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/home/home2_twama_d6_lpl4w0_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/home/home3_twama_d6_lpl4w0_0926 >> ../slurmconfig/home0926/results/lthp_home_d6.md
done



python3 -m domainbed.scripts.list_top_hparams --dataset OfficeHome --test_env 0 --key_uniq args.hpstep --algorithm ERM --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/home/home0_erm_lp_dn1.0_0926 >> ../slurmconfig/home0926/results/lthp_home_dn.md
python3 -m domainbed.scripts.list_top_hparams --dataset OfficeHome --test_env 0 --key_uniq args.hpstep --algorithm ERM --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/home/home0_erm_lp_dn0.10_0926 >> ../slurmconfig/home0926/results/lthp_home_dn.md
python3 -m domainbed.scripts.list_top_hparams --dataset OfficeHome --test_env 0 --key_uniq args.hpstep --algorithm ERM --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/home/home0_erm_lp_dn0.25_0926 >> ../slurmconfig/home0926/results/lthp_home_dn.md
python3 -m domainbed.scripts.list_top_hparams --dataset OfficeHome --test_env 0 --key_uniq args.hpstep --algorithm ERM --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/home/home0_erm_lp_dn1.5_0926 >> ../slurmconfig/home0926/results/lthp_home_dn.md
python3 -m domainbed.scripts.list_top_hparams --dataset OfficeHome --test_env 0 --key_uniq args.hpstep --algorithm ERM --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/home/home0_erm_lp_dn4.0_0926 >> ../slurmconfig/home0926/results/lthp_home_dn.md
python3 -m domainbed.scripts.list_top_hparams --dataset OfficeHome --test_env 0 --key_uniq args.hpstep --algorithm ERM --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/home/home0_erm_lp_dn2.0_0926 >> ../slurmconfig/home0926/results/lthp_home_dn.md
python3 -m domainbed.scripts.list_top_hparams --dataset OfficeHome --test_env 0 --key_uniq args.hpstep --algorithm ERM --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/home/home0_erm_lp_dn0_0926 >> ../slurmconfig/home0926/results/lthp_home_dn.md
python3 -m domainbed.scripts.list_top_hparams --dataset OfficeHome --test_env 0 --key_uniq args.hpstep --algorithm ERM --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/home/home0_erm_lp_dn0.5_0926 >> ../slurmconfig/home0926/results/lthp_home_dn.md
