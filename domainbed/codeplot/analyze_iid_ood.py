

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

for test_env in 0 1 2 3
do

rm ../slurmconfig/terra0926/results/lthp_terra${test_env}.md
python3 -m domainbed.scripts.list_top_hparams --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra0_erm_lp_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra1_erm_lp_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra2_erm_lp_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra3_erm_lp_0926/ --dataset TerraIncognita --test_env ${test_env} --algorithm ERM >> ../slurmconfig/terra0926/results/lthp_terra${test_env}.md


python3 -m domainbed.scripts.list_top_hparams --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra0_erm_lplw_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra1_erm_lplw_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra2_erm_lplw_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra3_erm_lplw_0926/ --dataset TerraIncognita --test_env ${test_env} --algorithm MA >> ../slurmconfig/terra0926/results/lthp_terra${test_env}.md

python3 -m domainbed.scripts.list_top_hparams --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra0_twama_lpl4_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra1_twama_lpl4_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra2_twama_lpl4_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra3_twama_lpl4_0926/ --dataset TerraIncognita --test_env ${test_env} --algorithm TWAMA >> ../slurmconfig/terra0926/results/lthp_terra${test_env}.md

python3 -m domainbed.scripts.list_top_hparams --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra0_twama_lpl4w0_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra1_twama_lpl4w0_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra2_twama_lpl4w0_0926/ /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra3_twama_lpl4w0_0926/ --dataset TerraIncognita --test_env ${test_env} --algorithm TWAMA >> ../slurmconfig/terra0926/results/lthp_terra${test_env}.md

done

