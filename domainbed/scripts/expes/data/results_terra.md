
python3 -m domainbed.scripts.collect_results --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra0_erm_lp_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra1_erm_lp_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra2_erm_lp_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra3_erm_lp_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra0_erm_lp_dn0_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra1_erm_lp_dn0_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra2_erm_lp_dn0_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra3_erm_lp_dn0_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra0_ermf_lp_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra1_ermf_lp_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra2_ermf_lp_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra3_ermf_lp_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra0_erm_lplw_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra1_erm_lplw_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra2_erm_lplw_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra3_erm_lplw_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra0_twama_lpl4_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra1_twama_lpl4_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra2_twama_lpl4_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra3_twama_lpl4_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra0_twama_lpl4w0_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra1_twama_lpl4w0_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra2_twama_lpl4w0_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra3_twama_lpl4w0_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra0_twama_lpl4w0a_i_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra1_twama_lpl4w0a_i_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra2_twama_lpl4w0a_i_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra3_twama_lpl4w0a_i_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra0_twama_lpl4w0_i_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra1_twama_lpl4w0_i_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra2_twama_lpl4w0_i_0926 /private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra3_twama_lpl4w0_i_0926

-------- Linear probing
Algorithm             L100                  L38                   L43                   L46                   Avg
noft_LP               0.2973899288          0.3182693542          0.3639798489          0.3074144891
noft_LPLW             0.3820195096          0.3739889588          0.4445843829          0.2999787550


-------- ERM

Algorithm             L100                  L38                   L43                   L46                   Avg
erm_lp_0926           58.8 +/- 0.0          42.1 +/- 0.0          55.6 +/- 0.0          42.2 +/- 0.0          49.7
erm_lp_dn0_0926       51.0 +/- 0.0          35.4 +/- 0.0          51.9 +/- 0.0          33.7 +/- 0.0          43.0
ermf_lp_0926          60.8 +/- 0.0          35.0 +/- 0.0          58.8 +/- 0.0          35.1 +/- 0.0          47.4
erm_lplw_0926         48.5 +/- 0.0          40.9 +/- 0.0          55.9 +/- 0.0          38.4 +/- 0.0          45.9
twama_lpl4_0926       50.5 +/- 0.0          44.6 +/- 0.0          51.6 +/- 0.0          41.9 +/- 0.0          47.2
twama_lpl4w0_0926     58.6 +/- 0.0          47.6 +/- 0.0          51.0 +/- 0.0          37.9 +/- 0.0          48.8
twama_lpl4w0a_i_0926  60.6 +/- 0.0          45.7 +/- 0.0          55.8 +/- 0.0          35.7 +/- 0.0          49.4
twama_lpl4w0_i_0926   48.7 +/- 0.0          51.5 +/- 0.0          51.5 +/- 0.0          39.6 +/- 0.0          47.8

-------- KEYTEST=val
erm_lp_0926           90.4 +/- 0.0          91.8 +/- 0.0          92.0 +/- 0.0          93.2 +/- 0.0          91.8
erm_lp_dn0_0926       90.8 +/- 0.0          92.0 +/- 0.0          92.6 +/- 0.0          93.3 +/- 0.0          92.2
ermf_lp_0926          90.6 +/- 0.0          91.6 +/- 0.0          92.3 +/- 0.0          93.2 +/- 0.0          91.9
erm_lplw_0926         90.9 +/- 0.0          91.8 +/- 0.0          92.9 +/- 0.0          93.7 +/- 0.0          92.3
twama_lpl4_0926       90.7 +/- 0.0          91.8 +/- 0.0          92.3 +/- 0.0          93.3 +/- 0.0          92.0
twama_lpl4w0_0926     90.5 +/- 0.0          91.7 +/- 0.0          92.6 +/- 0.0          93.4 +/- 0.0          92.1
twama_lpl4w0a_i_0926  90.9 +/- 0.0          91.4 +/- 0.0          92.1 +/- 0.0          93.3 +/- 0.0          91.9
twama_lpl4w0_i_0926   90.9 +/- 0.0          91.4 +/- 0.0          92.2 +/- 0.0          93.4 +/- 0.0          92.0

-------- KEYTACC=ma
erm_lplw_0926         55.2 +/- 0.0          48.3 +/- 0.0          60.3 +/- 0.0          43.5 +/- 0.0          51.8
twama_lpl4_0926       58.6 +/- 0.0          47.2 +/- 0.0          60.3 +/- 0.0          43.4 +/- 0.0          52.4
twama_lpl4w0_0926     57.5 +/- 0.0          47.2 +/- 0.0          61.5 +/- 0.0          42.6 +/- 0.0          52.2
twama_lpl4w0a_i_0926  60.8 +/- 0.0          46.9 +/- 0.0          59.9 +/- 0.0          43.7 +/- 0.0          52.8
twama_lpl4w0_i_0926   56.2 +/- 0.0          47.0 +/- 0.0          58.0 +/- 0.0          41.2 +/- 0.0          50.6

-------- DiWA
D_LPLW                & 57.5 & 47.6 & 59.9 & 42.6 & 51.9
D_LP LPLW 40          & 58.3 & 49.3 & 60.2 & 40.0 & 52.0
D_LP LPLW top10       & 57.6 & 49.8 & 60.1 & 41.9 & 52.3
D_TWA                 & 59.7 & 49.5 & 61.8 & 41.1 & 53.0

D_TWAW0               & 60.2 & 48.0 & 60.5 & 39.9 & 52.2
lpl4w0a_i_0926        & 59.9 & 48.9 & 60.9 & 40.2 & 52.5
twama_lpl4w0_i_0926   & 58.5 & 49.3 & 60.6 & 41.3 & 52.4


-------- DiWAMA

DMA_LPLW              & 58.2 & 48.7 & 62.1 & 44.8 & 53.4
D_ERMLP               & 57.7 & 49.4 & 60.5 & 39.5 & 51.8
DMA_TWAMA             & 57.6 & 49.1 & 62.0 & 43.2 & 53.0
DMA_TWAW0             & 57.4 & 48.7 & 61.8 & 43.3 & 52.8
malpl4w0a_i_0926      & 58.9 & 49.5 & 62.1 & 42.3 & 53.2
matwama_lpl4w0_i_0926 & 58.5 & 49.8 & 61.8 & 43.5 & 53.4


# Diversity
(pytorch) alexandrerame@devfair0751:~/slurmconfig/terra0926/inf/runs$ cat ensterra_twama_lpl4w0_i_0926.slurm_66339715.out | grep printres
printres:  {'acc': 0.5935456654714195, 'acc_ens': 0.585108626871968, 'acc_netm': 0.5322294874499052, 'divq_netm': 0.8903464597981359, 'divr_netm': 0.5523191896447147, 'length': 20, 'step': 'last', 'testenv': 0, 'topk': 0, 'train_acc': 0.9218789890222109, 'train_acc_ens': 0.9456216492213428, 'train_acc_netm': 0.9124457492979321, 'train_divq_netm': 0.9300381267131549, 'train_divr_netm': 1.6292880378913828}
printres:  {'acc': 0.491885784716516, 'acc_ens': 0.44546014790468363, 'acc_netm': 0.4307980690221857, 'divq_netm': 0.9598898308543808, 'divr_netm': 0.27297235948103415, 'length': 20, 'step': 'last', 'testenv': 1, 'topk': 0, 'train_acc': 0.9259766963673749, 'train_acc_ens': 0.9393420150788211, 'train_acc_netm': 0.9123886223440711, 'train_divq_netm': 0.9406401005146799, 'train_divr_netm': 1.452933377855641}
printres:  {'acc': 0.6017632241813602, 'acc_ens': 0.615617128463476, 'acc_netm': 0.5623047858942065, 'divq_netm': 0.8806236288862441, 'divr_netm': 0.5914471132398491, 'length': 20, 'step': 'last', 'testenv': 2, 'topk': 0, 'train_acc': 0.9285187914517318, 'train_acc_ens': 0.9476787030213707, 'train_acc_netm': 0.9213829525915008, 'train_divq_netm': 0.9471038374453394, 'train_divr_netm': 1.4719227602062592}
printres:  {'acc': 0.4055753867074622, 'acc_ens': 0.4139044705082441, 'acc_netm': 0.38431922488526266, 'divq_netm': 0.8901495403555966, 'divr_netm': 0.36846366991971197, 'length': 20, 'step': 'last', 'testenv': 3, 'topk': 0, 'train_acc': 0.9392789373814042, 'train_acc_ens': 0.9606939550013553, 'train_acc_netm': 0.9349146110056925, 'train_divq_netm': 0.9479637147427326, 'train_divr_netm': 1.673843440120325}
printres:  {'acc': 0.5859523307319131, 'acc_ens': 0.5853195528369542, 'acc_netm': 0.5636152710398651, 'divq_netm': 0.9540546721210148, 'divr_netm': 0.36880862681261345, 'length': 20, 'step': 'last', 'testenv': 0, 'topk': 0, 'train_acc': 0.9091141179474087, 'train_acc_ens': 0.9361756446259892, 'train_acc_netm': 0.9264871074802145, 'train_divq_netm': 0.9878964377268663, 'train_divr_netm': 0.6510667005623533}


(pytorch) alexandrerame@devfair0751:~/slurmconfig/terra0926/inf/runs$ cat ensterra_erm_lplw_0926.slurm_66340020.out | grep printres
printres:  {'acc': 0.5756169584475849, 'acc_ens': 0.5840539970470365, 'acc_netm': 0.5147226323560429, 'divq_netm': 0.8519891117223829, 'divr_netm': 0.6573583681130651, 'length': 10, 'step': 'last', 'testenv': 0, 'topk': -10, 'train_acc': 0.9323461833035487, 'train_acc_ens': 0.9425580801633904, 'train_acc_netm': 0.9174623436303294, 'train_divq_netm': 0.9405907092565455, 'train_divr_netm': 1.5191850378080571}
printres:  {'acc': 0.47709531635168445, 'acc_ens': 0.43488085456039444, 'acc_netm': 0.42206244864420706, 'divq_netm': 0.9635233529599329, 'divr_netm': 0.25435380609052216, 'length': 10, 'step': 'last', 'testenv': 1, 'topk': -10, 'train_acc': 0.9294037011651817, 'train_acc_ens': 0.9366004112405757, 'train_acc_netm': 0.9159355723098013, 'train_divq_netm': 0.950344155117407, 'train_divr_netm': 1.3344484787567086}
printres:  {'acc': 0.6022670025188916, 'acc_ens': 0.5949622166246852, 'acc_netm': 0.5554156171284635, 'divq_netm': 0.8952154196342467, 'divr_netm': 0.5390237098933104, 'length': 10, 'step': 'last', 'testenv': 2, 'topk': -10, 'train_acc': 0.9405551461557357, 'train_acc_ens': 0.9491525423728814, 'train_acc_netm': 0.9244166052566936, 'train_divq_netm': 0.9486325217503626, 'train_divr_netm': 1.4920939533179984}
printres:  {'acc': 0.3929967703552609, 'acc_ens': 0.39673635900050996, 'acc_netm': 0.36865544790073085, 'divq_netm': 0.9053620071877881, 'divr_netm': 0.33918122452950217, 'length': 10, 'step': 'last', 'testenv': 3, 'topk': -10, 'train_acc': 0.9528327460016265, 'train_acc_ens': 0.959880726484142, 'train_acc_netm': 0.9373542965573327, 'train_divq_netm': 0.9481902169532513, 'train_divr_netm': 1.6945942698578307}
(pytorch) alexandrerame@devfair0751:~/slurmconfig/terra0926/inf/runs$ cat ensterra_erm_lp_0926.slurm_66339695.out | grep printres
printres:  {'acc': 0.5996625184560219, 'acc_ens': 0.5899599240666527, 'acc_netm': 0.5411200168740773, 'divq_netm': 0.8623784436994459, 'divr_netm': 0.653839874968502, 'length': 20, 'step': 'last', 'testenv': 0, 'topk': 0, 'train_acc': 0.9206025019147307, 'train_acc_ens': 0.9374521317334695, 'train_acc_netm': 0.9109394945111055, 'train_divq_netm': 0.9374044249185305, 'train_divr_netm': 1.4881244031784853}
printres:  {'acc': 0.49465899753492193, 'acc_ens': 0.4494658997534922, 'acc_netm': 0.4402783483976993, 'divq_netm': 0.9685541790369809, 'divr_netm': 0.23744586838977283, 'length': 20, 'step': 'last', 'testenv': 1, 'topk': 0, 'train_acc': 0.9201507882111035, 'train_acc_ens': 0.9372858122001371, 'train_acc_netm': 0.9103666895133653, 'train_divq_netm': 0.9404597100991413, 'train_divr_netm': 1.4412026540466172}
printres:  {'acc': 0.6055415617128463, 'acc_ens': 0.6128463476070529, 'acc_netm': 0.556448362720403, 'divq_netm': 0.8723446024589261, 'divr_netm': 0.6085306056184655, 'length': 20, 'step': 'last', 'testenv': 2, 'topk': 0, 'train_acc': 0.9287644313436502, 'train_acc_ens': 0.9476787030213707, 'train_acc_netm': 0.9188037337263572, 'train_divq_netm': 0.9410250463666856, 'train_divr_netm': 1.5332003772464466}
printres:  {'acc': 0.3885772565017848, 'acc_ens': 0.4052354241033486, 'acc_netm': 0.37572667006629273, 'divq_netm': 0.9085812808507339, 'divr_netm': 0.32332018792274536, 'length': 20, 'step': 'last', 'testenv': 3, 'topk': 0, 'train_acc': 0.937381404174573, 'train_acc_ens': 0.9555435077256709, 'train_acc_netm': 0.9306993765248034, 'train_divq_netm': 0.9478018253823611, 'train_divr_netm': 1.588103716517099}
