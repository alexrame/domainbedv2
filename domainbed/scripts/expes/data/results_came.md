(pytorch) alexandrerame@devfair0751:~/domainbedv2$ python3 -m domainbed.scripts.collect_results --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/came/came*

-------- Dataset: WILDSCamelyon, model selection method: training-domain validation set
Algorithm             hospital_0            hospital_1            hospital_2            hospital_3            hospital_4            Avg
erm_lp_0926           96.7 +/- 0.0          95.0 +/- 0.0          96.9 +/- 0.0          96.3 +/- 0.0          X                     X
twama_r_lpl4w0_0926   96.5 +/- 0.0          94.8 +/- 0.0          96.7 +/- 0.0          95.2 +/- 0.0          X                     X
twama_r_lpl4w0a_0926  94.8 +/- 0.0          X                     X                     X                     X                     X
twama_rxrx_lp_0926    96.4 +/- 0.0          95.1 +/- 0.0          X                     X                     X                     X

(pytorch) alexandrerame@devfair0751:~/domainbedv2$ KEYACC=ma python3 -m domainbed.scripts.collect_results --input_dirs /private/home/alexandrerame/dataplace/experiments/domainbed/came/came*

-------- Dataset: WILDSCamelyon, model selection method: training-domain validation set
Algorithm             hospital_0            hospital_1            hospital_2            hospital_3            hospital_4            Avg
erm_lp_0926           96.7 +/- 0.0          94.8 +/- 0.0          96.1 +/- 0.0          95.0 +/- 0.0          X                     X
twama_r_lpl4w0_0926   96.3 +/- 0.0          94.7 +/- 0.0          96.2 +/- 0.0          95.2 +/- 0.0          X                     X
twama_r_lpl4w0a_0926  96.9 +/- 0.0          X                     X                     X                     X                     X
twama_rxrx_lp_0926    95.7 +/- 0.0          94.0 +/- 0.0          X                     X                     X                     X
