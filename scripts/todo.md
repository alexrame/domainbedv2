# new

     JOBID PARTITION                           NAME     USER ST       TIME  NODES NODELIST(REASON)
     68912      hard   stacku_rmeld_kl2_bs128.slurm     rame  R 1-01:17:42      1 zz
     68910      hard   stacku_rmdvb_kl2_bs128.slurm     rame  R 1-01:34:03      1 top
     69155      hard stacku_rmdvb_rmeld_kl2_bs128.s     rame PD       0:00      1 (Priority)
     69060      hard stacku_gpt2_kl2_bs128_length32     rame  R    4:45:35      1 top

     68798      hard       assistant_hfdv2_kl.slurm     rame  R 1-18:13:21      1 zeppelin
     68765      hard         assistant_hfb_kl.slurm     rame  R 1-20:52:33      1 aerosmith

* inference
     69035      hard infassistant_bkl_steps_0412.sl     rame  R       1:22      1 thin
     69062      hard infassistant_bklv2_steps_0414.     rame  R       0:00      1 thin

     69160      hard infassistant_dklv2_steps_0412.     rame PD       0:00      1 (Priority)


* jz

    896921   gpu_p13 assistant_multiv4_kl005_newpef  utr15kn  R       7:10      1 r9i1n8

sbatch -A edr@v100 fts_rouge4bleu1.slurm
sbatch -A edr@v100 fts_rouge3bleu2.slurm
sbatch -A edr@v100 fts_rouge2bleu3.slurm

    900902   gpu_p13      jz_inf_fts_initbleu.slurm  utr15kn  R       0:59      1 r9i2n4
