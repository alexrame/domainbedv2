# new

* assistant

     68798      hard       assistant_hfdv2_kl.slurm     rame  R       0:01      1 zeppelin
     68765      hard assistant_hfb_kl.slurm     rame  R       0:01      1 aerosmith

     69036      hard infassistant_dklv2_steps_0412.     rame  R       0:43      1 led
     69035      hard infassistant_bkl_steps_0412.sl     rame  R       1:22      1 thin
     69062      hard infassistant_bklv2_steps_0414.     rame  R       0:00      1 thin

    896921   gpu_p13 assistant_multiv4_kl005_newpef  utr15kn  R       7:10      1 r9i1n8

* stacku
     68912      hard   stacku_rmeld_kl2_bs128.slurm     rame PD       0:00      1 (Resources)
     68910      hard   stacku_rmdvb_kl2_bs128.slurm     rame  R       1:46      1 top
     69060      hard stacku_gpt2_kl2_bs128_length32     rame  R       0:01      1 top

     69038      hard     infstacku_rmdvb_0414.slurm     rame  R       0:00      1 lizzy
     69037      hard      infstacku_gpt2_0414.slurm     rame  R       0:36      1 lizzy

* captioning
sbatch -A edr@v100 fts_rouge4bleu1.slurm
sbatch -A edr@v100 fts_rouge3bleu2.slurm
sbatch -A edr@v100 fts_rouge2bleu3.slurm

    900838   gpu_p13     jz_inf_fts_rougebleu.slurm  utr15kn  R       2:55      1 r7i1n2
    900902   gpu_p13      jz_inf_fts_initbleu.slurm  utr15kn  R       0:59      1 r9i2n4
    900929   gpu_p13     jz_inf_fts_initrouge.slurm  utr15kn PD       0:00      1 (None)
