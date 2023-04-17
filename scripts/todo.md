# new

* infstack
     69458      hard infstacku_wainitrmdvb311_0415.     rame  R       0:00      1 thin
     69457      hard infstacku_wainitrmeld391_0415.     rame  R       0:22      1 led


     69462      hard      infstacku_gpt2_0417.slurm     rame  R    6:43:19      1 zeppelin
     69522      hard infstacku_warmdvb311rmeld391_0     rame  R    2:52:47      1 lizzy
     69521      hard infstacku_multirmeldrmdvb_0415     rame  R    2:57:12      1 led

* assistant
     69590      hard     assistant_rmdvlv_kl2.slurm     rame PD       0:00      1 (Priority)
     69577      hard     assistant_rmdvbv_kl2.slurm     rame  R      33:33      1 lizzy
     69575      hard      assistant_rmeld_kl2.slurm     rame  R      40:25      1 lizzy

* infassitant
     69587      hard infassistant_bv2_steps_0417.sl     rame PD       0:00      1 (Priority)
     69589      hard infassistant_dv2_steps_0417.sl     rame PD       0:00      1 (Priority)

* jz
    938092   gpu_p13 assistant_multirmeldrmdvbv.slu  utr15kn PD       0:00      1 (Priority)
    938021   gpu_p13 assistant_multirmeldrmdvlv.slu  utr15kn  R       0:29      1 r9i6n0
    934355   gpu_p13   assistant_dvxrposv2_kl.slurm  utr15kn  R    1:14:07      1 r7i1n8


cat infstacku_wainitrmeld391_0415.slurm-69457.out | grep "d\[" >> /data/rame/logs_experiments_notebook/nlp/llamastacku/infstacku_wainitrmeld391_0415.py
cat infstacku_wainitrmdvb311_0415.slurm-69458.out | grep "d\[" >> /data/rame/logs_experiments_notebook/nlp/llamastacku/infstacku_wainitrmdvb311_0415.py
