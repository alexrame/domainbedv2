# JZ captioning
     JOBID PARTITION                           NAME     USER ST       TIME  NODES NODELIST(REASON)
    705105   gpu_p13            assistant_hfg.slurm  utr15kn  R    6:31:58      1 r7i1n7
    705103   gpu_p13      assistant_hfb_kl005.slurm  utr15kn  R    6:32:09      1 r7i1n3
    705104   gpu_p13            assistant_hfd.slurm  utr15kn  R    6:32:09      1 r7i1n5
    716483   gpu_p13            assistant_hfb.slurm  utr15kn  R    2:41:50      1 r8i1n4

    740752   gpu_p13         assistant_hfb_v2.slurm  utr15kn  R       6:11      1 r8i1n4


    720727   gpu_p13         stack_gpt2_kl005.slurm  utr15kn  R       1:48      1 r7i5n8
    720726   gpu_p13        stack_rmvdb_kl005.slurm  utr15kn  R       1:53      1 r8i5n0
    720743   gpu_p13        stack_multi_kl005.slurm  utr15kn  R       0:18      1 r7i6n0

(pytorch) [utr15kn@jean-zay4: inf]$ sc
     JOBID PARTITION                           NAME     USER ST       TIME  NODES NODELIST(REASON)
    740752   gpu_p13         assistant_hfb_v2.slurm  utr15kn  R   12:39:30      1 r8i1n4
    758634   gpu_p13            assistant_hfl.slurm  utr15kn PD       0:00      1 (Priority)
    758090   gpu_p13 inf_stack_finetuned_step_0405_  utr15kn  R       0:02      1 r7i1n6
    758089   gpu_p13 inf_stack_finetuned_wa_0405_32  utr15kn  R       0:32      1 r8i1n2


(nlp) rame@hacienda:~/slurmconfig/trl/infassistant$ sc
c
     JOBID PARTITION                           NAME     USER ST       TIME  NODES NODELIST(REASON)
     68168      hard infstack_multi01steps_0406.slu     rame PD       0:00      1 (Resources)
     68167      hard infassistant_bd_steps_0406.slu     rame  R       3:10      1 zeppelin
     68161      hard infstack_wa0to1_0406_step241.s     rame  R       3:53      1 led
     68160      hard     infstack_wa0to1_0406.slurm     rame  R       3:56      1 led
     68159      hard  infstack_wainitto1_0406.slurm     rame  R       3:59      1 led
     68158      hard  infstack_wainitto0_0406.slurm     rame  R       4:02      1 lizzy
     68154      hard     infstack_0steps_0406.slurm     rame  R       7:00      1 thin
     68155      hard     infstack_1steps_0406.slurm     rame  R       7:00      1 thin
     68139      hard infassistant_b_steps_0406.slur     rame  R      23:41      1 zeppelin
     68169      hard infassistant_wainittodkl_0406.     rame PD       0:00      1 (Priority)




# new

* assistant
     68389      hard infassistant_multikl_steps_040     rame  R       0:00      1 lizzy
     68321      hard infassistant_bkl_steps_0406.sl     rame  R       0:20      1 zeppelin
     68390      hard infassistant_wabkltodkl_0410_1     rame  R       0:02      1 thin
     68233      hard         assistant_hfd_kl.slurm     rame  R   17:21:46      1 aerosmith
    810719   gpu_p13 assistant_hfb_kl005_v2_warmup.  utr15kn PD       0:00      1 (Resources)

* stack
     JOBID PARTITION                           NAME     USER ST       TIME  NODES NODELIST(REASON)
   68320      hard infstack_0steps_bs126_0406.slu     rame  R    2:15:11      1 led
     68290      hard               stack_gpt2.slurm     rame  R   13:01:11      1 top
    809767   gpu_p13          stack_rmeld_kl0.slurm  utr15kn  R      27:24      1 r7i3n7
     68400      hard stack_gpt2_bs64_kl005_lr_v2_wa     rame  R       0:48      1 zz
     68397      hard stack_gpt2_bs64_kl005_lr_v2.sl     rame  R      16:46      1 top

# captioning fts

## train

    810084   gpu_p13               fts_bleuv2.slurm  utr15kn PD       0:00      1 (None)
    810003   gpu_p13         fts_rouge1bleu05.slurm  utr15kn  R       3:21      1 r8i6n7
    810074   gpu_p13              fts_rougev2.slurm  utr15kn  R       0:23      1 r9i6n6

## inf
    810108   gpu_p13 jz_inf_fts_rougebleu_predens.s  utr15kn  R       0:02      1 r8i6n1
    810162   gpu_p13 jz_inf_fts_bleustep5bleustep10  utr15kn PD       0:00      1 (Resources)
    810179   gpu_p13 jz_inf_fts_rougestep5rougestep  utr15kn PD       0:00      1 (None)

## todo inf

bleu step 5 to other step 5
bleu step 5 to other step 5

