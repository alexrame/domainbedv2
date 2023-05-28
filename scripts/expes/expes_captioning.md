# TODO Program

The weights should be from epoch5


* Launch eval from jz
PermissionError: [Errno 13] Permission denied: 'java'
https://gist.github.com/AbeHandler/54b32b77d63bd6629d4bcf4e718fd201


* Launch fts



# Main

```
srun -p hard --gpus-per-node=1 -t 1400 --pty bash
srun -p hard -w aerosmith --gpus-per-node=1 -t 2400 --pty bash
export DATA_DIR=/data/rame/ExpansionNet_v2/github_ignore_material
```

## eval


python test.py --features_path ${DATA_DIR}/raw_data/features_rf.hdf5 --is_end_to_end False --coeffs [0.5,0.5] --save_model_path /data/rame/ExpansionNet_v2/github_ignore_material/saves/ftsbleubs18lr1e-5/checkpoint_2023-03-07-21:05:09_epoch5it6293bs18_bleu_.pth /data/rame/ExpansionNet_v2/github_ignore_material/saves/ftsrougebs18lr1e-5/checkpoint_2023-03-07-21:07:31_epoch5it6293bs18_rouge_.pth



>> python test.py --is_end_to_end False --features_path ${DATA_DIR}/raw_data/features_rf.hdf5 --save_model_path /data/rame/ExpansionNet_v2/github_ignore_material/saves/ftsbleu1rouge05bs18lr1e-5/checkpoint_2023-03-09-04:34:58_epoch8it6293bs18_bleu-1,rouge-0.5_.pth

>> python test.py --is_end_to_end True --save_model_path ${DATA_DIR}/saves/rf_model-002.pth

>> python test.py --is_end_to_end True --ensemble wa --coeffs [-0.1,1.1]  --save_model_path ${DATA_DIR}/saves/wa/model_bleu_bs12_epoch3.pth ${DATA_DIR}/saves/wa/model_rougebs18_epoch5.pth ${DATA_DIR}/saves/wa/model_cider.pth

bash eval_wa.sh /data/rame/ExpansionNet_v2/github_ignore_material/saves/wa/model_bleu_bs12_epoch1.pth /data/rame/ExpansionNet_v2/github_ignore_material/saves/wa/model_bleu_bs12_epoch3.pth

## fts

>> python train.py --optim_type radam --seed 775533 --sched_type custom_warmup_anneal  \
    --warmup 1 --anneal_coeff 0.8 --anneal_every_epoch 1 --lr 1e-5 --batch_size 18 \
    --is_end_to_end False --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/ --num_accum 2\
    --body_save_path ${DATA_DIR}/saves/rf_model-002.pth --num_epochs 9\
    --partial_load True     --features_path ${DATA_DIR}/raw_data/features_rf.hdf5\
    --save_path ${DATA_DIR}/saves/ftsrougebs18lr1e-5/ --reinforce rouge

python train.py --optim_type radam --seed 775533 --sched_type custom_warmup_anneal  \
    --warmup 1 --anneal_coeff 0.8 --anneal_every_epoch 1 --lr 1e-5 --batch_size 18 \
    --is_end_to_end False --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/ --num_accum 2\
    --body_save_path ${DATA_DIR}/saves/rf_model-002.pth --num_epochs 9\
    --partial_load True     --features_path ${DATA_DIR}/raw_data/features_rf.hdf5\
    --save_path ${DATA_DIR}/saves/ftsbleu1rouge05bs18lr1e-5/ --reinforce bleu-1,rouge-0.5

Remarks:
- 1-4 too big
- 1-6 good but does not change much
- 1e-5 good

## e2e
python train.py --optim_type radam --seed 775533 --sched_type custom_warmup_anneal --warmup 1 --anneal_coeff 1.0 --lr 5e-6 --batch_size 18 --num_accum 2 --is_end_to_end True --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/ --body_save_path ${DATA_DIR}/saves/rf_model-002.pth --num_epochs 6 --save_path ${DATA_DIR}/saves/e2eciderbs18lr5e-6/ --reinforce cider --print_every_iter 100


/gpfswork/rech/edr/utr15kn/conda/envs/pytorch/bin/python3 /gpfsdswork/projects/rech/edr/utr15kn/domainbedv2/captioning/train.py\
 --optim_type radam --seed 775533 --sched_type custom_warmup_anneal  \
 --warmup 1 --anneal_coeff 1.0 --lr 5e-6 --batch_size 18 --num_accum 2\
 --is_end_to_end True --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/\
 --body_save_path ${DATA_DIR}/saves/rf_model-002.pth --num_epochs 6\
 --body_save_path ${DATA_DIR}/saves/wa/model_cider.pth\
 --num_epochs 6\
     --save_path ${DATA_DIR}/saves/e2eciderbs18lr5e-6/ --reinforce cider


Remarks
- 5e-5 too big
- 1e-5 was good for blue ?
- trying 5e-6, little too small



# Olds
## test xe

>> python test.py --N_enc 3 --N_dec 3 --model_dim 512 --num_gpus 1 --eval_beam_sizes [5] --is_end_to_end True --save_model_path ${DATA_DIR}/saves/xe_model-003.pth --ddp_sync_port 12345

Evaluation on Validation Set
Evaluation Phase over 5000 BeamSize: 5  elapsed: 28 m 10 s
[('CIDEr', 1.2784), ('Bleu_1', 0.7755), ('Bleu_2', 0.6235), ('Bleu_3', 0.4899), ('Bleu_4', 0.3827), ('ROUGE_L', 0.5893), ('SPICE', 0.2325), ('METEOR', 0.3025)]
Evaluation on Test Set
Evaluation Phase over 5000 BeamSize: 5  elapsed: 27 m 7 s
[('CIDEr', 1.2736), ('Bleu_1', 0.771), ('Bleu_2', 0.6166), ('Bleu_3', 0.4831), ('Bleu_4', 0.3756), ('ROUGE_L', 0.5851), ('SPICE', 0.2348), ('METEOR', 0.3006)]


# test rf

>> python test.py --N_enc 3 --N_dec 3 --model_dim 512 --num_gpus 1 --eval_beam_sizes [5] --is_end_to_end True --save_model_path ${DATA_DIR}/saves/rf_model-002.pth
[('CIDEr', 1.3965), ('Bleu_1', 0.8321), ('Bleu_2', 0.6864), ('Bleu_3', 0.5408), ('Bleu_4', 0.4168), ('ROUGE_L', 0.6059), ('SPICE', 0.2428), ('METEOR', 0.3036)]
[('CIDEr', 1.3958), ('Bleu_1', 0.8271), ('Bleu_2', 0.6772), ('Bleu_3', 0.533), ('Bleu_4', 0.4098), ('ROUGE_L', 0.6034), ('SPICE', 0.2441), ('METEOR', 0.3016)]


>> python test.py --is_end_to_end False --save_model_path ${DATA_DIR}/saves/rf_model-002.pth --features_path ${DATA_DIR}/raw_data/features_rf.hdf5
[('CIDEr', 1.3706), ('Bleu_1', 0.8246), ('Bleu_2', 0.6769), ('Bleu_3', 0.531), ('Bleu_4', 0.4079), ('ROUGE_L', 0.6004), ('SPICE', 0.2395), ('METEOR', 0.2993)]
Evaluation on Test Set
[('CIDEr', 1.3729), ('Bleu_1', 0.8227), ('Bleu_2', 0.6703), ('Bleu_3', 0.526), ('Bleu_4', 0.4045), ('ROUGE_L', 0.5984), ('SPICE', 0.2401), ('METEOR', 0.2975)]








## eval


python test.py --N_enc 3 --N_dec 3 --model_dim 512 --num_gpus 1 --eval_beam_sizes [5] --is_end_to_end True --ensemble wa_5 --save_model_path /data/rame/ExpansionNet_v2/github_ignore_material/saves/e2erouge/rouge_model_epoch5.pth /data/rame/ExpansionNet_v2/github_ignore_material/saves/wa/bleu_model_epoch3.pth

python test.py --N_enc 3 --N_dec 3 --model_dim 512 --num_gpus 1 --eval_beam_sizes [5] --is_end_to_end True --save_model_path /data/rame/ExpansionNet_v2/github_ignore_material/saves/wa/

--ensemble wa_10
[('CIDEr', 1.3965), ('Bleu_1', 0.8321), ('Bleu_2', 0.6864), ('Bleu_3', 0.5408), ('Bleu_4', 0.4168), ('ROUGE_L', 0.6059), ('SPICE', 0.2428), ('METEOR', 0.3036)]

--ensemble ens
[('CIDEr', 1.3936), ('Bleu_1', 0.8462), ('Bleu_2', 0.6963), ('Bleu_3', 0.5468), ('Bleu_4', 0.42), ('ROUGE_L', 0.6081), ('SPICE', 0.2421), ('METEOR', 0.3019)]

--ensemble wa
[('CIDEr', 1.3874), ('Bleu_1', 0.845), ('Bleu_2', 0.6939), ('Bleu_3', 0.5429), ('Bleu_4', 0.4149), ('ROUGE_L', 0.6055), ('SPICE', 0.2436), ('METEOR', 0.3015)]

--ensemble wa_2
[('CIDEr', 1.3656), ('Bleu_1', 0.8551), ('Bleu_2', 0.6961), ('Bleu_3', 0.5393), ('Bleu_4', 0.4085), ('ROUGE_L', 0.602), ('SPICE', 0.2384), ('METEOR', 0.2965)]

--ensemble wa_0
[('CIDEr', 1.3389), ('Bleu_1', 0.8572), ('Bleu_2', 0.6917), ('Bleu_3', 0.5305), ('Bleu_4', 0.3982), ('ROUGE_L', 0.5972), ('SPICE', 0.2315), ('METEOR', 0.2913)]
# fts

## bleu2
python train.py --N_enc 3 --N_dec 3 --model_dim 512 --optim_type radam --seed 775533  --sched_type custom_warmup_anneal      --warmup 1 --lr 1e-4 --anneal_coeff 0.8 --anneal_every_epoch 1 --enc_drop 0.1     --
dec_drop 0.1 --enc_input_drop 0.1 --dec_input_drop 0.1 --drop_other 0.1      --batch_size 24 --num_accum 2 --num_gpus 1 --ddp_sync_port 11318 --eval_beam_sizes [5]      --save_path ${DATA_DIR}/saves/fts/ --save_every_minutes 60 --how_many_checkpoints 1
      --is_end_to_end False --partial_load True     --features_path ${DATA_DIR}/raw_data/features_rf.hdf5     --body_save_path ${DATA_DIR}/saves/rf_model-002.pth     --print_every_iter 2000 --eval_every_iter 99999     --reinforce bleu2 --num_epochs 9

[('CIDEr', 1.3484), ('Bleu_1', 0.8364), ('Bleu_2', 0.6922), ('Bleu_3', 0.5431), ('Bleu_4', 0.4156), ('ROUGE_L', 0.6041), ('SPICE', 0.2386), ('METEOR', 0.2971)]


# bleu4

export DATA_DIR=/data/rame/ExpansionNet_v2/github_ignore_material
python train.py --N_enc 3 --N_dec 3  \
    --model_dim 512 --optim_type radam --seed 775533  --sched_type custom_warmup_anneal  \
    --warmup 1 --lr 1e-4 --anneal_coeff 0.8 --anneal_every_epoch 1 --enc_drop 0.1 \
    --dec_drop 0.1 --enc_input_drop 0.1 --dec_input_drop 0.1 --drop_other 0.1  \
    --batch_size 16 --num_accum 2 --num_gpus 1 --ddp_sync_port 11317 --eval_beam_sizes [5]  \
     --save_every_minutes 60 --how_many_checkpoints 1  \
    --is_end_to_end False --partial_load True \
    --print_every_iter 2000 --eval_every_iter 99999 --num_epochs 9 \
    --features_path ${DATA_DIR}/raw_data/features_rf.hdf5 \
    --body_save_path ${DATA_DIR}/saves/rf_model-002.pth \
    --save_path ${DATA_DIR}/saves/ftsbleu4/ --reinforce bleu4

Evaluation on Validation Set
Evaluation Phase over 5000 BeamSize: 5  elapsed: 2 m 38 s
[('CIDEr', 1.2424), ('Bleu_1', 0.7909), ('Bleu_2', 0.6505), ('Bleu_3', 0.5138), ('Bleu_4', 0.3955), ('ROUGE_L', 0.5891), ('SPICE', 0.2244), ('METEOR', 0.287)]
Saved to checkpoint_2023-03-01-14:45:03_epoch1it7080bs16_bleu4_.pth

# rouge

export DATA_DIR=/data/rame/ExpansionNet_v2/github_ignore_material
python train.py --N_enc 3 --N_dec 3  \
    --model_dim 512 --optim_type radam --seed 775533  --sched_type custom_warmup_anneal  \
    --warmup 1 --lr 1e-4 --anneal_coeff 0.8 --anneal_every_epoch 1 --enc_drop 0.1 \
    --dec_drop 0.1 --enc_input_drop 0.1 --dec_input_drop 0.1 --drop_other 0.1  \
    --batch_size 16 --num_accum 2 --num_gpus 1 --ddp_sync_port 11317 --eval_beam_sizes [5]  \
     --save_every_minutes 60 --how_many_checkpoints 1  \
    --is_end_to_end False --partial_load True \
    --print_every_iter 2000 --eval_every_iter 99999 --num_epochs 9 \
    --features_path ${DATA_DIR}/raw_data/features_rf.hdf5 \
    --body_save_path ${DATA_DIR}/saves/rf_model-002.pth \
    --save_path ${DATA_DIR}/saves/ftsrouge/ --reinforce rouge

## meteor
export DATA_DIR=/data/rame/ExpansionNet_v2/github_ignore_material
python train.py --N_enc 3 --N_dec 3  \
    --model_dim 512 --optim_type radam --seed 775533  --sched_type custom_warmup_anneal  \
    --warmup 1 --lr 1e-4 --anneal_coeff 0.8 --anneal_every_epoch 1 --enc_drop 0.1 \
    --dec_drop 0.1 --enc_input_drop 0.1 --dec_input_drop 0.1 --drop_other 0.1  \
    --batch_size 16 --num_accum 2 --num_gpus 1 --ddp_sync_port 11317 --eval_beam_sizes [5]  \
     --save_every_minutes 60 --how_many_checkpoints 1  \
    --is_end_to_end False --partial_load True \
    --print_every_iter 2000 --eval_every_iter 99999 --num_epochs 9 \
    --features_path ${DATA_DIR}/raw_data/features_rf.hdf5 \
    --body_save_path ${DATA_DIR}/saves/rf_model-002.pth \
    --save_path ${DATA_DIR}/saves/ftsmeteor/ --reinforce meteor
# e2e

## bleu short

(pytorch) rame@aerosmith:~/ExpansionNet_v2$ python train.py --N_enc 3 --N_dec 3  \
>     --model_dim 512 --optim_type radam --seed 775533 --sched_type custom_warmup_anneal  \
>     --warmup 1 --anneal_coeff 1.0 --lr 1e-6 --enc_drop 0.1 \
>     --dec_drop 0.1 --enc_input_drop 0.1 --dec_input_drop 0.1 --drop_other 0.1  \
>     --batch_size 12 --num_accum 2 --num_gpus 1 --ddp_sync_port 11317 --eval_beam_sizes [5]  \
>     --save_path ${DATA_DIR}/saves/ftse2e/ --save_every_minutes 60 --how_many_checkpoints 1  \
>     --is_end_to_end True --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/ --partial_load True \
>     --body_save_path ${DATA_DIR}/saves/rf_model-002.pth \
>     --print_every_iter 1500 --eval_every_iter 999999 \
>     --reinforce bleu --num_epochs 1

Evaluation on Validation Set
Evaluation Phase over 5000 BeamSize: 5  elapsed: 4 m 27 s
[('CIDEr', 1.3881), ('Bleu_1', 0.8373), ('Bleu_2', 0.688), ('Bleu_3', 0.5395), ('Bleu_4', 0.4138), ('ROUGE_L', 0.6049), ('SPICE', 0.2433), ('METEOR', 0.3015)]
Saved to checkpoint_2023-02-28-22:06:50_epoch0it9440bs12_bleu_.pth
[('CIDEr', 1.3881), ('Bleu_1', 0.8373), ('Bleu_2', 0.688), ('Bleu_3', 0.5395), ('Bleu_4', 0.4138), ('ROUGE_L', 0.6049), ('SPICE', 0.2433), ('METEOR', 0.3015)]
[('CIDEr', 1.3869), ('Bleu_1', 0.8348), ('Bleu_2', 0.6817), ('Bleu_3', 0.5349), ('Bleu_4', 0.4095), ('ROUGE_L', 0.603), ('SPICE', 0.2445), ('METEOR', 0.2995)]


python test.py --N_enc 3 --N_dec 3 --model_dim 512 --num_gpus 1 --eval_beam_sizes [5] --is_end_to_end True --save_model_path ${DATA_DIR}/saves/ftse2echeckpoint_2023-02-28-22:06:50_epoch0it9440bs12_bleu_.pth

## bleu long


python test.py --N_enc 3 --N_dec 3 --model_dim 512 --num_gpus 1 --eval_beam_sizes [5] --is_end_to_end True --save_model_path ${DATA_DIR}/saves/ftse2echeckpoint_2023-03-01-08:27:20_epoch2it9440bs12_bleu_.pth
Saved to checkpoint_2023-03-01-02:02:30_epoch0it9440bs12_bleu_.pth
Saved to checkpoint_2023-03-01-05:17:24_epoch1it9440bs12_bleu_.pth
Saved to checkpoint_2023-03-01-08:27:20_epoch2it9440bs12_bleu_.pth
Saved to checkpoint_2023-03-01-11:44:54_epoch3it9440bs12_bleu_.pth


[('CIDEr', 1.3638), ('Bleu_1', 0.8442), ('Bleu_2', 0.6893), ('Bleu_3', 0.5349), ('Bleu_4', 0.4063), ('ROUGE_L', 0.6026), ('SPICE', 0.2399), ('METEOR', 0.2974)]
[('CIDEr', 1.3555), ('Bleu_1', 0.847), ('Bleu_2', 0.6888), ('Bleu_3', 0.533), ('Bleu_4', 0.4035), ('ROUGE_L', 0.6003), ('SPICE', 0.2399), ('METEOR', 0.2958)]
[('CIDEr', 1.3503), ('Bleu_1', 0.8556), ('Bleu_2', 0.6928), ('Bleu_3', 0.5325), ('Bleu_4', 0.4007), ('ROUGE_L', 0.5997), ('SPICE', 0.2327), ('METEOR', 0.2928)]
[('CIDEr', 1.3389), ('Bleu_1', 0.8572), ('Bleu_2', 0.6917), ('Bleu_3', 0.5305), ('Bleu_4', 0.3982), ('ROUGE_L', 0.5972), ('SPICE', 0.2315), ('METEOR', 0.2913)]

## bleu4

python train.py --N_enc 3 --N_dec 3  \
    --model_dim 512 --optim_type radam --seed 775533 --sched_type custom_warmup_anneal  \
    --warmup 1 --anneal_coeff 1.0 --lr 5e-5 --enc_drop 0.1 \
    --dec_drop 0.1 --enc_input_drop 0.1 --dec_input_drop 0.1 --drop_other 0.1  \
    --batch_size 18 --num_accum 2 --num_gpus 1 --ddp_sync_port 11318 --eval_beam_sizes [5]  \
    --save_every_minutes 60 --how_many_checkpoints 1  \
    --is_end_to_end True --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/ --partial_load True \
    --body_save_path ${DATA_DIR}/saves/rf_model-002.pth \
    --print_every_iter 1500 --eval_every_iter 999999 --num_epochs 5 \
    --save_path ${DATA_DIR}/saves/e2ebleu4lr5e-5/ --reinforce bleu4


# rouge
python train.py --N_enc 3 --N_dec 3  \
    --model_dim 512 --optim_type radam --seed 775533 --sched_type custom_warmup_anneal  \
    --warmup 1 --anneal_coeff 1.0 --lr 1e-5 --enc_drop 0.1 \
    --dec_drop 0.1 --enc_input_drop 0.1 --dec_input_drop 0.1 --drop_other 0.1  \
    --batch_size 12 --num_accum 2 --num_gpus 1 --ddp_sync_port 11317 --eval_beam_sizes [5]  \
    --save_every_minutes 60 --how_many_checkpoints 1  \
    --is_end_to_end True --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/ --partial_load True \
    --body_save_path ${DATA_DIR}/saves/rf_model-002.pth \
    --print_every_iter 1500 --eval_every_iter 999999 --num_epochs 5 \
    --save_path ${DATA_DIR}/saves/e2erougebs12/ --reinforce rouge


python train.py --optim_type radam --seed 775533 --sched_type custom_warmup_anneal  \
    --warmup 1 --anneal_coeff 1.0 --lr 5e-6 --batch_size 18 --num_accum 2\
    --is_end_to_end True --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/\
    --body_save_path ${DATA_DIR}/saves/rf_model-002.pth --num_epochs 5\
    --save_path ${DATA_DIR}/saves/e2erougebs18lr5e-6/ --reinforce rouge
