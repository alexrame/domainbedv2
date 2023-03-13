def todict(l):
    return {k: v for k, v in l}


l_step = []
l_step.append(
    [
        ('CIDEr', 1.3965), ('Bleu_1', 0.8321), ('Bleu_2', 0.6864), ('Bleu_3', 0.5408),
        ('Bleu_4', 0.4168), ('ROUGE_L', 0.6059), ('SPICE', 0.2428), ('METEOR', 0.3036),
        ("lambda", 1.0)
    ]
)
l_step.append(
    [
        ('CIDEr', 1.3591), ('Bleu_1', 0.8244), ('Bleu_2', 0.6813), ('Bleu_3', 0.5399),
        ('Bleu_4', 0.4183), ('ROUGE_L', 0.6052), ('SPICE', 0.2384), ('METEOR', 0.3),
        ("lambda", 0.8)
    ]
)
# Saved to checkpoint_2023-03-01-14:47:16_epoch0it6293bs18_bleu4_.pth

l_step.append(
    [
        ('CIDEr', 1.3538), ('Bleu_1', 0.822), ('Bleu_2', 0.681), ('Bleu_3', 0.5409),
        ('Bleu_4', 0.4201), ('ROUGE_L', 0.6055), ('SPICE', 0.2373), ('METEOR', 0.2992),
        ("lambda", 0.6)
    ]
)
# Saved to checkpoint_2023-03-01-18:08:16_epoch1it6293bs18_bleu4_.pth

l_step.append(
    [
        ('CIDEr', 1.3526), ('Bleu_1', 0.8226), ('Bleu_2', 0.6813), ('Bleu_3', 0.5412),
        ('Bleu_4', 0.4194), ('ROUGE_L', 0.6061), ('SPICE', 0.237), ('METEOR', 0.2994),
        ("lambda", 0.4)
    ]
)
# Saved to checkpoint_2023-03-01-21:20:30_epoch2it6293bs18_bleu4_.pth

l_step.append(
    [
        ('CIDEr', 1.3392), ('Bleu_1', 0.8181), ('Bleu_2', 0.6764), ('Bleu_3', 0.5358),
        ('Bleu_4', 0.4152), ('ROUGE_L', 0.6039), ('SPICE', 0.2368), ('METEOR', 0.299),
        ("lambda", 0.2)
    ]
)
# Saved to checkpoint_2023-03-02-00:34:42_epoch3it6293bs18_bleu4_.pth

l_step.append(
    [
        ('CIDEr', 1.3412), ('Bleu_1', 0.8188), ('Bleu_2', 0.6789), ('Bleu_3', 0.5398),
        ('Bleu_4', 0.4194), ('ROUGE_L', 0.6047), ('SPICE', 0.2366), ('METEOR', 0.2988),
        ("lambda", 0)
    ]
)
# Saved to checkpoint_2023-03-02-04:03:51_epoch4it6293bs18_bleu4_.pth

l_step = [todict(l) for l in l_step]

# python train.py --optim_type radam --seed 775533 --sched_type custom_warmup_anneal --warmup 1 --anneal_coeff 1.0 --lr 5e-6 --batch_size 18 --num_accum 2 --is_end_to_end True --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/ --body_save_path ${DATA_DIR}/saves/rf_model-002.pth --num_epochs 5 --save_path ${DATA_DIR}/saves/e2ebleu4bs18lr5e-6/ --reinforce bleu4

l_steplr = []

l_steplr.append(
    [
        ('CIDEr', 1.3965), ('Bleu_1', 0.8321), ('Bleu_2', 0.6864), ('Bleu_3', 0.5408),
        ('Bleu_4', 0.4168), ('ROUGE_L', 0.6059), ('SPICE', 0.2428), ('METEOR', 0.3036),
        ("lambda", 1.0)
    ]
)

l_steplr.append([('CIDEr', 1.3779), ('Bleu_1', 0.8263), ('Bleu_2', 0.6835), ('Bleu_3', 0.5417), ('Bleu_4', 0.4203), ('ROUGE_L', 0.6076), ('SPICE', 0.2399), ('METEOR', 0.3019), ("lambda", 0.5)])
l_steplr.append([('CIDEr', 1.3725), ('Bleu_1', 0.8247), ('Bleu_2', 0.6826), ('Bleu_3', 0.5415), ('Bleu_4', 0.4197), ('ROUGE_L', 0.6062), ('SPICE', 0.2398), ('METEOR', 0.3008), ("lambda", 0.)])

l_steplr = [todict(l) for l in l_steplr]

# python train.py --N_enc 3 --N_dec 3       --model_dim 512 --optim_type radam --seed 775533 --sched_type custom_warmup_anneal       --warmup 1 --anneal_coeff 1.0 --lr 2e-5 --enc_drop 0.1      --dec_drop 0.1 --enc_input_drop 0.1 --dec_input_drop 0.1 --drop_other 0.1       --batch_size 18 --num_accum 2 --num_gpus 1 --ddp_sync_port 11319 --eval_beam_sizes [5]       --save_every_minutes 60 --how_many_checkpoints 1       --is_end_to_end True --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/ --partial_load True      --body_save_path ${DATA_DIR}/saves/rf_model-002.pth      --print_every_iter 1500 --eval_every_iter 999999 --num_epochs 5      --save_path ${DATA_DIR}/saves/e2ebleu4lr2e-5/ --reinforce bleu4

# l_stepbs2e5 = []
# [('CIDEr', 1.3122), ('Bleu_1', 0.8108), ('Bleu_2', 0.6665), ('Bleu_3', 0.5271), ('Bleu_4', 0.4076), ('ROUGE_L', 0.5984), ('SPICE', 0.233), ('METEOR', 0.2957)]
# # Saved to checkpoint_2023-03-03-01:47:40_epoch0it6293bs18_bleu4_.pth
# [('CIDEr', 1.3139), ('Bleu_1', 0.8114), ('Bleu_2', 0.6691), ('Bleu_3', 0.5306), ('Bleu_4', 0.4106), ('ROUGE_L', 0.5986), ('SPICE', 0.234), ('METEOR', 0.2961)]
# # Saved to checkpoint_2023-03-03-05:18:22_epoch1it6293bs18_bleu4_.pth
# [('CIDEr', 1.3076), ('Bleu_1', 0.8101), ('Bleu_2', 0.6673), ('Bleu_3', 0.5284), ('Bleu_4', 0.4091), ('ROUGE_L', 0.5989), ('SPICE', 0.2328), ('METEOR', 0.2953)]


l_wa = []
l_wa.append(
    [
        ('CIDEr', 1.3965), ('Bleu_1', 0.8321), ('Bleu_2', 0.6864), ('Bleu_3', 0.5408),
        ('Bleu_4', 0.4168), ('ROUGE_L', 0.6059), ('SPICE', 0.2428), ('METEOR', 0.3036),
        ("lambda", 1.0)
    ]
)

l_wa.append(
    [
        ('CIDEr', 1.3944), ('Bleu_1', 0.8324), ('Bleu_2', 0.6873), ('Bleu_3', 0.5429),
        ('Bleu_4', 0.42), ('ROUGE_L', 0.6072), ('SPICE', 0.2426), ('METEOR', 0.3037),
        ("lambda", 0.8)
    ]
)
l_wa.append(
    [
        ('CIDEr', 1.3827), ('Bleu_1', 0.8289), ('Bleu_2', 0.6867), ('Bleu_3', 0.5446),
        ('Bleu_4', 0.4224), ('ROUGE_L', 0.6079), ('SPICE', 0.2409), ('METEOR', 0.303),
        ("lambda", 0.5)
    ]
)

l_wa.append(
    [
        ('CIDEr', 1.361), ('Bleu_1', 0.824), ('Bleu_2', 0.683), ('Bleu_3', 0.5432),
        ('Bleu_4', 0.4222), ('ROUGE_L', 0.6065), ('SPICE', 0.239), ('METEOR', 0.3011),
        ("lambda", 0.2)
    ]
)

l_wa.append(
    [
        ('CIDEr', 1.3412), ('Bleu_1', 0.8188), ('Bleu_2', 0.6789), ('Bleu_3', 0.5398),
        ('Bleu_4', 0.4194), ('ROUGE_L', 0.6047), ('SPICE', 0.2366), ('METEOR', 0.2988),
        ("lambda", 0)
    ]
)
l_wa = [todict(l) for l in l_wa]





