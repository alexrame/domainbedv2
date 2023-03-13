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
        ('CIDEr', 1.3611), ('Bleu_1', 0.8247), ('Bleu_2', 0.6806), ('Bleu_3', 0.5381),
        ('Bleu_4', 0.4161), ('ROUGE_L', 0.6075), ('SPICE', 0.2385), ('METEOR', 0.3),
        ("lambda", 0.8)
    ]
)

l_step.append(
    [
        ('CIDEr', 1.3552), ('Bleu_1', 0.8229), ('Bleu_2', 0.6795), ('Bleu_3', 0.537),
        ('Bleu_4', 0.4152), ('ROUGE_L', 0.6067), ('SPICE', 0.2395), ('METEOR', 0.3),
        ("lambda", 0.6)
    ]
)

l_step.append(
    [
        ('CIDEr', 1.3519), ('Bleu_1', 0.8214), ('Bleu_2', 0.6787), ('Bleu_3', 0.537),
        ('Bleu_4', 0.4168), ('ROUGE_L', 0.6085), ('SPICE', 0.2377), ('METEOR', 0.2993),
        ("lambda", 0.4)
    ]
)

l_step.append(
    [
        ('CIDEr', 1.3474), ('Bleu_1', 0.822), ('Bleu_2', 0.6785), ('Bleu_3', 0.5357),
        ('Bleu_4', 0.4141), ('ROUGE_L', 0.6072), ('SPICE', 0.2376), ('METEOR', 0.2994),
        ("lambda", 0.2)
    ]
)

l_step.append(
    [
        ('CIDEr', 1.3481), ('Bleu_1', 0.8179), ('Bleu_2', 0.6766), ('Bleu_3', 0.5349),
        ('Bleu_4', 0.415), ('ROUGE_L', 0.6075), ('SPICE', 0.2369), ('METEOR', 0.2982),
        ("lambda", 0)
    ]
)

l_step = [todict(l) for l in l_step]

# python train.py --N_enc 3 --N_dec 3  \
#     --model_dim 512 --optim_type radam --seed 775533 --sched_type custom_warmup_anneal  \
#     --warmup 1 --anneal_coeff 1.0 --lr 1e-5 --enc_drop 0.1 \
#     --dec_drop 0.1 --enc_input_drop 0.1 --dec_input_drop 0.1 --drop_other 0.1  \
#     --batch_size 12 --num_accum 2 --num_gpus 1 --ddp_sync_port 11317 --eval_beam_sizes [5]  \
#     --save_every_minutes 60 --how_many_checkpoints 1  \
#     --is_end_to_end True --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/ --partial_load True \
#     --body_save_path ${DATA_DIR}/saves/rf_model-002.pth \
#     --print_every_iter 1500 --eval_every_iter 999999 --num_epochs 5 \
#     --save_path ${DATA_DIR}/saves/e2erougebs12/ --reinforce rouge

l_stepbs12 = []

l_stepbs12.append(
    [
        ('CIDEr', 1.3965), ('Bleu_1', 0.8321), ('Bleu_2', 0.6864), ('Bleu_3', 0.5408),
        ('Bleu_4', 0.4168), ('ROUGE_L', 0.6059), ('SPICE', 0.2428), ('METEOR', 0.3036),
        ("lambda", 1.0)
    ]
)
l_stepbs12.append(
    [
        ('CIDEr', 1.3469), ('Bleu_1', 0.8187), ('Bleu_2', 0.675), ('Bleu_3', 0.5324),
        ('Bleu_4', 0.4106), ('ROUGE_L', 0.6056), ('SPICE', 0.2358), ('METEOR', 0.2986),
        ("lambda", 0.6)
    ]
)
l_stepbs12.append(
    [
        ('CIDEr', 1.3489), ('Bleu_1', 0.8201), ('Bleu_2', 0.6771), ('Bleu_3', 0.5341),
        ('Bleu_4', 0.4128), ('ROUGE_L', 0.6065), ('SPICE', 0.2369), ('METEOR', 0.2991),
        ("lambda", 0.4)
    ]
)
l_stepbs12.append(
    [
        ('CIDEr', 1.3422), ('Bleu_1', 0.8175), ('Bleu_2', 0.6757), ('Bleu_3', 0.5338),
        ('Bleu_4', 0.4125), ('ROUGE_L', 0.6085), ('SPICE', 0.2364), ('METEOR', 0.2984),
        ("lambda", 0.2)
    ]
)
l_stepbs12.append(
    [
        ('CIDEr', 1.3388), ('Bleu_1', 0.8152), ('Bleu_2', 0.6744), ('Bleu_3', 0.5344),
        ('Bleu_4', 0.4142), ('ROUGE_L', 0.6064), ('SPICE', 0.2332), ('METEOR', 0.296),
        ("lambda", 0.0)
    ]
)
# Saved to checkpoint_2023-03-03-05:12:07_epoch3it9440bs12_rouge_.pth

l_stepbs12 = [todict(l) for l in l_stepbs12]

# python train.py --optim_type radam --seed 775533 --sched_type custom_warmup_anneal  \
# >     --warmup 1 --anneal_coeff 1.0 --lr 5e-6 --batch_size 18 --num_accum 2\
# >     --is_end_to_end True --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/\
# >     --body_save_path ${DATA_DIR}/saves/rf_model-002.pth --num_epochs 5\
# >     --save_path ${DATA_DIR}/saves/e2erougebs18lr5e-6/ --reinforce rouge

l_steplr = []

l_steplr.append(
    [
        ('CIDEr', 1.3965), ('Bleu_1', 0.8321), ('Bleu_2', 0.6864), ('Bleu_3', 0.5408),
        ('Bleu_4', 0.4168), ('ROUGE_L', 0.6059), ('SPICE', 0.2428), ('METEOR', 0.3036),
        ("lambda", 1.0)
    ]
)

l_steplr.append([
    ('CIDEr', 1.3707), ('Bleu_1', 0.8258), ('Bleu_2', 0.6822), ('Bleu_3', 0.5391),
    ('Bleu_4', 0.4165), ('ROUGE_L', 0.6072), ('SPICE', 0.2393), ('METEOR', 0.3004), ("lambda", 0.8)
])

l_steplr.append([
    ('CIDEr', 1.3708), ('Bleu_1', 0.826), ('Bleu_2', 0.6834), ('Bleu_3', 0.5403), ('Bleu_4', 0.418),
    ('ROUGE_L', 0.6092), ('SPICE', 0.2384), ('METEOR', 0.3014), ("lambda", 0.6)
])
l_steplr.append([
    ('CIDEr', 1.3643), ('Bleu_1', 0.8237), ('Bleu_2', 0.6809), ('Bleu_3', 0.5382),
    ('Bleu_4', 0.4167), ('ROUGE_L', 0.6092), ('SPICE', 0.2383), ('METEOR', 0.3007), ("lambda", 0.4)
])
l_steplr.append([
    ('CIDEr', 1.3674), ('Bleu_1', 0.825), ('Bleu_2', 0.6828), ('Bleu_3', 0.5401),
    ('Bleu_4', 0.4184), ('ROUGE_L', 0.6095), ('SPICE', 0.2393), ('METEOR', 0.301), ("lambda", 0.2)
])
l_steplr.append([
    ('CIDEr', 1.3596), ('Bleu_1', 0.8211), ('Bleu_2', 0.68), ('Bleu_3', 0.5385), ('Bleu_4', 0.4175),
    ('ROUGE_L', 0.6087), ('SPICE', 0.237), ('METEOR', 0.2999), ("lambda", 0.0)
])
# Saved to checkpoint_2023-03-04-03:44:37_epoch4it6293bs18_rouge_.pth

l_steplr = [todict(l) for l in l_steplr]

# python test.py --N_enc 3 --N_dec 3 --model_dim 512 --num_gpus 1 --eval_beam_sizes [5] --is_end_to_end True --save_model_path /data/rame/ExpansionNet_v2/github_ignore_material/saves/e2erouge/rouge_model_epoch5.pth /data/rame/ExpansionNet_v2/github_ignore_material/saves/wa/bleu_model_epoch3.pth --ensemble wa_5

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
        ('CIDEr', 1.3532), ('Bleu_1', 0.8205), ('Bleu_2', 0.6781), ('Bleu_3', 0.5355),
        ('Bleu_4', 0.4149), ('ROUGE_L', 0.607), ('SPICE', 0.2379), ('METEOR', 0.2988),
        ("lambda", 0.1)
    ]
)

l_wa.append(
    [
        ('CIDEr', 1.365), ('Bleu_1', 0.8247), ('Bleu_2', 0.6831), ('Bleu_3', 0.5404),
        ('Bleu_4', 0.4184), ('ROUGE_L', 0.6089), ('SPICE', 0.2397), ('METEOR', 0.3006),
        ("lambda", 0.25)
    ]
)

l_wa.append(
    [
        ('CIDEr', 1.375), ('Bleu_1', 0.8282), ('Bleu_2', 0.686), ('Bleu_3', 0.5427),
        ('Bleu_4', 0.4198), ('ROUGE_L', 0.6089), ('SPICE', 0.2414), ('METEOR', 0.3018),
        ("lambda", 0.4)
    ]
)

l_wa.append(
    [
        ('CIDEr', 1.382), ('Bleu_1', 0.83), ('Bleu_2', 0.6872), ('Bleu_3', 0.5438),
        ('Bleu_4', 0.421), ('ROUGE_L', 0.6092), ('SPICE', 0.2421), ('METEOR', 0.3027),
        ("lambda", 0.5)
    ]
)

l_wa.append(
    [
        ('CIDEr', 1.3877), ('Bleu_1', 0.832), ('Bleu_2', 0.6882), ('Bleu_3', 0.544),
        ('Bleu_4', 0.4209), ('ROUGE_L', 0.6091), ('SPICE', 0.242), ('METEOR', 0.3029),
        ("lambda", 0.6)
    ]
)

l_wa.append(
    [
        ('CIDEr', 1.3891), ('Bleu_1', 0.8329), ('Bleu_2', 0.6876), ('Bleu_3', 0.5425),
        ('Bleu_4', 0.4186), ('ROUGE_L', 0.6078), ('SPICE', 0.2419), ('METEOR', 0.3031),
        ("lambda", 0.75)
    ]
)

l_wa.append(
    [
        ('CIDEr', 1.3927), ('Bleu_1', 0.8311), ('Bleu_2', 0.6848), ('Bleu_3', 0.5393),
        ('Bleu_4', 0.4158), ('ROUGE_L', 0.6062), ('SPICE', 0.2415), ('METEOR', 0.303),
        ("lambda", 0.9)
    ]
)

l_wa.append(
    [
        ('CIDEr', 1.3481), ('Bleu_1', 0.8179), ('Bleu_2', 0.6766), ('Bleu_3', 0.5349),
        ('Bleu_4', 0.415), ('ROUGE_L', 0.6075), ('SPICE', 0.2369), ('METEOR', 0.2982),
        ("lambda", 0)
    ]
)

l_wa = [todict(l) for l in l_wa]
l_wa = sorted(l_wa, key=lambda x: x["lambda"], reverse=True)


