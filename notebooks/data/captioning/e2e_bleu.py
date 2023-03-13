def todict(l):
    return {k: v for k, v in l}


# python train.py --N_enc 3 --N_dec 3 --model_dim 512 --optim_type radam --seed 775533 --sched_type custom_warmup_anneal --warmup 1 --anneal_coeff 1.0 --lr 1e-5 --enc_drop 0.1 --dec_drop 0.1 --enc_input_drop 0.1 --dec_input_drop 0.1 --drop_other 0.1 --batch_size 12 --num_accum 2 --num_gpus 1 --ddp_sync_port 11317 --eval_beam_sizes [5] --save_path ${DATA_DIR}/saves/ftse2e --save_every_minutes 60 --how_many_checkpoints 1 --is_end_to_end True --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/ --partial_load True --body_save_path ${DATA_DIR}/saves/rf_model-002.pth --print_every_iter 1500 --eval_every_iter 999999 --reinforce bleu --num_epochs 5

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
        ('CIDEr', 1.3638), ('Bleu_1', 0.8442), ('Bleu_2', 0.6893), ('Bleu_3', 0.5349),
        ('Bleu_4', 0.4063), ('ROUGE_L', 0.6026), ('SPICE', 0.2399), ('METEOR', 0.2974),
        ("lambda", 0.75)
    ]
)
l_step.append(
    [
        ('CIDEr', 1.3555), ('Bleu_1', 0.847), ('Bleu_2', 0.6888), ('Bleu_3', 0.533),
        ('Bleu_4', 0.4035), ('ROUGE_L', 0.6003), ('SPICE', 0.2399), ('METEOR', 0.2958),
        ("lambda", 0.5)
    ]
)
l_step.append(
    [
        ('CIDEr', 1.3503), ('Bleu_1', 0.8556), ('Bleu_2', 0.6928), ('Bleu_3', 0.5325),
        ('Bleu_4', 0.4007), ('ROUGE_L', 0.5997), ('SPICE', 0.2327), ('METEOR', 0.2928),
        ("lambda", 0.25)
    ]
)
l_step.append(
    [
        ('CIDEr', 1.3389), ('Bleu_1', 0.8572), ('Bleu_2', 0.6917), ('Bleu_3', 0.5305),
        ('Bleu_4', 0.3982), ('ROUGE_L', 0.5972), ('SPICE', 0.2315), ('METEOR', 0.2913),
        ("lambda", 0.)
    ]
)
l_step = [todict(l) for l in l_step]


# python train.py --optim_type radam --seed 775533 --sched_type custom_warmup_anneal  \
# >     --warmup 1 --anneal_coeff 1.0 --lr 5e-6 --batch_size 18 --num_accum 2\
# >     --is_end_to_end True --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/\
# >     --body_save_path ${DATA_DIR}/saves/rf_model-002.pth --num_epochs 5\
# >     --save_path ${DATA_DIR}/saves/e2ebleubs18lr5e-6/ --reinforce bleu

l_steplr = []
l_steplr.append(
    [
        ('CIDEr', 1.3965), ('Bleu_1', 0.8321), ('Bleu_2', 0.6864), ('Bleu_3', 0.5408),
        ('Bleu_4', 0.4168), ('ROUGE_L', 0.6059), ('SPICE', 0.2428), ('METEOR', 0.3036),
        ("lambda", 1.0)
    ]
)
l_steplr.append([('CIDEr', 1.3767), ('Bleu_1', 0.8413), ('Bleu_2', 0.688), ('Bleu_3', 0.5365), ('Bleu_4', 0.4087), ('ROUGE_L', 0.6038), ('SPICE', 0.2414), ('METEOR', 0.2993), ("lambda", 0.8)])
l_steplr.append([('CIDEr', 1.3795), ('Bleu_1', 0.8454), ('Bleu_2', 0.692), ('Bleu_3', 0.5409), ('Bleu_4', 0.4135), ('ROUGE_L', 0.6052), ('SPICE', 0.2431), ('METEOR', 0.3001), ("lambda", 0.6)])
l_steplr.append([('CIDEr', 1.3728), ('Bleu_1', 0.8458), ('Bleu_2', 0.6912), ('Bleu_3', 0.5387), ('Bleu_4', 0.4108), ('ROUGE_L', 0.6037), ('SPICE', 0.2433), ('METEOR', 0.299), ("lambda", 0.4)])
l_steplr.append([('CIDEr', 1.3647), ('Bleu_1', 0.8461), ('Bleu_2', 0.6901), ('Bleu_3', 0.5362), ('Bleu_4', 0.4078), ('ROUGE_L', 0.6025), ('SPICE', 0.2417), ('METEOR', 0.2974), ("lambda", 0.2)])
l_steplr.append([('CIDEr', 1.3656), ('Bleu_1', 0.8465), ('Bleu_2', 0.6904), ('Bleu_3', 0.537), ('Bleu_4', 0.4083), ('ROUGE_L', 0.6024), ('SPICE', 0.2421), ('METEOR', 0.2975), ("lambda", 0.0)])
# /data/rame/ExpansionNet_v2/github_ignore_material/saves/wa/model_bleu_bs12lr5e6_epoch4.pth

l_steplr = [todict(l) for l in l_steplr]

l_ens = []
l_ens.append(
    [
        ('CIDEr', 1.3965), ('Bleu_1', 0.8321), ('Bleu_2', 0.6864), ('Bleu_3', 0.5408),
        ('Bleu_4', 0.4168), ('ROUGE_L', 0.6059), ('SPICE', 0.2428), ('METEOR', 0.3036),
        ("lambda", 1.0)
    ]
)
l_ens.append(
    [
        ('CIDEr', 1.3936), ('Bleu_1', 0.8462), ('Bleu_2', 0.6963), ('Bleu_3', 0.5468),
        ('Bleu_4', 0.42), ('ROUGE_L', 0.6081), ('SPICE', 0.2421), ('METEOR', 0.3019),
        ("lambda", 0.5)
    ]
)

l_ens.append(
    [
        ('CIDEr', 1.3389), ('Bleu_1', 0.8572), ('Bleu_2', 0.6917), ('Bleu_3', 0.5305),
        ('Bleu_4', 0.3982), ('ROUGE_L', 0.5972), ('SPICE', 0.2315), ('METEOR', 0.2913),
        ("lambda", 0.)
    ]
)
l_ens = [todict(l) for l in l_ens]

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
        ('CIDEr', 1.392), ('Bleu_1', 0.8326), ('Bleu_2', 0.6856), ('Bleu_3', 0.5396),
        ('Bleu_4', 0.4159), ('ROUGE_L', 0.6056), ('SPICE', 0.2427), ('METEOR', 0.3032),
        ("lambda", 0.95)
    ]
)

l_wa.append(
    [
        ('CIDEr', 1.3933), ('Bleu_1', 0.8336), ('Bleu_2', 0.6864), ('Bleu_3', 0.5398),
        ('Bleu_4', 0.4155), ('ROUGE_L', 0.6059), ('SPICE', 0.2423), ('METEOR', 0.3031),
        ("lambda", 0.9)
    ]
)

l_wa.append(
    [
        ('CIDEr', 1.3979), ('Bleu_1', 0.8371), ('Bleu_2', 0.6896), ('Bleu_3', 0.542),
        ('Bleu_4', 0.4169), ('ROUGE_L', 0.6067), ('SPICE', 0.243), ('METEOR', 0.3034),
        ("lambda", 0.8)
    ]
)

# l_wa.append(
#     [
#         ('CIDEr', 0.0), ('Bleu_1', 0.0518), ('Bleu_2', 0.0231), ('Bleu_3', 0.0111),
#         ('Bleu_4', 0.0029), ('ROUGE_L', 0.0926), ('SPICE', 0.0187), ('METEOR', 0.0461),
#         ("lambda", 0.75)
#     ]
# )

l_wa.append(
    [
        ('CIDEr', 1.3939), ('Bleu_1', 0.8408), ('Bleu_2', 0.692), ('Bleu_3', 0.5434),
        ('Bleu_4', 0.4171), ('ROUGE_L', 0.6069), ('SPICE', 0.2434), ('METEOR', 0.3032),
        ("lambda", 0.7)
    ]
)

l_wa.append(
    [
        ('CIDEr', 1.3874), ('Bleu_1', 0.845), ('Bleu_2', 0.6939), ('Bleu_3', 0.5429),
        ('Bleu_4', 0.4149), ('ROUGE_L', 0.6055), ('SPICE', 0.2436), ('METEOR', 0.3015),
        ("lambda", 0.5)
    ]
)

l_wa.append(
    [
        ('CIDEr', 1.3831), ('Bleu_1', 0.8489), ('Bleu_2', 0.696), ('Bleu_3', 0.5432),
        ('Bleu_4', 0.414), ('ROUGE_L', 0.6052), ('SPICE', 0.2429), ('METEOR', 0.3002),
        ("lambda", 0.4)
    ]
)
l_wa.append(
    [
        ('CIDEr', 1.3756), ('Bleu_1', 0.8532), ('Bleu_2', 0.697), ('Bleu_3', 0.5421),
        ('Bleu_4', 0.4122), ('ROUGE_L', 0.604), ('SPICE', 0.2407),
        ('METEOR', 0.2987), ("lambda", 0.3)
    ]
)

l_wa.append(
    [
        ('CIDEr', 1.3656), ('Bleu_1', 0.8551), ('Bleu_2', 0.6961), ('Bleu_3', 0.5393),
        ('Bleu_4', 0.4085), ('ROUGE_L', 0.602), ('SPICE', 0.2384), ('METEOR', 0.2965),
        ("lambda", 0.2)
    ]
)

l_wa.append(
    [
        ('CIDEr', 1.3526), ('Bleu_1', 0.8553), ('Bleu_2', 0.6929), ('Bleu_3', 0.5336),
        ('Bleu_4', 0.4017), ('ROUGE_L', 0.599), ('SPICE', 0.2347), ('METEOR', 0.2937),
        ("lambda", 0.1)
    ]
)

l_wa.append(
    [
        ('CIDEr', 1.3444), ('Bleu_1', 0.8558), ('Bleu_2', 0.6918), ('Bleu_3', 0.5315),
        ('Bleu_4', 0.3998), ('ROUGE_L', 0.5968), ('SPICE', 0.2327), ('METEOR', 0.2923),
        ("lambda", 0.05)
    ]
)

l_wa.append(
    [
        ('CIDEr', 1.3389), ('Bleu_1', 0.8572), ('Bleu_2', 0.6917), ('Bleu_3', 0.5305),
        ('Bleu_4', 0.3982), ('ROUGE_L', 0.5972), ('SPICE', 0.2315), ('METEOR', 0.2913),
        ("lambda", 0.)
    ]
)

l_wa = [todict(l) for l in l_wa]


l_wableubleu = []
l_wableubleu.append(
    [
        ('CIDEr', 1.3389), ('Bleu_1', 0.8572), ('Bleu_2', 0.6917), ('Bleu_3', 0.5305),
        ('Bleu_4', 0.3982), ('ROUGE_L', 0.5972), ('SPICE', 0.2315), ('METEOR', 0.2913),
        ("lambda", 1.)
    ]
)

l_wableubleu.append([
    ('CIDEr', 1.3441), ('Bleu_1', 0.8568), ('Bleu_2', 0.6926), ('Bleu_3', 0.532),
    ('Bleu_4', 0.3993), ('ROUGE_L', 0.5976), ('SPICE', 0.2333), ('METEOR', 0.2919),
    ("lambda", 0.9)
])

l_wableubleu.append([
    ('CIDEr', 1.3524), ('Bleu_1', 0.8578), ('Bleu_2', 0.6946), ('Bleu_3', 0.5348),
    ('Bleu_4', 0.403), ('ROUGE_L', 0.6002), ('SPICE', 0.2358), ('METEOR', 0.2937),
    ("lambda", 0.8)
])

l_wableubleu.append([
    ('CIDEr', 1.3609), ('Bleu_1', 0.8589), ('Bleu_2', 0.6966), ('Bleu_3', 0.5371),
    ('Bleu_4', 0.4052), ('ROUGE_L', 0.6007), ('SPICE', 0.2374), ('METEOR', 0.2947),
    ("lambda", 0.7)
])

l_wableubleu.append([
    ('CIDEr', 1.3665), ('Bleu_1', 0.8572), ('Bleu_2', 0.6966), ('Bleu_3', 0.5387),
    ('Bleu_4', 0.4074), ('ROUGE_L', 0.6017), ('SPICE', 0.2392), ('METEOR', 0.2957),
    ("lambda", 0.6)
])

l_wableubleu.append([
    ('CIDEr', 1.3658), ('Bleu_1', 0.8566), ('Bleu_2', 0.6967), ('Bleu_3', 0.5388),
    ('Bleu_4', 0.4079), ('ROUGE_L', 0.6023), ('SPICE', 0.24), ('METEOR', 0.2963),
    ("lambda", 0.5)
])

l_wableubleu.append([
    ('CIDEr', 1.3646), ('Bleu_1', 0.8548), ('Bleu_2', 0.6952), ('Bleu_3', 0.538),
    ('Bleu_4', 0.4073), ('ROUGE_L', 0.6027), ('SPICE', 0.2404), ('METEOR', 0.2972),
    ("lambda", 0.4)
])

l_wableubleu.append([
    ('CIDEr', 1.3623), ('Bleu_1', 0.8529), ('Bleu_2', 0.694), ('Bleu_3', 0.5371),
    ('Bleu_4', 0.4065), ('ROUGE_L', 0.602), ('SPICE', 0.2406), ('METEOR', 0.297),
    ("lambda", 0.3)
])

l_wableubleu.append([
    ('CIDEr', 1.3598), ('Bleu_1', 0.8519), ('Bleu_2', 0.693), ('Bleu_3', 0.5363),
    ('Bleu_4', 0.4059), ('ROUGE_L', 0.6014), ('SPICE', 0.2404), ('METEOR', 0.2969),
    ("lambda", 0.25)
])

l_wableubleu.append([
    ('CIDEr', 1.3555), ('Bleu_1', 0.8484), ('Bleu_2', 0.6897), ('Bleu_3', 0.5337),
    ('Bleu_4', 0.4039), ('ROUGE_L', 0.6003), ('SPICE', 0.2396), ('METEOR', 0.296),
    ("lambda", 0.1)
])

l_wableubleu.append([
    ('CIDEr', 1.3555), ('Bleu_1', 0.847), ('Bleu_2', 0.6888), ('Bleu_3', 0.533), ('Bleu_4', 0.4035),
    ('ROUGE_L', 0.6003), ('SPICE', 0.2399), ('METEOR', 0.2958),
    ("lambda", 0.)
])
l_wableubleu = [todict(l) for l in l_wableubleu]
