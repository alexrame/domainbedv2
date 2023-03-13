def todict(l):
    return {k: v for k, v in l}


# python train.py --optim_type radam --seed 775533 --sched_type custom_warmup_anneal --warmup 1 --anneal_coeff 1.0 --lr 5e-6 --batch_size 18 --num_accum 2 --is_end_to_end True --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/ --body_save_path ${DATA_DIR}/saves/rf_model-002.pth --num_epochs 5 --save_path ${DATA_DIR}/saves/e2ebleubleu4bs18lr5e-6/ --reinforce bleu,bleu4

l_step = []

l_step.append(
    [
        ('CIDEr', 1.3965), ('Bleu_1', 0.8321), ('Bleu_2', 0.6864), ('Bleu_3', 0.5408),
        ('Bleu_4', 0.4168), ('ROUGE_L', 0.6059), ('SPICE', 0.2428), ('METEOR', 0.3036),
        ("lambda", 1.0)
    ]
)
l_step.append([
    ('CIDEr', 1.3794), ('Bleu_1', 0.8333), ('Bleu_2', 0.6863), ('Bleu_3', 0.541),
    ('Bleu_4', 0.4169), ('ROUGE_L', 0.6057), ('SPICE', 0.2421), ('METEOR', 0.3018),
    ("lambda", 0.8)
])
l_step.append([
    ('CIDEr', 1.3788), ('Bleu_1', 0.8346), ('Bleu_2', 0.6891), ('Bleu_3', 0.5436),
    ('Bleu_4', 0.4198), ('ROUGE_L', 0.6076), ('SPICE', 0.2417), ('METEOR', 0.3019),
    ("lambda", 0.6)
])
l_step.append([
    ('CIDEr', 1.3807), ('Bleu_1', 0.8359), ('Bleu_2', 0.6907), ('Bleu_3', 0.546),
    ('Bleu_4', 0.4223), ('ROUGE_L', 0.6084), ('SPICE', 0.2416), ('METEOR', 0.302),
    ("lambda", 0.4)
    ])
l_step.append([
    ('CIDEr', 1.3769), ('Bleu_1', 0.8347), ('Bleu_2', 0.6885), ('Bleu_3', 0.5441),
    ('Bleu_4', 0.4211), ('ROUGE_L', 0.6083), ('SPICE', 0.2422), ('METEOR', 0.3019),
    ("lambda", 0.2)
])
l_step.append([
    ('CIDEr', 1.3763), ('Bleu_1', 0.8348), ('Bleu_2', 0.6889), ('Bleu_3', 0.5451),
    ('Bleu_4', 0.4225), ('ROUGE_L', 0.6086), ('SPICE', 0.2422), ('METEOR', 0.3024),
    ("lambda", 0.0)
])

l_step = [todict(l) for l in l_step]

l_wa_bleu4bleu = []

l_wa_bleu4bleu.append(
    [
        ('CIDEr', 1.3389), ('Bleu_1', 0.8572), ('Bleu_2', 0.6917), ('Bleu_3', 0.5305),
        ('Bleu_4', 0.3982), ('ROUGE_L', 0.5972), ('SPICE', 0.2315), ('METEOR', 0.2913),
        ("lambda", 1.)
    ]
)

l_wa_bleu4bleu.append(
    [
        ('CIDEr', 1.3521), ('Bleu_1', 0.8553), ('Bleu_2', 0.694), ('Bleu_3', 0.5355),
        ('Bleu_4', 0.404), ('ROUGE_L', 0.5995), ('SPICE', 0.2353), ('METEOR', 0.2939),
        ("lambda", 0.9)
    ]
)

l_wa_bleu4bleu.append(
    [
        ('CIDEr', 1.3638), ('Bleu_1', 0.853), ('Bleu_2', 0.696), ('Bleu_3', 0.5405),
        ('Bleu_4', 0.4106), ('ROUGE_L', 0.6024), ('SPICE', 0.2383), ('METEOR', 0.2968),
        ("lambda", 0.8)
    ]
)
l_wa_bleu4bleu.append(
    [
        ('CIDEr', 1.3741), ('Bleu_1', 0.8445), ('Bleu_2', 0.6947), ('Bleu_3', 0.5451),
        ('Bleu_4', 0.4175), ('ROUGE_L', 0.6046), ('SPICE', 0.2417), ('METEOR', 0.3002),
        ("lambda", 0.6)
    ]
)

l_wa_bleu4bleu.append(
    [
        ('CIDEr', 1.3741), ('Bleu_1', 0.8407), ('Bleu_2', 0.6933), ('Bleu_3', 0.5461),
        ('Bleu_4', 0.4195), ('ROUGE_L', 0.606), ('SPICE', 0.2413), ('METEOR', 0.3009),
        ("lambda", 0.5)
    ]
)
l_wa_bleu4bleu.append(
    [
        ('CIDEr', 1.3791), ('Bleu_1', 0.8381), ('Bleu_2', 0.6935), ('Bleu_3', 0.5485),
        ('Bleu_4', 0.4237), ('ROUGE_L', 0.6081), ('SPICE', 0.2418), ('METEOR', 0.302),
        ("lambda", 0.4)
    ]
)
l_wa_bleu4bleu.append(
    [
        ('CIDEr', 1.3751), ('Bleu_1', 0.835), ('Bleu_2', 0.6922), ('Bleu_3', 0.549),
        ('Bleu_4', 0.4252), ('ROUGE_L', 0.6089), ('SPICE', 0.2408), ('METEOR', 0.3024),
        ("lambda", 0.3)
    ]
)

l_wa_bleu4bleu.append(
    [
        ('CIDEr', 1.3672), ('Bleu_1', 0.8301), ('Bleu_2', 0.6889), ('Bleu_3', 0.5475),
        ('Bleu_4', 0.4247), ('ROUGE_L', 0.6084), ('SPICE', 0.2395), ('METEOR', 0.3017),
        ("lambda", 0.2)
    ]
)

l_wa_bleu4bleu.append(
    [
        ('CIDEr', 1.3545), ('Bleu_1', 0.8247), ('Bleu_2', 0.6835), ('Bleu_3', 0.543),
        ('Bleu_4', 0.4218), ('ROUGE_L', 0.6057), ('SPICE', 0.2379), ('METEOR', 0.3),
        ("lambda", 0.1)
    ]
)

l_wa_bleu4bleu.append(
    [
        ('CIDEr', 1.3412), ('Bleu_1', 0.8188), ('Bleu_2', 0.6789), ('Bleu_3', 0.5398),
        ('Bleu_4', 0.4194), ('ROUGE_L', 0.6047), ('SPICE', 0.2366), ('METEOR', 0.2988),
        ("lambda", 0.)
    ]
)

l_wa_bleu4bleu = [todict(l) for l in l_wa_bleu4bleu]
