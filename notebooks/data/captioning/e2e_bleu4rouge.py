def todict(l):
    return {k: v for k, v in l}


# python train.py --optim_type radam --seed 775533 --sched_type custom_warmup_anneal --warmup 1 --anneal_coeff 1.0 --lr 5e-6 --batch_size 18 --num_accum 2 --is_end_to_end True --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/ --body_save_path ${DATA_DIR}/saves/rf_model-002.pth --num_epochs 5 --save_path ${DATA_DIR}/saves/e2ebleu4rougebs18lr5e-6/ --reinforce bleu4,rouge

l_step = []

l_step.append(
    [
        ('CIDEr', 1.3965), ('Bleu_1', 0.8321), ('Bleu_2', 0.6864), ('Bleu_3', 0.5408),
        ('Bleu_4', 0.4168), ('ROUGE_L', 0.6059), ('SPICE', 0.2428), ('METEOR', 0.3036),
        ("lambda", 1.0)
    ]
)
l_step.append([
    ('CIDEr', 1.3824), ('Bleu_1', 0.8273), ('Bleu_2', 0.6859), ('Bleu_3', 0.5446),
    ('Bleu_4', 0.4228), ('ROUGE_L', 0.6086), ('SPICE', 0.2406), ('METEOR', 0.3016), ("lambda", 0.8)
])
l_step.append([
    ('CIDEr', 1.3727), ('Bleu_1', 0.8265), ('Bleu_2', 0.6847), ('Bleu_3', 0.5427),
    ('Bleu_4', 0.4209), ('ROUGE_L', 0.6093), ('SPICE', 0.2394), ('METEOR', 0.3016), ("lambda", 0.6)
])
l_step.append([
    ('CIDEr', 1.3715), ('Bleu_1', 0.8273), ('Bleu_2', 0.6846), ('Bleu_3', 0.5429),
    ('Bleu_4', 0.4219), ('ROUGE_L', 0.6076), ('SPICE', 0.2399), ('METEOR', 0.3015), ("lambda", 0.4)
])
l_step.append([
    ('CIDEr', 1.3645), ('Bleu_1', 0.823), ('Bleu_2', 0.6823), ('Bleu_3', 0.5417),
    ('Bleu_4', 0.4206), ('ROUGE_L', 0.6069), ('SPICE', 0.2392), ('METEOR', 0.3011), ("lambda", 0.2)
])
l_step.append([
    ('CIDEr', 1.3543), ('Bleu_1', 0.8225), ('Bleu_2', 0.68), ('Bleu_3', 0.5386), ('Bleu_4', 0.4178),
    ('ROUGE_L', 0.6062), ('SPICE', 0.2387), ('METEOR', 0.3), ("lambda", 0.0)
])
# Saved to model_bleu4rouge_bs18lr5e6_epoch4.pth
l_step = [todict(l) for l in l_step]

l_wa_rougebleu4 = []

l_wa_rougebleu4.append(
    [
        ('CIDEr', 1.3412), ('Bleu_1', 0.8188), ('Bleu_2', 0.6789), ('Bleu_3', 0.5398),
        ('Bleu_4', 0.4194), ('ROUGE_L', 0.6047), ('SPICE', 0.2366), ('METEOR', 0.2988),
        ("lambda", 1.)
    ]
)

l_wa_rougebleu4.append(
    [
        ('CIDEr', 1.3485), ('Bleu_1', 0.8216), ('Bleu_2', 0.6806), ('Bleu_3', 0.5408),
        ('Bleu_4', 0.42), ('ROUGE_L', 0.6051), ('SPICE', 0.2373), ('METEOR', 0.2992),
        ("lambda", 0.9)
    ]
)
l_wa_rougebleu4.append(
    [
        ('CIDEr', 1.3588), ('Bleu_1', 0.823), ('Bleu_2', 0.6834), ('Bleu_3', 0.5444),
        ('Bleu_4', 0.424), ('ROUGE_L', 0.6079), ('SPICE', 0.2387), ('METEOR', 0.3011),
        ("lambda", 0.75)
    ]
)
l_wa_rougebleu4.append(
    [
        ('CIDEr', 1.3615), ('Bleu_1', 0.8237), ('Bleu_2', 0.6834), ('Bleu_3', 0.5436),
        ('Bleu_4', 0.423), ('ROUGE_L', 0.6092), ('SPICE', 0.2384), ('METEOR', 0.3011),
        ("lambda", 0.6)
    ]
)
l_wa_rougebleu4.append(
    [
        ('CIDEr', 1.362), ('Bleu_1', 0.8239), ('Bleu_2', 0.6841), ('Bleu_3', 0.5444),
        ('Bleu_4', 0.424), ('ROUGE_L', 0.61), ('SPICE', 0.2386), ('METEOR', 0.3013),
        ("lambda", 0.5)
    ]
)
l_wa_rougebleu4.append(
    [
        ('CIDEr', 1.3593), ('Bleu_1', 0.823), ('Bleu_2', 0.6828), ('Bleu_3', 0.5424),
        ('Bleu_4', 0.4218), ('ROUGE_L', 0.6098), ('SPICE', 0.2386), ('METEOR', 0.3006),
        ("lambda", 0.4)
    ]
)
l_wa_rougebleu4.append(
    [
        ('CIDEr', 1.3572), ('Bleu_1', 0.8213), ('Bleu_2', 0.6806), ('Bleu_3', 0.5398),
        ('Bleu_4', 0.419), ('ROUGE_L', 0.6093), ('SPICE', 0.2384), ('METEOR', 0.2999),
        ("lambda", 0.25)
    ]
)
l_wa_rougebleu4.append(
    [
        ('CIDEr', 1.3504), ('Bleu_1', 0.8183), ('Bleu_2', 0.677), ('Bleu_3', 0.5353),
        ('Bleu_4', 0.4152), ('ROUGE_L', 0.6074), ('SPICE', 0.2373), ('METEOR', 0.2984),
        ("lambda", 0.1)
    ]
)
l_wa_rougebleu4.append(
    [
        ('CIDEr', 1.3481), ('Bleu_1', 0.8179), ('Bleu_2', 0.6766), ('Bleu_3', 0.5349),
        ('Bleu_4', 0.415), ('ROUGE_L', 0.6075), ('SPICE', 0.2369), ('METEOR', 0.2982),
        ("lambda", 0.)
    ]
)

l_wa_rougebleu4 = [todict(l) for l in l_wa_rougebleu4]
