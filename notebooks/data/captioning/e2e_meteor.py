# python train.py --optim_type radam --seed 775533 --sched_type custom_warmup_anneal --warmup 1 --anneal_coeff 1.0 --lr 5e-6 --batch_size 18 --num_accum 2 --is_end_to_end True --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/ --body_save_path ${DATA_DIR}/saves/rf_model-002.pth --num_epochs 5 --save_path ${DATA_DIR}/saves/e2emeteorbs18lr5e-6/ --reinforce meteor


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
        ('CIDEr', 1.3527), ('Bleu_1', 0.808), ('Bleu_2', 0.6559), ('Bleu_3', 0.5111),
        ('Bleu_4', 0.3891), ('ROUGE_L', 0.602), ('SPICE', 0.2478), ('METEOR', 0.3073),
        ("lambda", 0.8)
    ]
)
l_step.append(
    [
        ('CIDEr', 1.3412), ('Bleu_1', 0.8015), ('Bleu_2', 0.649), ('Bleu_3', 0.5051),
        ('Bleu_4', 0.3838), ('ROUGE_L', 0.6013), ('SPICE', 0.25), ('METEOR', 0.3094),
        ("lambda", 0.6)
    ]
)

l_step.append(
    [
        ('CIDEr', 1.3219), ('Bleu_1', 0.7946), ('Bleu_2', 0.6408), ('Bleu_3', 0.4969),
        ('Bleu_4', 0.3765), ('ROUGE_L', 0.6001), ('SPICE', 0.2511), ('METEOR', 0.3104),
        ("lambda", 0.4)
    ]
)

l_step.append(
    [
        ('CIDEr', 1.3149), ('Bleu_1', 0.7909), ('Bleu_2', 0.6372), ('Bleu_3', 0.4932),
        ('Bleu_4', 0.3728), ('ROUGE_L', 0.5992), ('SPICE', 0.2529), ('METEOR', 0.3109),
        ("lambda", 0.2)
    ]
)

l_step.append(
    [
        ('CIDEr', 1.3028), ('Bleu_1', 0.7913), ('Bleu_2', 0.6353), ('Bleu_3', 0.4904),
        ('Bleu_4', 0.3699), ('ROUGE_L', 0.5969), ('SPICE', 0.2511), ('METEOR', 0.3102),
        ("lambda", 0.0)
    ]
)
# Saved to checkpoint_2023-03-05-07:16:46_epoch4it6293bs18_meteor_.pth



l_step = [todict(l) for l in l_step]


# python train.py --optim_type radam --seed 775533 --sched_type custom_warmup_anneal --warmup 1 --anneal_coeff 1.0 --lr 1e-5 --batch_size 18 --num_accum 2 --is_end_to_end True --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/ --body_save_path ${DATA_DIR}/saves/rf_model-002.pth --num_epochs 5 --save_path ${DATA_DIR}/saves/e2emeteorbs18lr1e-5/ --reinforce meteor
# [('CIDEr', 1.3121), ('Bleu_1', 0.7887), ('Bleu_2', 0.6335), ('Bleu_3', 0.4897), ('Bleu_4', 0.371), ('ROUGE_L', 0.5958), ('SPICE', 0.2489), ('METEOR', 0.3073)]
# Saved to checkpoint_2023-03-08-13:43:57_epoch0it6293bs18_meteor_.pth
# [('CIDEr', 1.2996), ('Bleu_1', 0.7851), ('Bleu_2', 0.6287), ('Bleu_3', 0.4852), ('Bleu_4', 0.3661), ('ROUGE_L', 0.5962), ('SPICE', 0.2507), ('METEOR', 0.3099)]
# Saved to checkpoint_2023-03-08-17:15:39_epoch1it6293bs18_meteor_.pth
