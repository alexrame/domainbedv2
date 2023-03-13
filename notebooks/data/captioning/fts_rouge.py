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

# python train.py --optim_type radam --seed 775533 --sched_type custom_warmup_anneal  \
#     --warmup 1 --anneal_coeff 0.8 --anneal_every_epoch 1 --lr 5e-5 --batch_size 18 \
#     --is_end_to_end False --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/ --num_accum 2\
#     --body_save_path ${DATA_DIR}/saves/rf_model-002.pth --num_epochs 9\
#     --partial_load True     --features_path ${DATA_DIR}/raw_data/features_rf.hdf5\
#     --save_path ${DATA_DIR}/saves/ftsrougebs18lr5e-5/ --reinforce rouge


# python train.py --optim_type radam --seed 775533 --sched_type custom_warmup_anneal      --warmup 1 --anneal_coeff 0.8 --anneal_every_epoch 1 --lr 1e-5 --batch_size 18     --is_end_to_end False --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/ --num_accum 2    --body_save_path ${DATA_DIR}/saves/rf_model-002.pth --num_epochs 9    --partial_load True     --features_path ${DATA_DIR}/raw_data/features_rf.hdf5    --save_path ${DATA_DIR}/saves/ftsrougebs18lr1e-5/ --reinforce rouge

# python train.py --optim_type radam --seed 775533 --sched_type custom_warmup_anneal      --warmup 1 --anneal_coeff 0.8 --anneal_every_epoch 1 --lr 1e-5 --batch_size 18     --is_end_to_end False --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/ --num_accum 2 --num_epochs 15    --partial_load True     --features_path ${DATA_DIR}/raw_data/features_rf.hdf5    --save_path ${DATA_DIR}/saves/ftsrougebs18lr1e-5/ --reinforce rouge

# ${DATA_DIR}/saves/ftsrougebs18lr1e-5/
l_step.append([('CIDEr', 1.3792), ('Bleu_1', 0.8266), ('Bleu_2', 0.6838), ('Bleu_3', 0.5403), ('Bleu_4', 0.4178), ('ROUGE_L', 0.6084), ('SPICE', 0.2402), ('METEOR', 0.3016)])
l_step.append([('CIDEr', 1.3737), ('Bleu_1', 0.8246), ('Bleu_2', 0.6835), ('Bleu_3', 0.5414), ('Bleu_4', 0.4197), ('ROUGE_L', 0.6088), ('SPICE', 0.2393), ('METEOR', 0.301)])
l_step.append([('CIDEr', 1.3752), ('Bleu_1', 0.8263), ('Bleu_2', 0.6844), ('Bleu_3', 0.5415), ('Bleu_4', 0.4194), ('ROUGE_L', 0.6096), ('SPICE', 0.2398), ('METEOR', 0.3017)])
l_step.append([('CIDEr', 1.3724), ('Bleu_1', 0.8256), ('Bleu_2', 0.6837), ('Bleu_3', 0.5415), ('Bleu_4', 0.4195), ('ROUGE_L', 0.6101), ('SPICE', 0.2392), ('METEOR', 0.3014)])
l_step.append([('CIDEr', 1.3726), ('Bleu_1', 0.8249), ('Bleu_2', 0.6836), ('Bleu_3', 0.541), ('Bleu_4', 0.4192), ('ROUGE_L', 0.6107), ('SPICE', 0.2395), ('METEOR', 0.3014)])
# Saved to checkpoint_2023-03-07-17:42:17_epoch4it6293bs18_rouge_.pth
l_step.append([('CIDEr', 1.3701), ('Bleu_1', 0.8234), ('Bleu_2', 0.682), ('Bleu_3', 0.54), ('Bleu_4', 0.4187), ('ROUGE_L', 0.61), ('SPICE', 0.2388), ('METEOR', 0.3011)])
# Saved to checkpoint_2023-03-07-21:07:31_epoch5it6293bs18_rouge_.pt

l_step.append({"Bleu_1": 0.8258, "Bleu_2": 0.6844, "Bleu_3": 0.5421, "Bleu_4": 0.4205, "CIDEr": 1.3738, "METEOR": 0.3016, "ROUGE_L": 0.6104, "SPICE": 0.2397, "epoch": 0, "reinforce": "rouge", "step": 44051})
l_step.append({"Bleu_1": 0.8247, "Bleu_2": 0.6833, "Bleu_3": 0.5412, "Bleu_4": 0.4197, "CIDEr": 1.371, "METEOR": 0.3014, "ROUGE_L": 0.6106, "SPICE": 0.2397, "epoch": 0, "reinforce": "rouge", "step": 50344})
l_step.append({"Bleu_1": 0.8247, "Bleu_2": 0.6826, "Bleu_3": 0.5403, "Bleu_4": 0.4187, "CIDEr": 1.3692, "METEOR": 0.3013, "ROUGE_L": 0.6101, "SPICE": 0.2392, "epoch": 0, "reinforce": "rouge", "step": 56637})
l_step.append({"Bleu_1": 0.8244, "Bleu_2": 0.6818, "Bleu_3": 0.5387, "Bleu_4": 0.4168, "CIDEr": 1.3647, "METEOR": 0.3009, "ROUGE_L": 0.6089, "SPICE": 0.239, "epoch": 0, "reinforce": "rouge", "step": 62930})
l_step.append({"Bleu_1": 0.8235, "Bleu_2": 0.6816, "Bleu_3": 0.5394, "Bleu_4": 0.4182, "CIDEr": 1.3668, "METEOR": 0.3009, "ROUGE_L": 0.6094, "SPICE": 0.2392, "epoch": 0, "reinforce": "rouge", "step": 69223})
l_step.append({"Bleu_1": 0.824, "Bleu_2": 0.682, "Bleu_3": 0.5396, "Bleu_4": 0.4183, "CIDEr": 1.3669, "METEOR": 0.3008, "ROUGE_L": 0.6095, "SPICE": 0.2392, "epoch": 0, "reinforce": "rouge", "step": 75516})
l_step.append({"Bleu_1": 0.8246, "Bleu_2": 0.6823, "Bleu_3": 0.5395, "Bleu_4": 0.4179, "CIDEr": 1.3681, "METEOR": 0.3009, "ROUGE_L": 0.6097, "SPICE": 0.2392, "epoch": 0, "reinforce": "rouge", "step": 81809})
l_step.append({"Bleu_1": 0.8245, "Bleu_2": 0.6826, "Bleu_3": 0.5402, "Bleu_4": 0.4189, "CIDEr": 1.3689, "METEOR": 0.301, "ROUGE_L": 0.61, "SPICE": 0.2395, "epoch": 0, "reinforce": "rouge", "step": 88102})
l_step.append({"Bleu_1": 0.8242, "Bleu_2": 0.6826, "Bleu_3": 0.5403, "Bleu_4": 0.4188, "CIDEr": 1.3685, "METEOR": 0.301, "ROUGE_L": 0.6101, "SPICE": 0.2393, "epoch": 0, "reinforce": "rouge", "step": 94395})


def todict(l):
    if isinstance(l, dict):
        d = l
    else:
        d = {k: v for k, v in l}
    return d


l_step = [todict(l) for l in l_step]
for i, l in enumerate(l_step):
    l['lambda'] = 1 - i / len(l_step)


l_step_lr5 = []
# python train.py --optim_type radam --seed 775533 --sched_type custom_warmup_anneal      --warmup 1 --anneal_coeff 0.8 --anneal_every_epoch 1 --lr 5e-5 --batch_size 18     --is_end_to_end False --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/ --num_accum 2    --body_save_path ${DATA_DIR}/saves/rf_model-002.pth --num_epochs 9    --partial_load True     --features_path ${DATA_DIR}/raw_data/features_rf.hdf5    --save_path ${DATA_DIR}/saves/ftsrougebs18lr5e-5/ --reinforce rouge
l_step_lr5.append([('CIDEr', 1.345), ('Bleu_1', 0.819), ('Bleu_2', 0.6744), ('Bleu_3', 0.5328), ('Bleu_4', 0.412), ('ROUGE_L', 0.6061), ('SPICE', 0.2371), ('METEOR', 0.2992)])
l_step_lr5.append([('CIDEr', 1.3401), ('Bleu_1', 0.8167), ('Bleu_2', 0.6738), ('Bleu_3', 0.5325), ('Bleu_4', 0.4126), ('ROUGE_L', 0.6063), ('SPICE', 0.2355), ('METEOR', 0.2984)])
l_step_lr5.append([('CIDEr', 1.34), ('Bleu_1', 0.8162), ('Bleu_2', 0.6764), ('Bleu_3', 0.5353), ('Bleu_4', 0.4147), ('ROUGE_L', 0.608), ('SPICE', 0.2355), ('METEOR', 0.2978)])
# Saved to checkpoint_2023-03-07-14:39:45_epoch2it6293bs18_rouge_.pth

l_wa_step5 = []

l_wa_step5.append([('CIDEr', 1.3626), ('Bleu_1', 0.8211), ('Bleu_2', 0.6797), ('Bleu_3', 0.5381), ('Bleu_4', 0.417), ('ROUGE_L', 0.6091), ('SPICE', 0.238), ('METEOR', 0.2999), ('lambda', [1.2, -0.2])])
l_wa_step5.append([('CIDEr', 1.3658), ('Bleu_1', 0.8226), ('Bleu_2', 0.681), ('Bleu_3', 0.5392), ('Bleu_4', 0.4181), ('ROUGE_L', 0.6098), ('SPICE', 0.2386), ('METEOR', 0.3008), ('lambda', [1.1, -0.1])])
l_wa_step5.append([('CIDEr', 1.3734), ('Bleu_1', 0.8251), ('Bleu_2', 0.6835), ('Bleu_3', 0.5411), ('Bleu_4', 0.4194), ('ROUGE_L', 0.6101), ('SPICE', 0.2393), ('METEOR', 0.3015), ('lambda', [0.9, 0.1])])
l_wa_step5.append([('CIDEr', 1.3782), ('Bleu_1', 0.8262), ('Bleu_2', 0.6844), ('Bleu_3', 0.542), ('Bleu_4', 0.42), ('ROUGE_L', 0.6096), ('SPICE', 0.2403), ('METEOR', 0.3021), ('lambda', [0.8, 0.2])])
l_wa_step5.append([('CIDEr', 1.3797), ('Bleu_1', 0.8281), ('Bleu_2', 0.6856), ('Bleu_3', 0.5425), ('Bleu_4', 0.4202), ('ROUGE_L', 0.6093), ('SPICE', 0.2407), ('METEOR', 0.3023), ('lambda', [0.7, 0.3])])
l_wa_step5.append([('CIDEr', 1.3842), ('Bleu_1', 0.8296), ('Bleu_2', 0.6865), ('Bleu_3', 0.543), ('Bleu_4', 0.4202), ('ROUGE_L', 0.6093), ('SPICE', 0.2413), ('METEOR', 0.3028), ('lambda', [0.6, 0.4])])
l_wa_step5.append([('CIDEr', 1.3875), ('Bleu_1', 0.8305), ('Bleu_2', 0.687), ('Bleu_3', 0.5431), ('Bleu_4', 0.4201), ('ROUGE_L', 0.6084), ('SPICE', 0.242), ('METEOR', 0.3029), ('lambda', [0.5, 0.5])])
l_wa_step5.append([('CIDEr', 1.3898), ('Bleu_1', 0.8313), ('Bleu_2', 0.6872), ('Bleu_3', 0.5429), ('Bleu_4', 0.4196), ('ROUGE_L', 0.6077), ('SPICE', 0.2423), ('METEOR', 0.3028), ('lambda', [0.4, 0.6])])
l_wa_step5.append([('CIDEr', 1.3928), ('Bleu_1', 0.8323), ('Bleu_2', 0.6874), ('Bleu_3', 0.5426), ('Bleu_4', 0.4193), ('ROUGE_L', 0.6076), ('SPICE', 0.2427), ('METEOR', 0.3032), ('lambda', [0.30000000000000004, 0.7])])
l_wa_step5.append([('CIDEr', 1.394), ('Bleu_1', 0.8328), ('Bleu_2', 0.6876), ('Bleu_3', 0.5424), ('Bleu_4', 0.4187), ('ROUGE_L', 0.6074), ('SPICE', 0.2429), ('METEOR', 0.3034), ('lambda', [0.19999999999999996, 0.8])])
l_wa_step5.append([('CIDEr', 1.3959), ('Bleu_1', 0.8329), ('Bleu_2', 0.6874), ('Bleu_3', 0.5419), ('Bleu_4', 0.4179), ('ROUGE_L', 0.6065), ('SPICE', 0.2429), ('METEOR', 0.3035), ('lambda', [0.09999999999999998, 0.9])])
l_wa_step5.append([('CIDEr', 1.3973), ('Bleu_1', 0.8303), ('Bleu_2', 0.6843), ('Bleu_3', 0.5385), ('Bleu_4', 0.4147), ('ROUGE_L', 0.6046), ('SPICE', 0.2424), ('METEOR', 0.3033), ('lambda', [-0.10000000000000009, 1.1])])
l_wa_step5.append([('CIDEr', 1.3968), ('Bleu_1', 0.8278), ('Bleu_2', 0.6818), ('Bleu_3', 0.5361), ('Bleu_4', 0.4124), ('ROUGE_L', 0.604), ('SPICE', 0.2423), ('METEOR', 0.303), ('lambda', [-0.19999999999999996, 1.2])])


def todict(l):
    d = {k: v for k, v in l}
    d["lambdas"] = d["lambda"]
    assert 0.9999 < d["lambda"][0] + d["lambda"][1] < 1.001
    d["lambda"] = d["lambda"][1]
    return d


l_wa_step5 = [todict(l) for l in l_wa_step5]
l_wa_step5 = sorted(l_wa_step5, key=lambda d: d["lambda"], reverse=False)
