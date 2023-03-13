def todict(l):
    return {k: v for k, v in l}


# python train.py --optim_type radam --seed 775533 --sched_type custom_warmup_anneal  \
#     --warmup 1 --anneal_coeff 0.8 --anneal_every_epoch 1 --lr 1e-5 --batch_size 18 \
#     --is_end_to_end False --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/ --num_accum 2\
#     --body_save_path ${DATA_DIR}/saves/rf_model-002.pth --num_epochs 9\
#     --partial_load True     --features_path ${DATA_DIR}/raw_data/features_rf.hdf5\
#     --save_path ${DATA_DIR}/saves/ftsbleubs18lr1e-5/ --reinforce bleu

l_step = []

l_step.append(
    [
        ('CIDEr', 1.3965), ('Bleu_1', 0.8321), ('Bleu_2', 0.6864), ('Bleu_3', 0.5408),
        ('Bleu_4', 0.4168), ('ROUGE_L', 0.6059), ('SPICE', 0.2428), ('METEOR', 0.3036),
        ("lambda", 1.0)
    ]
)

# python train.py --optim_type radam --seed 775533 --sched_type custom_warmup_anneal      --warmup 1 --anneal_coeff 0.8 --anneal_every_epoch 1 --lr 1e-5 --batch_size 18     --is_end_to_end False --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/ --num_accum 2     --num_epochs 9    --partial_load True     --features_path ${DATA_DIR}/raw_data/features_rf.hdf5    --save_path ${DATA_DIR}/saves/ftsbleubs18lr1e-5/ --reinforce bleu --body_save_path ${DATA_DIR}/saves/rf_model-002.pth

l_step.append(
    [
        ('CIDEr', 1.3771), ('Bleu_1', 0.8419), ('Bleu_2', 0.6892), ('Bleu_3', 0.5377),
        ('Bleu_4', 0.4103), ('ROUGE_L', 0.6042), ('SPICE', 0.2418), ('METEOR', 0.2997)
    ]
)
l_step.append(
    [
        ('CIDEr', 1.3763), ('Bleu_1', 0.8434), ('Bleu_2', 0.6899), ('Bleu_3', 0.5371),
        ('Bleu_4', 0.4088), ('ROUGE_L', 0.6026), ('SPICE', 0.2412), ('METEOR', 0.2993)
    ]
)
l_step.append(
    [
        ('CIDEr', 1.3764), ('Bleu_1', 0.8449), ('Bleu_2', 0.6908), ('Bleu_3', 0.5379),
        ('Bleu_4', 0.4095), ('ROUGE_L', 0.6037), ('SPICE', 0.2417), ('METEOR', 0.2998)
    ]
)
l_step.append(
    [
        ('CIDEr', 1.3785), ('Bleu_1', 0.8486), ('Bleu_2', 0.6937), ('Bleu_3', 0.5396),
        ('Bleu_4', 0.4108), ('ROUGE_L', 0.6042), ('SPICE', 0.2419), ('METEOR', 0.3001)
    ]
)
l_step.append(
    [
        ('CIDEr', 1.3725), ('Bleu_1', 0.8483), ('Bleu_2', 0.6925), ('Bleu_3', 0.5379),
        ('Bleu_4', 0.4092), ('ROUGE_L', 0.6038), ('SPICE', 0.2415), ('METEOR', 0.299)
    ]
)
l_step.append(
    [
        ('CIDEr', 1.3707), ('Bleu_1', 0.8497), ('Bleu_2', 0.6931), ('Bleu_3', 0.538),
        ('Bleu_4', 0.4086), ('ROUGE_L', 0.6038), ('SPICE', 0.2413), ('METEOR', 0.299)
    ]
)
# epoch5
l_step.append(
    [
        ('CIDEr', 1.3719), ('Bleu_1', 0.8513), ('Bleu_2', 0.6941), ('Bleu_3', 0.5386),
        ('Bleu_4', 0.4092), ('ROUGE_L', 0.6034), ('SPICE', 0.2405), ('METEOR', 0.2982)
    ]
)
# Saved to checkpoint_2023-03-08-11:44:31_epoch6it6293bs18_bleu_.pth

l_step.append([('CIDEr', 1.3737), ('Bleu_1', 0.8532), ('Bleu_2', 0.6957), ('Bleu_3', 0.5394), ('Bleu_4', 0.4094), ('ROUGE_L', 0.6033), ('SPICE', 0.2386), ('METEOR', 0.2973)])
l_step.append([('CIDEr', 1.374), ('Bleu_1', 0.8534), ('Bleu_2', 0.6951), ('Bleu_3', 0.5386), ('Bleu_4', 0.4085), ('ROUGE_L', 0.603), ('SPICE', 0.2378), ('METEOR', 0.2971)])
# Saved to checkpoint_2023-03-08-13:11:41_epoch7it6293bs18_bleu_.pth
# checkpoint_2023-03-08-14:45:00_epoch8it6293bs18_bleu_.pth
l_step.append({"Bleu_1": 0.8513, "Bleu_2": 0.6941, "Bleu_3": 0.5386, "Bleu_4": 0.4092, "CIDEr": 1.3719, "METEOR": 0.2982, "ROUGE_L": 0.6034, "SPICE": 0.2405, "epoch": 0, "reinforce": "bleu", "step": 44051})
l_step.append({"Bleu_1": 0.8532, "Bleu_2": 0.6957, "Bleu_3": 0.5394, "Bleu_4": 0.4094, "CIDEr": 1.3737, "METEOR": 0.2973, "ROUGE_L": 0.6033, "SPICE": 0.2386, "epoch": 0, "reinforce": "bleu", "step": 50344})
l_step.append({"Bleu_1": 0.8534, "Bleu_2": 0.6951, "Bleu_3": 0.5386, "Bleu_4": 0.4085, "CIDEr": 1.374, "METEOR": 0.2971, "ROUGE_L": 0.603, "SPICE": 0.2378, "epoch": 0, "reinforce": "bleu", "step": 56637})
l_step.append({"Bleu_1": 0.8534, "Bleu_2": 0.6949, "Bleu_3": 0.5377, "Bleu_4": 0.4077, "CIDEr": 1.3713, "METEOR": 0.2962, "ROUGE_L": 0.6022, "SPICE": 0.237, "epoch": 0, "reinforce": "bleu", "step": 62930})
# checkpoint_2023-03-08-17:27:15_epoch9it6293bs18_bleu_.pth


def todict(l):
    if isinstance(l, dict):
        d = l
    else:
        d = {k: v for k, v in l}
    return d


l_step = [todict(l) for l in l_step]
for i, l in enumerate(l_step):
    l['lambda'] = 1 - i / len(l_step)

l_wa = []
l_wa.append([('CIDEr', 1.3577), ('Bleu_1', 0.8561), ('Bleu_2', 0.6946), ('Bleu_3', 0.5352), ('Bleu_4', 0.4043), ('ROUGE_L', 0.6007), ('SPICE', 0.2326), ('METEOR', 0.2928), ('lambda', [1.2, -0.2])])
l_wa.append([('CIDEr', 1.3645), ('Bleu_1', 0.8545), ('Bleu_2', 0.6946), ('Bleu_3', 0.5364), ('Bleu_4', 0.406), ('ROUGE_L', 0.6016), ('SPICE', 0.235), ('METEOR', 0.2946), ('lambda', [1.1, -0.1])])
l_wa.append([('CIDEr', 1.378), ('Bleu_1', 0.853), ('Bleu_2', 0.6963), ('Bleu_3', 0.5404), ('Bleu_4', 0.4107), ('ROUGE_L', 0.6031), ('SPICE', 0.2393), ('METEOR', 0.2981), ('lambda', [0.9, 0.1])])
l_wa.append([('CIDEr', 1.3835), ('Bleu_1', 0.8518), ('Bleu_2', 0.6966), ('Bleu_3', 0.5424), ('Bleu_4', 0.4133), ('ROUGE_L', 0.6041), ('SPICE', 0.2414), ('METEOR', 0.2997), ('lambda', [0.8, 0.2])])
l_wa.append([('CIDEr', 1.3897), ('Bleu_1', 0.8489), ('Bleu_2', 0.696), ('Bleu_3', 0.5437), ('Bleu_4', 0.4155), ('ROUGE_L', 0.6049), ('SPICE', 0.2425), ('METEOR', 0.3011), ('lambda', [0.7, 0.3])])
l_wa.append([('CIDEr', 1.3914), ('Bleu_1', 0.8457), ('Bleu_2', 0.6941), ('Bleu_3', 0.5434), ('Bleu_4', 0.416), ('ROUGE_L', 0.6055), ('SPICE', 0.2427), ('METEOR', 0.3019), ('lambda', [0.6, 0.4])])
l_wa.append([('CIDEr', 1.3933), ('Bleu_1', 0.8426), ('Bleu_2', 0.6926), ('Bleu_3', 0.5428), ('Bleu_4', 0.4158), ('ROUGE_L', 0.6052), ('SPICE', 0.2431), ('METEOR', 0.3022), ('lambda', [0.5, 0.5])])
l_wa.append([('CIDEr', 1.3946), ('Bleu_1', 0.8408), ('Bleu_2', 0.692), ('Bleu_3', 0.5432), ('Bleu_4', 0.417), ('ROUGE_L', 0.606), ('SPICE', 0.2435), ('METEOR', 0.3028), ('lambda', [0.4, 0.6])])
l_wa.append([('CIDEr', 1.3943), ('Bleu_1', 0.8384), ('Bleu_2', 0.6904), ('Bleu_3', 0.5423), ('Bleu_4', 0.4168), ('ROUGE_L', 0.606), ('SPICE', 0.2435), ('METEOR', 0.303), ('lambda', [0.30000000000000004, 0.7])])
l_wa.append([('CIDEr', 1.3963), ('Bleu_1', 0.8364), ('Bleu_2', 0.6893), ('Bleu_3', 0.5422), ('Bleu_4', 0.4175), ('ROUGE_L', 0.6063), ('SPICE', 0.2435), ('METEOR', 0.3032), ('lambda', [0.19999999999999996, 0.8])])
l_wa.append([('CIDEr', 1.3969), ('Bleu_1', 0.8339), ('Bleu_2', 0.6876), ('Bleu_3', 0.5415), ('Bleu_4', 0.4173), ('ROUGE_L', 0.606), ('SPICE', 0.2435), ('METEOR', 0.3034), ('lambda', [0.09999999999999998, 0.9])])
l_wa.append([('CIDEr', 1.3965), ('Bleu_1', 0.8321), ('Bleu_2', 0.6864), ('Bleu_3', 0.5408), ('Bleu_4', 0.4168), ('ROUGE_L', 0.6059), ('SPICE', 0.2428), ('METEOR', 0.3036), ('lambda', [0.0, 1.0])])
l_wa.append([('CIDEr', 1.3953), ('Bleu_1', 0.8193), ('Bleu_2', 0.6752), ('Bleu_3', 0.5316), ('Bleu_4', 0.4092), ('ROUGE_L', 0.6051), ('SPICE', 0.2421), ('METEOR', 0.3028), ('lambda', [-0.10000000000000009, 1.1])])
l_wa.append([('CIDEr', 1.3918), ('Bleu_1', 0.8163), ('Bleu_2', 0.6727), ('Bleu_3', 0.5296), ('Bleu_4', 0.4075), ('ROUGE_L', 0.6048), ('SPICE', 0.2416), ('METEOR', 0.3029), ('lambda', [-0.19999999999999996, 1.2])])

l_wa_step5 = []
l_wa_step5.append([('CIDEr', 1.3604), ('Bleu_1', 0.851), ('Bleu_2', 0.6918), ('Bleu_3', 0.5349), ('Bleu_4', 0.4046), ('ROUGE_L', 0.6022), ('SPICE', 0.2409), ('METEOR', 0.2968), ('lambda', [1.2, -0.2])])
l_wa_step5.append([('CIDEr', 1.3662), ('Bleu_1', 0.8513), ('Bleu_2', 0.6933), ('Bleu_3', 0.5372), ('Bleu_4', 0.4072), ('ROUGE_L', 0.6032), ('SPICE', 0.2411), ('METEOR', 0.2982), ('lambda', [1.1, -0.1])])
l_wa_step5.append([('CIDEr', 1.3775), ('Bleu_1', 0.8486), ('Bleu_2', 0.6938), ('Bleu_3', 0.54), ('Bleu_4', 0.411), ('ROUGE_L', 0.6047), ('SPICE', 0.2414), ('METEOR', 0.3002), ('lambda', [0.9, 0.1])])
l_wa_step5.append([('CIDEr', 1.3819), ('Bleu_1', 0.8471), ('Bleu_2', 0.6934), ('Bleu_3', 0.5409), ('Bleu_4', 0.413), ('ROUGE_L', 0.6049), ('SPICE', 0.2422), ('METEOR', 0.3008), ('lambda', [0.8, 0.2])])
l_wa_step5.append([('CIDEr', 1.3881), ('Bleu_1', 0.8459), ('Bleu_2', 0.6937), ('Bleu_3', 0.5426), ('Bleu_4', 0.4154), ('ROUGE_L', 0.6056), ('SPICE', 0.2429), ('METEOR', 0.3018), ('lambda', [0.7, 0.3])])
l_wa_step5.append([('CIDEr', 1.3934), ('Bleu_1', 0.8452), ('Bleu_2', 0.6945), ('Bleu_3', 0.5444), ('Bleu_4', 0.4176), ('ROUGE_L', 0.6067), ('SPICE', 0.2436), ('METEOR', 0.3023), ('lambda', [0.6, 0.4])])
l_wa_step5.append([('CIDEr', 1.3946), ('Bleu_1', 0.8424), ('Bleu_2', 0.693), ('Bleu_3', 0.5437), ('Bleu_4', 0.4173), ('ROUGE_L', 0.6062), ('SPICE', 0.2437), ('METEOR', 0.3024), ('lambda', [0.5, 0.5])])
l_wa_step5.append([('CIDEr', 1.3957), ('Bleu_1', 0.8401), ('Bleu_2', 0.6919), ('Bleu_3', 0.5438), ('Bleu_4', 0.4183), ('ROUGE_L', 0.6061), ('SPICE', 0.2436), ('METEOR', 0.3027), ('lambda', [0.4, 0.6])])
l_wa_step5.append([('CIDEr', 1.3956), ('Bleu_1', 0.8382), ('Bleu_2', 0.6904), ('Bleu_3', 0.5429), ('Bleu_4', 0.4179), ('ROUGE_L', 0.606), ('SPICE', 0.2434), ('METEOR', 0.3028), ('lambda', [0.30000000000000004, 0.7])])
l_wa_step5.append([('CIDEr', 1.3971), ('Bleu_1', 0.8361), ('Bleu_2', 0.6891), ('Bleu_3', 0.5423), ('Bleu_4', 0.4177), ('ROUGE_L', 0.6065), ('SPICE', 0.2434), ('METEOR', 0.3032), ('lambda', [0.19999999999999996, 0.8])])

def todict(l):

    def transform(d):
        d["lambdas"] = d["lambda"]
        assert 0.9999 < d["lambda"][0] + d["lambda"][1] < 1.001
        d["lambda"] = d["lambda"][1]

    d = {k: v for k, v in l}
    transform(d)
    return d

l_wa = [todict(l) for l in l_wa]
l_wa = sorted(l_wa, key=lambda d: d["lambda"], reverse=False)

l_wa_step5 = [todict(l) for l in l_wa_step5]
l_wa_step5 = sorted(l_wa_step5, key=lambda d: d["lambda"], reverse=False)

