# python train.py --optim_type radam --seed 775533 --sched_type custom_warmup_anneal  \
#     --warmup 1 --anneal_coeff 0.8 --anneal_every_epoch 1 --lr 1e-5 --batch_size 18 \
#     --is_end_to_end False --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/ --num_accum 2\
#     --body_save_path ${DATA_DIR}/saves/rf_model-002.pth --num_epochs 9\
#     --partial_load True     --features_path ${DATA_DIR}/raw_data/features_rf.hdf5\
#     --save_path ${DATA_DIR}/saves/ftsbleu4bs18lr1e-5/ --reinforce bleu4


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

l_step.append([('CIDEr', 1.38), ('Bleu_1', 0.8272), ('Bleu_2', 0.6866), ('Bleu_3', 0.5454), ('Bleu_4', 0.4236), ('ROUGE_L', 0.6071), ('SPICE', 0.2406), ('METEOR', 0.3013)])
l_step.append({"Bleu_1": 0.8274, "Bleu_2": 0.6865, "Bleu_3": 0.5453, "Bleu_4": 0.4234, "CIDEr": 1.382, "METEOR": 0.3022, "ROUGE_L": 0.6076, "SPICE": 0.2407, "epoch": 0, "reinforce": "bleu4", "step": 12586})
l_step.append({"Bleu_1": 0.8262, "Bleu_2": 0.685, "Bleu_3": 0.5443, "Bleu_4": 0.4227, "CIDEr": 1.3766, "METEOR": 0.3019, "ROUGE_L": 0.6075, "SPICE": 0.2402, "epoch": 0, "reinforce": "bleu4", "step": 18879})
l_step.append({"Bleu_1": 0.8265, "Bleu_2": 0.6855, "Bleu_3": 0.545, "Bleu_4": 0.4239, "CIDEr": 1.3765, "METEOR": 0.3017, "ROUGE_L": 0.6071, "SPICE": 0.2409, "epoch": 0, "reinforce": "bleu4", "step": 25172})
l_step.append({"Bleu_1": 0.8263, "Bleu_2": 0.6851, "Bleu_3": 0.5446, "Bleu_4": 0.4233, "CIDEr": 1.3723, "METEOR": 0.302, "ROUGE_L": 0.6073, "SPICE": 0.2409, "epoch": 0, "reinforce": "bleu4", "step": 31465})
l_step.append({"Bleu_1": 0.8252, "Bleu_2": 0.6842, "Bleu_3": 0.5439, "Bleu_4": 0.4224, "CIDEr": 1.3703, "METEOR": 0.3017, "ROUGE_L": 0.6072, "SPICE": 0.24, "epoch": 0, "reinforce": "bleu4", "step": 37758})
l_step.append({"Bleu_1": 0.8263, "Bleu_2": 0.6856, "Bleu_3": 0.5456, "Bleu_4": 0.4244, "CIDEr": 1.3725, "METEOR": 0.3018, "ROUGE_L": 0.6079, "SPICE": 0.2403, "epoch": 0, "reinforce": "bleu4", "step": 44051})
l_step.append({"Bleu_1": 0.8254, "Bleu_2": 0.6847, "Bleu_3": 0.5447, "Bleu_4": 0.4237, "CIDEr": 1.371, "METEOR": 0.3013, "ROUGE_L": 0.6068, "SPICE": 0.24, "epoch": 0, "reinforce": "bleu4", "step": 50344})
l_step.append({"Bleu_1": 0.8243, "Bleu_2": 0.6838, "Bleu_3": 0.5442, "Bleu_4": 0.4235, "CIDEr": 1.3676, "METEOR": 0.3013, "ROUGE_L": 0.607, "SPICE": 0.2391, "epoch": 0, "reinforce": "bleu4", "step": 56637})

def todict(l):
    if isinstance(l, dict):
        d = l
    else:
        d = {k: v for k, v in l}
    return d


l_step = [todict(l) for l in l_step]
for i, l in enumerate(l_step):
    l['lambda'] = 1 - i / len(l_step)
