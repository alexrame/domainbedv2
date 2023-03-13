# python train.py --optim_type radam --seed 775533 --sched_type custom_warmup_anneal  \
#     --warmup 1 --anneal_coeff 0.8 --anneal_every_epoch 1 --lr 1e-5 --batch_size 18 \
#     --is_end_to_end False --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/ --num_accum 2\
#     --body_save_path ${DATA_DIR}/saves/rf_model-002.pth --num_epochs 9\
#     --partial_load True     --features_path ${DATA_DIR}/raw_data/features_rf.hdf5\
#     --save_path ${DATA_DIR}/saves/ftsmeteorbs18lr1e-5v2/ --reinforce meteor
# python train.py --optim_type radam --seed 775533 --sched_type custom_warmup_anneal      --warmup 1 --anneal_coeff 0.8 --anneal_every_epoch 1 --lr 1e-5 --batch_size 18     --is_end_to_end False --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/ --num_accum 2     --num_epochs 9    --partial_load True     --features_path ${DATA_DIR}/raw_data/features_rf.hdf5    --save_path ${DATA_DIR}/saves/ftsmeteorbs18lr1e-5/ --reinforce meteor --body_save_path ${DATA_DIR}/saves/rf_model-002.pth



l_step = []

l_step.append(
    [
        ('CIDEr', 1.3965), ('Bleu_1', 0.8321), ('Bleu_2', 0.6864), ('Bleu_3', 0.5408),
        ('Bleu_4', 0.4168), ('ROUGE_L', 0.6059), ('SPICE', 0.2428), ('METEOR', 0.3036),
        ("lambda", 1.0)
    ]
)
l_step.append([
    ('CIDEr', 1.3598), ('Bleu_1', 0.8005), ('Bleu_2', 0.6505), ('Bleu_3', 0.5073),
    ('Bleu_4', 0.3869), ('ROUGE_L', 0.6036), ('SPICE', 0.2495), ('METEOR', 0.3082)
])
l_step.append([
    ('CIDEr', 1.3329), ('Bleu_1', 0.7887), ('Bleu_2', 0.6374), ('Bleu_3', 0.4946),
    ('Bleu_4', 0.3753), ('ROUGE_L', 0.5998), ('SPICE', 0.2507), ('METEOR', 0.3097)
])
l_step.append([
    ('CIDEr', 1.3308), ('Bleu_1', 0.7894), ('Bleu_2', 0.6379), ('Bleu_3', 0.4947),
    ('Bleu_4', 0.3752), ('ROUGE_L', 0.6009), ('SPICE', 0.252), ('METEOR', 0.3105)
])
l_step.append([
    ('CIDEr', 1.3222), ('Bleu_1', 0.7847), ('Bleu_2', 0.6321), ('Bleu_3', 0.4892),
    ('Bleu_4', 0.3704), ('ROUGE_L', 0.5992), ('SPICE', 0.2527), ('METEOR', 0.3108)
])
l_step.append([
    ('CIDEr', 1.3198), ('Bleu_1', 0.7836), ('Bleu_2', 0.631), ('Bleu_3', 0.4882),
    ('Bleu_4', 0.3695), ('ROUGE_L', 0.5983), ('SPICE', 0.2541), ('METEOR', 0.3109)
])
l_step.append([
    ('CIDEr', 1.3236), ('Bleu_1', 0.7873), ('Bleu_2', 0.6337), ('Bleu_3', 0.4901),
    ('Bleu_4', 0.3712), ('ROUGE_L', 0.6), ('SPICE', 0.2523), ('METEOR', 0.3109)
])
l_step.append([
    ('CIDEr', 1.3168), ('Bleu_1', 0.7838), ('Bleu_2', 0.6298), ('Bleu_3', 0.4861),
    ('Bleu_4', 0.3674), ('ROUGE_L', 0.5994), ('SPICE', 0.2532), ('METEOR', 0.3117)
])
l_step.append([
    ('CIDEr', 1.3158), ('Bleu_1', 0.7836), ('Bleu_2', 0.6296), ('Bleu_3', 0.4863),
    ('Bleu_4', 0.3679), ('ROUGE_L', 0.5984), ('SPICE', 0.2526), ('METEOR', 0.3113)
])


def todict(l):
    if isinstance(l, dict):
        d = l
    else:
        d = {k: v for k, v in l}
    return d


l_step = [todict(l) for l in l_step]
for i, l in enumerate(l_step):
    l['lambda'] = 1 - i / len(l_step)


l_steplr = []
# python train.py --optim_type radam --seed 775533 --sched_type custom_warmup_anneal  \
#     --warmup 1 --anneal_coeff 0.8 --anneal_every_epoch 1 --lr 5e-6 --batch_size 18 \
#     --is_end_to_end False --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/ --num_accum 2\
#     --body_save_path ${DATA_DIR}/saves/rf_model-002.pth --num_epochs 9\
#     --partial_load True     --features_path ${DATA_DIR}/raw_data/features_rf.hdf5\
#     --save_path ${DATA_DIR}/saves/ftsmeteorbs18lr5e-6/ --reinforce meteor

l_steplr.append(
    [
        ('CIDEr', 1.3965), ('Bleu_1', 0.8321), ('Bleu_2', 0.6864), ('Bleu_3', 0.5408),
        ('Bleu_4', 0.4168), ('ROUGE_L', 0.6059), ('SPICE', 0.2428), ('METEOR', 0.3036),
        ("lambda", 1.0)
    ]
)
l_steplr.append([('CIDEr', 1.3728), ('Bleu_1', 0.8033), ('Bleu_2', 0.6547), ('Bleu_3', 0.5118), ('Bleu_4', 0.3915), ('ROUGE_L', 0.6029), ('SPICE', 0.249), ('METEOR', 0.3077)])
l_steplr.append([('CIDEr', 1.3629), ('Bleu_1', 0.801), ('Bleu_2', 0.6503), ('Bleu_3', 0.5065), ('Bleu_4', 0.3859), ('ROUGE_L', 0.6022), ('SPICE', 0.2495), ('METEOR', 0.3089)])
l_steplr.append([('CIDEr', 1.3507), ('Bleu_1', 0.7956), ('Bleu_2', 0.6449), ('Bleu_3', 0.5019), ('Bleu_4', 0.3819), ('ROUGE_L', 0.6017), ('SPICE', 0.2504), ('METEOR', 0.3092)])
l_steplr.append([('CIDEr', 1.3488), ('Bleu_1', 0.7938), ('Bleu_2', 0.6433), ('Bleu_3', 0.5005), ('Bleu_4', 0.3806), ('ROUGE_L', 0.602), ('SPICE', 0.2517), ('METEOR', 0.3104)])
l_steplr.append([('CIDEr', 1.3468), ('Bleu_1', 0.7937), ('Bleu_2', 0.6428), ('Bleu_3', 0.4996), ('Bleu_4', 0.3796), ('ROUGE_L', 0.6021), ('SPICE', 0.2517), ('METEOR', 0.3102)])
l_steplr.append([('CIDEr', 1.345), ('Bleu_1', 0.7926), ('Bleu_2', 0.6417), ('Bleu_3', 0.4989), ('Bleu_4', 0.3791), ('ROUGE_L', 0.6021), ('SPICE', 0.2519), ('METEOR', 0.3104)])
l_steplr.append([('CIDEr', 1.3431), ('Bleu_1', 0.7925), ('Bleu_2', 0.6408), ('Bleu_3', 0.4978), ('Bleu_4', 0.3783), ('ROUGE_L', 0.6018), ('SPICE', 0.252), ('METEOR', 0.3102)])
l_steplr.append([('CIDEr', 1.3435), ('Bleu_1', 0.7921), ('Bleu_2', 0.6409), ('Bleu_3', 0.4979), ('Bleu_4', 0.3783), ('ROUGE_L', 0.6025), ('SPICE', 0.2518), ('METEOR', 0.3107)])
l_steplr.append([('CIDEr', 1.3404), ('Bleu_1', 0.7918), ('Bleu_2', 0.6401), ('Bleu_3', 0.4971), ('Bleu_4', 0.3776), ('ROUGE_L', 0.6024), ('SPICE', 0.2515), ('METEOR', 0.3105)])
l_steplr.append([('CIDEr', 1.3404), ('Bleu_1', 0.7918), ('Bleu_2', 0.6401), ('Bleu_3', 0.4971), ('Bleu_4', 0.3776), ('ROUGE_L', 0.6024), ('SPICE', 0.2515), ('METEOR', 0.3105)])
l_steplr = [todict(l) for l in l_steplr]
for i, l in enumerate(l_steplr):
    l['lambda'] = 1 - i / len(l_steplr)

# python scripts/test_singlegpu.py --is_end_to_end False --features_path ${DATA_DIR}/raw_data/features_rf.hdf5 --save_model_path /data/rame/ExpansionNet_v2/github_ignore_material/saves/ftsmeteorbs18lr5e-6/checkpoint_2023-03-10-08:23:15_epoch8it6293bs18_meteor_.pth
# python train.py --optim_type radam --seed 775533 --sched_type custom_warmup_anneal  \
#     --warmup 1 --anneal_coeff 0.8 --anneal_every_epoch 1 --lr 5e-6 --batch_size 18 \
#     --is_end_to_end False --images_path ${DATA_DIR}/raw_data/MS_COCO_2014/ --num_accum 2\
#     --num_epochs 12\
#     --partial_load True     --features_path ${DATA_DIR}/raw_data/features_rf.hdf5\
#     --save_path ${DATA_DIR}/saves/ftsmeteorbs18lr5e-6/ --reinforce meteor
