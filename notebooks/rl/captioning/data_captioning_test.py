def todict(l):
    d = {k: v for k, v in l}
    if "lambda" in d:
        d["lambda"] = round(1 - d["lambda"], 1)
    return d


cider = todict([('CIDEr', 1.3958), ('Bleu_1', 0.8271), ('Bleu_2', 0.6772), ('Bleu_3', 0.533), ('Bleu_4', 0.4098), ('ROUGE_L', 0.6034), ('SPICE', 0.2441), ('METEOR', 0.3016)])


# (pytorch) bash-4.4$ cat jz_inf_fts_multirougebleu_test.slurm_1129560.out | grep CIDEr
l_multi = []
l_multi.append(
    [
        ('CIDEr', 1.3658), ('Bleu_1', 0.8182), ('Bleu_2', 0.6737), ('Bleu_3', 0.5324),
        ('Bleu_4', 0.4107), ('ROUGE_L', 0.6056), ('SPICE', 0.2409), ('METEOR', 0.2986),
        ('lambda', 0.0)
    ]
)
l_multi.append([('CIDEr', 1.3721), ('Bleu_1', 0.8262), ('Bleu_2', 0.6786), ('Bleu_3', 0.535), ('Bleu_4', 0.412), ('ROUGE_L', 0.6061), ('SPICE', 0.2419), ('METEOR', 0.2994), ("lambda", 0.1)])
l_multi.append([('CIDEr', 1.3811), ('Bleu_1', 0.832), ('Bleu_2', 0.6826), ('Bleu_3', 0.5381), ('Bleu_4', 0.4142), ('ROUGE_L', 0.6066), ('SPICE', 0.2435), ('METEOR', 0.3006), ("lambda", 0.2)])
l_multi.append([('CIDEr', 1.3827), ('Bleu_1', 0.8348), ('Bleu_2', 0.6838), ('Bleu_3', 0.5377), ('Bleu_4', 0.4131), ('ROUGE_L', 0.6054), ('SPICE', 0.244), ('METEOR', 0.3005), ("lambda", 0.3)])
l_multi.append([('CIDEr', 1.3804), ('Bleu_1', 0.8375), ('Bleu_2', 0.6857), ('Bleu_3', 0.5386), ('Bleu_4', 0.4133), ('ROUGE_L', 0.6061), ('SPICE', 0.243), ('METEOR', 0.3003), ("lambda", 0.4)])
l_multi.append([('CIDEr', 1.3799), ('Bleu_1', 0.8389), ('Bleu_2', 0.6863), ('Bleu_3', 0.5387), ('Bleu_4', 0.4126), ('ROUGE_L', 0.6056), ('SPICE', 0.2434), ('METEOR', 0.3003), ("lambda", 0.5)])
l_multi.append([('CIDEr', 1.3767), ('Bleu_1', 0.8408), ('Bleu_2', 0.6864), ('Bleu_3', 0.5372), ('Bleu_4', 0.4106), ('ROUGE_L', 0.6046), ('SPICE', 0.2433), ('METEOR', 0.2992), ("lambda", 0.6)])
l_multi.append([('CIDEr', 1.3798), ('Bleu_1', 0.8414), ('Bleu_2', 0.6869), ('Bleu_3', 0.5377), ('Bleu_4', 0.4114), ('ROUGE_L', 0.605), ('SPICE', 0.2436), ('METEOR', 0.2995), ("lambda", 0.7)])
l_multi.append([('CIDEr', 1.3772), ('Bleu_1', 0.8438), ('Bleu_2', 0.6869), ('Bleu_3', 0.5363), ('Bleu_4', 0.4089), ('ROUGE_L', 0.6024), ('SPICE', 0.2433), ('METEOR', 0.2981), ("lambda", 0.8)])
l_multi.append([('CIDEr', 1.3638), ('Bleu_1', 0.845), ('Bleu_2', 0.6858), ('Bleu_3', 0.533), ('Bleu_4', 0.4044), ('ROUGE_L', 0.6017), ('SPICE', 0.2427), ('METEOR', 0.2972), ("lambda", 0.9)])
l_multi.append(
    [
        ('CIDEr', 1.3717), ('Bleu_1', 0.8462), ('Bleu_2', 0.6866), ('Bleu_3', 0.5337),
        ('Bleu_4', 0.4051), ('ROUGE_L', 0.6013), ('SPICE', 0.2427), ('METEOR', 0.2962),
        ('lambda', 1.0)
    ]
)

# ftsrouge9bleu1bs18lr1e-5/checkpoint_2023-04-24-00:50:18_epoch5it6293bs18_bleu-1,rouge-9_.pth ftsrouge4bleu1bs18lr1e-5/checkpoint_2023-04-15-15:36:12_epoch5it6293bs18_bleu-1,rouge-4_.pth ftsrouge1bleu05bs18lr1e-5/checkpoint_2023-04-12-01:51:18_epoch5it6293bs18_bleu-0.5,rouge-1_.pth ftsrouge3bleu2bs18lr1e-5/checkpoint_2023-04-15-08:54:05_epoch5it6293bs18_bleu-2,rouge-3_.pth fts/model_bleurouge_epoch5it6293bs18.pth ftsrouge2bleu3bs18lr1e-5/checkpoint_2023-04-15-17:55:07_epoch5it6293bs18_bleu-3,rouge-2_.pth fts/model_bleu1rouge05_epoch5it6293bs18.pth ftsrouge1bleu4bs18lr1e-5/checkpoint_2023-04-14-06:15:59_epoch5it6293bs18_bleu-4,rouge-1_.pth ftsrouge1bleu9bs18lr1e-5/checkpoint_2023-04-24-00:50:17_epoch5it6293bs18_bleu-9,rouge-1_.pth

# (pytorch) bash-4.4$ cat jz_inf_fts_rougebleu_test.slurm_1129537.out | grep lambda
l_wa = []
l_wa.append([('CIDEr', 1.3658), ('Bleu_1', 0.8182), ('Bleu_2', 0.6737), ('Bleu_3', 0.5324), ('Bleu_4', 0.4107), ('ROUGE_L', 0.6056), ('SPICE', 0.2409), ('METEOR', 0.2986), ('lambda', 0.0)])
l_wa.append([('CIDEr', 1.3704), ('Bleu_1', 0.8215), ('Bleu_2', 0.6755), ('Bleu_3', 0.5335), ('Bleu_4', 0.4112), ('ROUGE_L', 0.606), ('SPICE', 0.2416), ('METEOR', 0.2988), ('lambda', 0.1)])
l_wa.append([('CIDEr', 1.3748), ('Bleu_1', 0.8248), ('Bleu_2', 0.6777), ('Bleu_3', 0.5349), ('Bleu_4', 0.4121), ('ROUGE_L', 0.6059), ('SPICE', 0.2418), ('METEOR', 0.2988), ('lambda', 0.2)])
l_wa.append([('CIDEr', 1.3785), ('Bleu_1', 0.828), ('Bleu_2', 0.6798), ('Bleu_3', 0.5361), ('Bleu_4', 0.4126), ('ROUGE_L', 0.6063), ('SPICE', 0.2423), ('METEOR', 0.2993), ('lambda', 0.3)])
l_wa.append([('CIDEr', 1.3793), ('Bleu_1', 0.832), ('Bleu_2', 0.682), ('Bleu_3', 0.5365), ('Bleu_4', 0.4119), ('ROUGE_L', 0.6057), ('SPICE', 0.2424), ('METEOR', 0.299), ('lambda', 0.4)])
l_wa.append([('CIDEr', 1.381), ('Bleu_1', 0.8361), ('Bleu_2', 0.684), ('Bleu_3', 0.5372), ('Bleu_4', 0.412), ('ROUGE_L', 0.6054), ('SPICE', 0.2429), ('METEOR', 0.299), ('lambda', 0.5)])
l_wa.append([('CIDEr', 1.3772), ('Bleu_1', 0.8392), ('Bleu_2', 0.6853), ('Bleu_3', 0.5369), ('Bleu_4', 0.4108), ('ROUGE_L', 0.6042), ('SPICE', 0.2433), ('METEOR', 0.2983), ('lambda', 0.6)])
l_wa.append([('CIDEr', 1.3736), ('Bleu_1', 0.8415), ('Bleu_2', 0.6853), ('Bleu_3', 0.5352), ('Bleu_4', 0.408), ('ROUGE_L', 0.6032), ('SPICE', 0.2427), ('METEOR', 0.2973), ('lambda', 0.7)])
l_wa.append([('CIDEr', 1.3762), ('Bleu_1', 0.8435), ('Bleu_2', 0.6864), ('Bleu_3', 0.5356), ('Bleu_4', 0.4081), ('ROUGE_L', 0.6028), ('SPICE', 0.2429), ('METEOR', 0.2971), ('lambda', 0.8)])
l_wa.append([('CIDEr', 1.3747), ('Bleu_1', 0.8451), ('Bleu_2', 0.6866), ('Bleu_3', 0.5347), ('Bleu_4', 0.4066), ('ROUGE_L', 0.6021), ('SPICE', 0.2427), ('METEOR', 0.2966), ('lambda', 0.9)])
l_wa.append([('CIDEr', 1.3717), ('Bleu_1', 0.8462), ('Bleu_2', 0.6866), ('Bleu_3', 0.5337), ('Bleu_4', 0.4051), ('ROUGE_L', 0.6013), ('SPICE', 0.2427), ('METEOR', 0.2962), ('lambda', 1.0)])

l_multi = [todict(l) for l in l_multi]
l_multi = l_multi[::-1]
l_wa = [todict(l) for l in l_wa]
l_wa = l_wa[::-1]
