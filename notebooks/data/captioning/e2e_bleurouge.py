def todict(l):
    return {k: v for k, v in l}


# python test.py --N_enc 3 --N_dec 3 --model_dim 512 --num_gpus 1 --eval_beam_sizes [5] --is_end_to_end True --save_model_path /data/rame/ExpansionNet_v2/github_ignore_material/saves/e2erouge/rouge_model_epoch5.pth /data/rame/ExpansionNet_v2/github_ignore_material/saves/wa/bleu_model_epoch3.pth --ensemble wa_5

l_wa_rougebleu = []

l_wa_rougebleu.append(
    [
        ('CIDEr', 1.3389), ('Bleu_1', 0.8572), ('Bleu_2', 0.6917), ('Bleu_3', 0.5305),
        ('Bleu_4', 0.3982), ('ROUGE_L', 0.5972), ('SPICE', 0.2315), ('METEOR', 0.2913),
        ("lambda", 1.)
    ]
)

l_wa_rougebleu.append(
    [
        ('CIDEr', 1.3656), ('Bleu_1', 0.8543), ('Bleu_2', 0.6967), ('Bleu_3', 0.5408),
        ('Bleu_4', 0.4106), ('ROUGE_L', 0.603), ('SPICE', 0.2381), ('METEOR', 0.2966),
        ("lambda", .8)
    ]
)

l_wa_rougebleu.append(
    [
        ('CIDEr', 1.3764), ('Bleu_1', 0.8474), ('Bleu_2', 0.6962), ('Bleu_3', 0.5447),
        ('Bleu_4', 0.4163), ('ROUGE_L', 0.6066), ('SPICE', 0.2422), ('METEOR', 0.3003),
        ("lambda", 0.6)
    ]
)

l_wa_rougebleu.append(
    [
        ('CIDEr', 1.3744), ('Bleu_1', 0.8418), ('Bleu_2', 0.6936), ('Bleu_3', 0.5444),
        ('Bleu_4', 0.4179), ('ROUGE_L', 0.608), ('SPICE', 0.2425), ('METEOR', 0.3007),
        ("lambda", 0.5)
    ]
)

l_wa_rougebleu.append(
    [
        ('CIDEr', 1.375), ('Bleu_1', 0.8379), ('Bleu_2', 0.6921), ('Bleu_3', 0.5449),
        ('Bleu_4', 0.4194), ('ROUGE_L', 0.6095), ('SPICE', 0.2421), ('METEOR', 0.3013),
        ("lambda", 0.4)
    ]
)

l_wa_rougebleu.append(
    [
        ('CIDEr', 1.3633), ('Bleu_1', 0.8274), ('Bleu_2', 0.6846), ('Bleu_3', 0.5411),
        ('Bleu_4', 0.4189), ('ROUGE_L', 0.6095), ('SPICE', 0.2393), ('METEOR', 0.2999),
        ("lambda", 0.2)
    ]
)

l_wa_rougebleu.append(
    [
        ('CIDEr', 1.3481), ('Bleu_1', 0.8179), ('Bleu_2', 0.6766), ('Bleu_3', 0.5349),
        ('Bleu_4', 0.415), ('ROUGE_L', 0.6075), ('SPICE', 0.2369), ('METEOR', 0.2982),
        ("lambda", 0.)
    ]
)

# [
#     ('CIDEr', 1.3281), ('Bleu_1', 0.8521), ('Bleu_2', 0.6888), ('Bleu_3', 0.5296),
#     ('Bleu_4', 0.3984), ('ROUGE_L', 0.5973), ('SPICE', 0.2305), ('METEOR', 0.2901),
#     ("lambda", [1, 0.5, -0.5])
# ]
# [
#     ('CIDEr', 1.3433), ('Bleu_1', 0.8295), ('Bleu_2', 0.6834), ('Bleu_3', 0.5372),
#     ('Bleu_4', 0.4134), ('ROUGE_L', 0.6068), ('SPICE', 0.2382), ('METEOR', 0.2972),
#     ("lambda", [0.5, 1, -0.5])
# ]
# [
#     ('CIDEr', 1.2944), ('Bleu_1', 0.8406), ('Bleu_2', 0.6798), ('Bleu_3', 0.5222),
#     ('Bleu_4', 0.3921), ('ROUGE_L', 0.5954), ('SPICE', 0.2269), ('METEOR', 0.2869),
#     ("lambdas", [1.,1.,-1])
# ]
# [
#     ('CIDEr', 1.3311), ('Bleu_1', 0.8429), ('Bleu_2', 0.6878), ('Bleu_3', 0.5338),
#     ('Bleu_4', 0.4048), ('ROUGE_L', 0.602), ('SPICE', 0.2344), ('METEOR', 0.2937),
#     ("lambdas", [0.8, 0.8, -0.6])
# ]

l_wa_rougebleu = [todict(l) for l in l_wa_rougebleu]

l_step_rougebleu = []

l_step_rougebleu.append(
    [
        ('CIDEr', 1.3965), ('Bleu_1', 0.8321), ('Bleu_2', 0.6864), ('Bleu_3', 0.5408),
        ('Bleu_4', 0.4168), ('ROUGE_L', 0.6059), ('SPICE', 0.2428), ('METEOR', 0.3036),
        ("lambda", 1.0)
    ]
)
l_step_rougebleu.append(
    [
        ('CIDEr', 1.388), ('Bleu_1', 0.8381), ('Bleu_2', 0.6901), ('Bleu_3', 0.5428),
        ('Bleu_4', 0.4179), ('ROUGE_L', 0.607), ('SPICE', 0.2428), ('METEOR', 0.302),
        ("lambda", 0.8)
    ]
)
l_step_rougebleu.append(
    [
        ('CIDEr', 1.3823), ('Bleu_1', 0.8379), ('Bleu_2', 0.6881), ('Bleu_3', 0.5407),
        ('Bleu_4', 0.4162), ('ROUGE_L', 0.6075), ('SPICE', 0.2437), ('METEOR', 0.302),
        ("lambda", 0.6)
    ]
)
l_step_rougebleu.append(
    [
        ('CIDEr', 1.3802), ('Bleu_1', 0.841), ('Bleu_2', 0.6914), ('Bleu_3', 0.543),
        ('Bleu_4', 0.4175), ('ROUGE_L', 0.6084), ('SPICE', 0.2434), ('METEOR', 0.3026),
        ("lambda", 0.4)
    ]
)
l_step_rougebleu.append(
    [
        ('CIDEr', 1.3784), ('Bleu_1', 0.8398), ('Bleu_2', 0.6913), ('Bleu_3', 0.5438),
        ('Bleu_4', 0.4181), ('ROUGE_L', 0.6088), ('SPICE', 0.2426), ('METEOR', 0.3016),
        ("lambda", 0.2)
    ]
)
# l_step_rougebleu.append([
#     ('CIDEr', 1.375), ('Bleu_1', 0.8416), ('Bleu_2', 0.6914), ('Bleu_3', 0.5424),
#     ('Bleu_4', 0.4165), ('ROUGE_L', 0.6071), ('SPICE', 0.2419), ('METEOR', 0.3009), ("lambda", 0.0)
# ])

# Saved to checkpoint_2023-03-04-06:43:40_epoch4it6293bs18_bleu,rouge_.pth
l_step_rougebleu = [todict(l) for l in l_step_rougebleu]
