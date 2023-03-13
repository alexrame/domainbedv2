# python test.py --features_path ${DATA_DIR}/raw_data/features_rf.hdf5 --is_end_to_end False --save_model_path /data/rame/ExpansionNet_v2/github_ignore_material/saves/ftsbleubs18lr1e-5/checkpoint_2023-03-07-21:05:09_epoch5it6293bs18_bleu_.pth /data/rame/ExpansionNet_v2/github_ignore_material/saves/ftsrougebs18lr1e-5/checkpoint_2023-03-07-21:07:31_epoch5it6293bs18_rouge_.pth --ensemble wa --coeffs [1.5,-0.5]

l_wa_epoch5 = []
l_wa_epoch5.append(
    [
        ('CIDEr', 1.3707), ('Bleu_1', 0.8497), ('Bleu_2', 0.6931), ('Bleu_3', 0.538),
        ('Bleu_4', 0.4086), ('ROUGE_L', 0.6038), ('SPICE', 0.2413), ('METEOR', 0.299),
        ('lambda', [1.0, 0.0])
    ]
)
l_wa_epoch5.append(
    [
        ('CIDEr', 1.3756), ('Bleu_1', 0.8486), ('Bleu_2', 0.6939), ('Bleu_3', 0.5399),
        ('Bleu_4', 0.4107), ('ROUGE_L', 0.6047), ('SPICE', 0.2413), ('METEOR', 0.3002),
        ('lambda', [0.9, 0.1])
    ]
)
l_wa_epoch5.append(
    [
        ('CIDEr', 1.3789), ('Bleu_1', 0.846), ('Bleu_2', 0.6932), ('Bleu_3', 0.5412),
        ('Bleu_4', 0.4132), ('ROUGE_L', 0.6054), ('SPICE', 0.2418), ('METEOR', 0.3006),
        ('lambda', [0.8, 0.2])
    ]
)
l_wa_epoch5.append(
    [
        ('CIDEr', 1.381), ('Bleu_1', 0.8342), ('Bleu_2', 0.6895), ('Bleu_3', 0.5443),
        ('Bleu_4', 0.4205), ('ROUGE_L', 0.6096), ('SPICE', 0.2412), ('METEOR', 0.3019),
        ('lambda', [0.30000000000000004, 0.7])
    ]
)
l_wa_epoch5.append(
    [
        ('CIDEr', 1.3732), ('Bleu_1', 0.8265), ('Bleu_2', 0.6842), ('Bleu_3', 0.5411),
        ('Bleu_4', 0.419), ('ROUGE_L', 0.6095), ('SPICE', 0.2398), ('METEOR', 0.3013),
        ('lambda', [0.09999999999999998, 0.9])
    ]
)
l_wa_epoch5.append(
    [
        ('CIDEr', 1.3662), ('Bleu_1', 0.8213), ('Bleu_2', 0.6803), ('Bleu_3', 0.5389),
        ('Bleu_4', 0.4181), ('ROUGE_L', 0.6098), ('SPICE', 0.2384), ('METEOR', 0.3007),
        ('lambda', [-0.10000000000000009, 1.1])
    ]
)

l_wa_epoch5.append(
    [
        ('CIDEr', 1.3641), ('Bleu_1', 0.816), ('Bleu_2', 0.6768), ('Bleu_3', 0.5371),
        ('Bleu_4', 0.4173), ('ROUGE_L', 0.609), ('SPICE', 0.2376), ('METEOR', 0.3),
        ('lambda', [-0.3, 1.3])
    ]
)
l_wa_epoch5.append(
    [
        ('CIDEr', 1.3701), ('Bleu_1', 0.8234), ('Bleu_2', 0.682), ('Bleu_3', 0.54),
        ('Bleu_4', 0.4187), ('ROUGE_L', 0.61), ('SPICE', 0.2388), ('METEOR', 0.3011),
        ('lambda', [0.0, 1.0])
    ]
)
l_wa_epoch5.append(
    [
        ('CIDEr', 1.3795), ('Bleu_1', 0.8325), ('Bleu_2', 0.6882), ('Bleu_3', 0.5436),
        ('Bleu_4', 0.4203), ('ROUGE_L', 0.6093), ('SPICE', 0.2409), ('METEOR', 0.3017),
        ('lambda', [0.25, 0.75])
    ]
)

l_wa_epoch5.append(
    [
        ('CIDEr', 1.3868), ('Bleu_1', 0.8395), ('Bleu_2', 0.6922), ('Bleu_3', 0.5448),
        ('Bleu_4', 0.4194), ('ROUGE_L', 0.6095), ('SPICE', 0.2422), ('METEOR', 0.3022),
        ('lambda', [0.5, 0.5])
    ]
)
l_wa_epoch5.append(
    [
        ('CIDEr', 1.3803), ('Bleu_1', 0.8449), ('Bleu_2', 0.6931), ('Bleu_3', 0.5419),
        ('Bleu_4', 0.4147), ('ROUGE_L', 0.606), ('SPICE', 0.2418), ('METEOR', 0.3012),
        ('lambda', [0.75, 0.25])
    ]
)
l_wa_epoch5.append(
    [
        ('CIDEr', 1.3707), ('Bleu_1', 0.8497), ('Bleu_2', 0.6931), ('Bleu_3', 0.538),
        ('Bleu_4', 0.4086), ('ROUGE_L', 0.6038), ('SPICE', 0.2413), ('METEOR', 0.299),
        ('lambda', [1., 0.])
    ]
)
l_wa_epoch5.append(
    [
        ('CIDEr', 1.3704), ('Bleu_1', 0.8506), ('Bleu_2', 0.6931), ('Bleu_3', 0.5374),
        ('Bleu_4', 0.4078), ('ROUGE_L', 0.6034), ('SPICE', 0.2412), ('METEOR', 0.2988),
        ('lambda', [1.05, -0.05])
    ]
)
l_wa_epoch5.append(
    [
        ('CIDEr', 1.3676), ('Bleu_1', 0.8511), ('Bleu_2', 0.6928), ('Bleu_3', 0.5366),
        ('Bleu_4', 0.407), ('ROUGE_L', 0.6028), ('SPICE', 0.2412), ('METEOR', 0.2982),
        ('lambda', [1.1, -0.1])
    ]
)
l_wa_epoch5.append(
    [
        ('CIDEr', 1.362), ('Bleu_1', 0.851), ('Bleu_2', 0.6911), ('Bleu_3', 0.5335),
        ('Bleu_4', 0.4029), ('ROUGE_L', 0.6015), ('SPICE', 0.241), ('METEOR', 0.2972),
        ('lambda', [1.2, -0.2])
    ]
)
l_wa_epoch5.append(
    [
        ('CIDEr', 1.3643),
        ('Bleu_1', 0.8508),
        ('Bleu_2', 0.6918),
        ('Bleu_3', 0.5348),
        ('Bleu_4', 0.4046),
        ('ROUGE_L', 0.602),
        ('SPICE', 0.2413),
        ('METEOR', 0.2977),
        ('lambda', [1.15, -0.15]),
    ]
)
l_wa_epoch5.append(
    [
        ('CIDEr', 1.36), ('Bleu_1', 0.8519), ('Bleu_2', 0.6909), ('Bleu_3', 0.5325),
        ('Bleu_4', 0.4015), ('ROUGE_L', 0.6012), ('SPICE', 0.2409), ('METEOR', 0.2968),
        ('lambda', [1.25, -0.25])
    ]
)
l_wa_epoch5.append(
    [
        ('CIDEr', 1.3578), ('Bleu_1', 0.8518), ('Bleu_2', 0.6903), ('Bleu_3', 0.5314),
        ('Bleu_4', 0.4), ('ROUGE_L', 0.6006), ('SPICE', 0.2408), ('METEOR', 0.2964),
        ('lambda', [1.3, -0.3])
    ]
)
l_wa_epoch5.append(
    [
        ('CIDEr', 1.3497), ('Bleu_1', 0.8514), ('Bleu_2', 0.6876), ('Bleu_3', 0.5271),
        ('Bleu_4', 0.3949), ('ROUGE_L', 0.5987), ('SPICE', 0.2398), ('METEOR', 0.2948),
        ('lambda', [1.4, -0.4])
    ]
)
l_wa_epoch5.append(
    [
        ('CIDEr', 1.3427), ('Bleu_1', 0.8507), ('Bleu_2', 0.684), ('Bleu_3', 0.5225),
        ('Bleu_4', 0.3897), ('ROUGE_L', 0.5964), ('SPICE', 0.2385), ('METEOR', 0.2933),
        ('lambda', [1.5, -0.5])
    ]
)
l_wa_epoch5.append(
    [
        ('CIDEr', 1.2944), ('Bleu_1', 0.8471), ('Bleu_2', 0.6693), ('Bleu_3', 0.5),
        ('Bleu_4', 0.3647), ('ROUGE_L', 0.5841), ('SPICE', 0.2319), ('METEOR', 0.2852),
        ('lambda', [2.0, -1.0])
    ]
)
# l_wa_epoch5.append(
#     [
#         ('CIDEr', 1.2047), ('Bleu_1', 0.8332), ('Bleu_2', 0.64), ('Bleu_3', 0.4626),
#         ('Bleu_4', 0.3252), ('ROUGE_L', 0.5647), ('SPICE', 0.2191), ('METEOR', 0.2711),
#         ('lambda', [2.5, -1.5])
#     ]
# )
# l_wa_epoch5.append(
#     [
#         ('CIDEr', 0.4609), ('Bleu_1', 0.3505), ('Bleu_2', 0.2056), ('Bleu_3', 0.1067),
#         ('Bleu_4', 0.0524), ('ROUGE_L', 0.3504), ('SPICE', 0.1114), ('METEOR', 0.1571),
#         ('lambda', [5.0, -4.0])
#     ]
# )

# epoch10

l_wa_epoch10 = []
l_wa_epoch10.append(
    [
        ('CIDEr', 1.3572), ('Bleu_1', 0.8565), ('Bleu_2', 0.6938), ('Bleu_3', 0.5333),
        ('Bleu_4', 0.4017), ('ROUGE_L', 0.5988), ('SPICE', 0.2326), ('METEOR', 0.2924),
        ('lambda', [1.2, -0.2])
    ]
)
l_wa_epoch10.append(
    [
        ('CIDEr', 1.3645), ('Bleu_1', 0.8554), ('Bleu_2', 0.6945), ('Bleu_3', 0.5357),
        ('Bleu_4', 0.4049), ('ROUGE_L', 0.6007), ('SPICE', 0.235), ('METEOR', 0.2944),
        ('lambda', [1.1, -0.1])
    ]
)
l_wa_epoch10.append(
    [
        ('CIDEr', 1.3713), ('Bleu_1', 0.8534), ('Bleu_2', 0.6949), ('Bleu_3', 0.5377),
        ('Bleu_4', 0.4077), ('ROUGE_L', 0.6022), ('SPICE', 0.237), ('METEOR', 0.2962),
        ('lambda', [1.0, 0.0])
    ]
)

l_wa_epoch10.append(
    [
        ('CIDEr', 1.3766), ('Bleu_1', 0.8524), ('Bleu_2', 0.6961), ('Bleu_3', 0.5404),
        ('Bleu_4', 0.4105), ('ROUGE_L', 0.6037), ('SPICE', 0.2391), ('METEOR', 0.298),
        ('lambda', [0.9, 0.1])
    ]
)
l_wa_epoch10.append(
    [
        ('CIDEr', 1.3802), ('Bleu_1', 0.8503), ('Bleu_2', 0.6959), ('Bleu_3', 0.542),
        ('Bleu_4', 0.4132), ('ROUGE_L', 0.6049), ('SPICE', 0.2409), ('METEOR', 0.2995),
        ('lambda', [0.8, 0.2])
    ]
)
l_wa_epoch10.append(
    [
        ('CIDEr', 1.3834), ('Bleu_1', 0.847), ('Bleu_2', 0.6949), ('Bleu_3', 0.5432),
        ('Bleu_4', 0.4151), ('ROUGE_L', 0.606), ('SPICE', 0.2421), ('METEOR', 0.3008),
        ('lambda', [0.7, 0.3])
    ]
)
l_wa_epoch10.append(
    [
        ('CIDEr', 1.3861), ('Bleu_1', 0.8445), ('Bleu_2', 0.6949), ('Bleu_3', 0.5448),
        ('Bleu_4', 0.4175), ('ROUGE_L', 0.6075), ('SPICE', 0.2427), ('METEOR', 0.3019),
        ('lambda', [0.6, 0.4])
    ]
)
l_wa_epoch10.append(
    [
        ('CIDEr', 1.3842), ('Bleu_1', 0.8416), ('Bleu_2', 0.6935), ('Bleu_3', 0.545),
        ('Bleu_4', 0.4184), ('ROUGE_L', 0.6082), ('SPICE', 0.2421), ('METEOR', 0.3019),
        ('lambda', [0.5, 0.5])
    ]
)
l_wa_epoch10.append(
    [
        ('CIDEr', 1.3827), ('Bleu_1', 0.8382), ('Bleu_2', 0.6918), ('Bleu_3', 0.5453),
        ('Bleu_4', 0.4202), ('ROUGE_L', 0.6087), ('SPICE', 0.2418), ('METEOR', 0.3017),
        ('lambda', [0.4, 0.6])
    ]
)
l_wa_epoch10.append(
    [
        ('CIDEr', 1.3801), ('Bleu_1', 0.8349), ('Bleu_2', 0.69), ('Bleu_3', 0.5447),
        ('Bleu_4', 0.4209), ('ROUGE_L', 0.609), ('SPICE', 0.2413), ('METEOR', 0.3019),
        ('lambda', [0.30000000000000004, 0.7])
    ]
)
l_wa_epoch10.append(
    [
        ('CIDEr', 1.3745), ('Bleu_1', 0.8315), ('Bleu_2', 0.6873), ('Bleu_3', 0.5426),
        ('Bleu_4', 0.4193), ('ROUGE_L', 0.6087), ('SPICE', 0.2402), ('METEOR', 0.3013),
        ('lambda', [0.19999999999999996, 0.8])
    ]
)
l_wa_epoch10.append(
    [
        ('CIDEr', 1.3708), ('Bleu_1', 0.8275), ('Bleu_2', 0.6846), ('Bleu_3', 0.5411),
        ('Bleu_4', 0.4188), ('ROUGE_L', 0.6094), ('SPICE', 0.2397), ('METEOR', 0.3011),
        ('lambda', [0.09999999999999998, 0.9])
    ]
)
l_wa_epoch10.append(
    [
        ('CIDEr', 1.3668), ('Bleu_1', 0.8235), ('Bleu_2', 0.6816), ('Bleu_3', 0.5394),
        ('Bleu_4', 0.4182), ('ROUGE_L', 0.6094), ('SPICE', 0.2392), ('METEOR', 0.3009),
        ('lambda', [0.0, 1.0])
    ]
)
l_wa_epoch10.append(
    [
        ('CIDEr', 1.3614), ('Bleu_1', 0.8204), ('Bleu_2', 0.6792), ('Bleu_3', 0.5377),
        ('Bleu_4', 0.417), ('ROUGE_L', 0.6087), ('SPICE', 0.2386), ('METEOR', 0.3001),
        ('lambda', [-0.10000000000000009, 1.1])
    ]
)
l_wa_epoch10.append(
    [
        ('CIDEr', 1.3594), ('Bleu_1', 0.8168), ('Bleu_2', 0.6769), ('Bleu_3', 0.5368),
        ('Bleu_4', 0.4171), ('ROUGE_L', 0.6089), ('SPICE', 0.238), ('METEOR', 0.2997),
        ('lambda', [-0.19999999999999996, 1.2])
    ]
)

















l_wa_epoch10.append([('CIDEr', 1.2494), ('Bleu_1', 0.8494), ('Bleu_2', 0.6633), ('Bleu_3', 0.4892), ('Bleu_4', 0.3545), ('ROUGE_L', 0.5764), ('SPICE', 0.2109), ('METEOR', 0.272), ('lambda', [2.0, -1.0])])
l_wa_epoch10.append([('CIDEr', 1.3231), ('Bleu_1', 0.8566), ('Bleu_2', 0.685), ('Bleu_3', 0.5193), ('Bleu_4', 0.3867), ('ROUGE_L', 0.5913), ('SPICE', 0.2236), ('METEOR', 0.285), ('lambda', [1.5, -0.5])])
l_wa_epoch10.append([('CIDEr', 1.3355), ('Bleu_1', 0.8577), ('Bleu_2', 0.6892), ('Bleu_3', 0.5253), ('Bleu_4', 0.3929), ('ROUGE_L', 0.5949), ('SPICE', 0.2267), ('METEOR', 0.2877), ('lambda', [1.4, -0.4])])
l_wa_epoch10.append([('CIDEr', 1.347), ('Bleu_1', 0.8575), ('Bleu_2', 0.692), ('Bleu_3', 0.5296), ('Bleu_4', 0.3976), ('ROUGE_L', 0.5972), ('SPICE', 0.23), ('METEOR', 0.2901), ('lambda', [1.3, -0.3])])
l_wa_epoch10.append([('CIDEr', 1.3533), ('Bleu_1', 0.814), ('Bleu_2', 0.6746), ('Bleu_3', 0.5351), ('Bleu_4', 0.4156), ('ROUGE_L', 0.6085), ('SPICE', 0.2375), ('METEOR', 0.2994), ('lambda', [-0.30000000000000004, 1.3])])
l_wa_epoch10.append([('CIDEr', 1.3517), ('Bleu_1', 0.8111), ('Bleu_2', 0.6728), ('Bleu_3', 0.5343), ('Bleu_4', 0.4154), ('ROUGE_L', 0.6084), ('SPICE', 0.2374), ('METEOR', 0.2992), ('lambda', [-0.3999999999999999, 1.4])])
l_wa_epoch10.append([('CIDEr', 1.3473), ('Bleu_1', 0.809), ('Bleu_2', 0.6713), ('Bleu_3', 0.5333), ('Bleu_4', 0.4155), ('ROUGE_L', 0.6076), ('SPICE', 0.237), ('METEOR', 0.299), ('lambda', [-0.5, 1.5])])
l_wa_epoch10.append([('CIDEr', 1.306), ('Bleu_1', 0.7917), ('Bleu_2', 0.6543), ('Bleu_3', 0.5179), ('Bleu_4', 0.4027), ('ROUGE_L', 0.6015), ('SPICE', 0.2331), ('METEOR', 0.2944), ('lambda', [-1.0, 2.0])])

def todict(l):
    def transform(d):
        d["lambdas"] = d["lambda"]
        assert 0.9999 < d["lambda"][0] + d["lambda"][1] < 1.001
        d["lambda"] = d["lambda"][1]
    d = {k: v for k, v in l}
    transform(d)
    return d

l_wa_epoch5 = [todict(l) for l in l_wa_epoch5]
l_wa_epoch5 = sorted(l_wa_epoch5, key=lambda d: d["lambda"], reverse=False)
l_wa_epoch10 = [todict(l) for l in l_wa_epoch10]
l_wa_epoch10 = sorted(l_wa_epoch10, key=lambda d: d["lambda"], reverse=False)

l_step = []
l_step.append(
    {
        k: v for k, v in [
            ('CIDEr', 1.3965), ('Bleu_1', 0.8321), ('Bleu_2', 0.6864), ('Bleu_3', 0.5408),
            ('Bleu_4', 0.4168), ('ROUGE_L', 0.6059), ('SPICE', 0.2428), ('METEOR',
                                                                         0.3036), ("lambda", 1.0)
        ]
    }
)
l_step.append({"Bleu_1": 0.8392, "Bleu_2": 0.6909, "Bleu_3": 0.5429, "Bleu_4": 0.4175, "CIDEr": 1.3848, "METEOR": 0.3018, "ROUGE_L": 0.6068, "SPICE": 0.242, "epoch": 0, "reinforce": "bleu,rouge", "step": 6293})
l_step.append({"Bleu_1": 0.8396, "Bleu_2": 0.691, "Bleu_3": 0.5432, "Bleu_4": 0.4182, "CIDEr": 1.3834, "METEOR": 0.3017, "ROUGE_L": 0.6077, "SPICE": 0.2424, "epoch": 0, "reinforce": "bleu,rouge", "step": 12586})
l_step.append({"Bleu_1": 0.8414, "Bleu_2": 0.6933, "Bleu_3": 0.5452, "Bleu_4": 0.4196, "CIDEr": 1.3853, "METEOR": 0.302, "ROUGE_L": 0.6089, "SPICE": 0.2434, "epoch": 0, "reinforce": "bleu,rouge", "step": 18879})
l_step.append({"Bleu_1": 0.8415, "Bleu_2": 0.6924, "Bleu_3": 0.5438, "Bleu_4": 0.4174, "CIDEr": 1.3843, "METEOR": 0.3023, "ROUGE_L": 0.6085, "SPICE": 0.2434, "epoch": 0, "reinforce": "bleu,rouge", "step": 25172})
l_step.append({"Bleu_1": 0.841, "Bleu_2": 0.6921, "Bleu_3": 0.5436, "Bleu_4": 0.4173, "CIDEr": 1.3799, "METEOR": 0.3021, "ROUGE_L": 0.6087, "SPICE": 0.2427, "epoch": 0, "reinforce": "bleu,rouge", "step": 31465})
l_step.append({"Bleu_1": 0.8414, "Bleu_2": 0.6922, "Bleu_3": 0.5438, "Bleu_4": 0.4179, "CIDEr": 1.3796, "METEOR": 0.3021, "ROUGE_L": 0.608, "SPICE": 0.2427, "epoch": 0, "reinforce": "bleu,rouge", "step": 37758})
l_step.append({"Bleu_1": 0.8408, "Bleu_2": 0.6918, "Bleu_3": 0.5437, "Bleu_4": 0.418, "CIDEr": 1.38, "METEOR": 0.3022, "ROUGE_L": 0.6076, "SPICE": 0.243, "epoch": 0, "reinforce": "bleu,rouge", "step": 44051})
l_step.append({"Bleu_1": 0.8408, "Bleu_2": 0.6918, "Bleu_3": 0.5438, "Bleu_4": 0.4181, "CIDEr": 1.3812, "METEOR": 0.3019, "ROUGE_L": 0.6079, "SPICE": 0.2427, "epoch": 0, "reinforce": "bleu,rouge", "step": 50344})
l_step.append({"Bleu_1": 0.8417, "Bleu_2": 0.6923, "Bleu_3": 0.5433, "Bleu_4": 0.4172, "CIDEr": 1.3807, "METEOR": 0.3022, "ROUGE_L": 0.6084, "SPICE": 0.2429, "epoch": 0, "reinforce": "bleu,rouge", "step": 56637})
for i, l in enumerate(l_step):
    l['lambda'] = 1 - i / len(l_step)

l_step_15 = []
l_step_15.append(
    {
        k: v for k, v in [
            ('CIDEr', 1.3965), ('Bleu_1', 0.8321), ('Bleu_2', 0.6864), ('Bleu_3', 0.5408),
            ('Bleu_4', 0.4168), ('ROUGE_L', 0.6059), ('SPICE', 0.2428), ('METEOR',
                                                                         0.3036), ("lambda", 1.0)
        ]
    }
)
# (pytorch) rame@hacienda:/data/rame/ExpansionNet_v2/github_ignore_material/saves/ftsbleu1rouge05bs18lr1e-5$ cat results.json
l_step_15.append({"Bleu_1": 0.8412, "Bleu_2": 0.6917, "Bleu_3": 0.5437, "Bleu_4": 0.4183, "CIDEr": 1.3888, "METEOR": 0.3016, "ROUGE_L": 0.6066, "SPICE": 0.2428, "epoch": 0, "reinforce": "bleu-1,rouge-0.5", "step": 6293})
l_step_15.append({"Bleu_1": 0.842, "Bleu_2": 0.6915, "Bleu_3": 0.5422, "Bleu_4": 0.4162, "CIDEr": 1.3825, "METEOR": 0.3014, "ROUGE_L": 0.6069, "SPICE": 0.2425, "epoch": 0, "reinforce": "bleu-1,rouge-0.5", "step": 12586})
l_step_15.append({"Bleu_1": 0.8438, "Bleu_2": 0.6933, "Bleu_3": 0.5435, "Bleu_4": 0.4169, "CIDEr": 1.3863, "METEOR": 0.3016, "ROUGE_L": 0.6079, "SPICE": 0.2431, "epoch": 0, "reinforce": "bleu-1,rouge-0.5", "step": 18879})
l_step_15.append({"Bleu_1": 0.8432, "Bleu_2": 0.6922, "Bleu_3": 0.5422, "Bleu_4": 0.4158, "CIDEr": 1.3828, "METEOR": 0.3021, "ROUGE_L": 0.6075, "SPICE": 0.2432, "epoch": 0, "reinforce": "bleu-1,rouge-0.5", "step": 25172})
l_step_15.append({"Bleu_1": 0.8434, "Bleu_2": 0.6921, "Bleu_3": 0.5424, "Bleu_4": 0.4159, "CIDEr": 1.3808, "METEOR": 0.3018, "ROUGE_L": 0.6077, "SPICE": 0.2432, "epoch": 0, "reinforce": "bleu-1,rouge-0.5", "step": 31465})
l_step_15.append({"Bleu_1": 0.8434, "Bleu_2": 0.6922, "Bleu_3": 0.5419, "Bleu_4": 0.4151, "CIDEr": 1.3797, "METEOR": 0.3017, "ROUGE_L": 0.6072, "SPICE": 0.2424, "epoch": 0, "reinforce": "bleu-1,rouge-0.5", "step": 37758})
l_step_15.append({"Bleu_1": 0.8444, "Bleu_2": 0.693, "Bleu_3": 0.5428, "Bleu_4": 0.4156, "CIDEr": 1.3817, "METEOR": 0.3017, "ROUGE_L": 0.6075, "SPICE": 0.2427, "epoch": 0, "reinforce": "bleu-1,rouge-0.5", "step": 44051})
l_step_15.append({"Bleu_1": 0.8445, "Bleu_2": 0.6931, "Bleu_3": 0.5434, "Bleu_4": 0.4167, "CIDEr": 1.382, "METEOR": 0.3017, "ROUGE_L": 0.6077, "SPICE": 0.2432, "epoch": 0, "reinforce": "bleu-1,rouge-0.5", "step": 50344})
l_step_15.append({"Bleu_1": 0.8449, "Bleu_2": 0.694, "Bleu_3": 0.5439, "Bleu_4": 0.4168, "CIDEr": 1.3826, "METEOR": 0.302, "ROUGE_L": 0.608, "SPICE": 0.243, "epoch": 0, "reinforce": "bleu-1,rouge-0.5", "step": 56637})
for i, l in enumerate(l_step):
    l['lambda'] = 1 - i / len(l_step)
