l_step = []

l_step.append(
    [
        ('CIDEr', 1.3965), ('Bleu_1', 0.8321), ('Bleu_2', 0.6864), ('Bleu_3', 0.5408),
        ('Bleu_4', 0.4168), ('ROUGE_L', 0.6059), ('SPICE', 0.2428), ('METEOR', 0.3036),
        ("lambda", 1.0)
    ]
)

l_step.append([('CIDEr', 1.3895), ('Bleu_1', 0.8337), ('Bleu_2', 0.6834), ('Bleu_3', 0.5361), ('Bleu_4', 0.4118), ('ROUGE_L', 0.6068), ('SPICE', 0.245), ('METEOR', 0.3046)])
l_step.append([('CIDEr', 1.3869), ('Bleu_1', 0.8346), ('Bleu_2', 0.684), ('Bleu_3', 0.5359), ('Bleu_4', 0.4102), ('ROUGE_L', 0.6083), ('SPICE', 0.2463), ('METEOR', 0.3057)])
l_step.append([('CIDEr', 1.3874), ('Bleu_1', 0.8353), ('Bleu_2', 0.6849), ('Bleu_3', 0.5369), ('Bleu_4', 0.412), ('ROUGE_L', 0.6091), ('SPICE', 0.246), ('METEOR', 0.306)])
l_step.append([('CIDEr', 1.3826), ('Bleu_1', 0.8341), ('Bleu_2', 0.6822), ('Bleu_3', 0.5336), ('Bleu_4', 0.4079), ('ROUGE_L', 0.6082), ('SPICE', 0.2469), ('METEOR', 0.3062)])
l_step.append([('CIDEr', 1.3837), ('Bleu_1', 0.8351), ('Bleu_2', 0.6843), ('Bleu_3', 0.5361), ('Bleu_4', 0.4106), ('ROUGE_L', 0.6078), ('SPICE', 0.2466), ('METEOR', 0.3062)])
l_step.append([('CIDEr', 1.3825), ('Bleu_1', 0.8356), ('Bleu_2', 0.6846), ('Bleu_3', 0.5362), ('Bleu_4', 0.4108), ('ROUGE_L', 0.6082), ('SPICE', 0.2467), ('METEOR', 0.3057)])
l_step.append([('CIDEr', 1.3805), ('Bleu_1', 0.8346), ('Bleu_2', 0.6829), ('Bleu_3', 0.5341), ('Bleu_4', 0.4085), ('ROUGE_L', 0.608), ('SPICE', 0.2469), ('METEOR', 0.3057)])
l_step.append([('CIDEr', 1.3821), ('Bleu_1', 0.8352), ('Bleu_2', 0.6834), ('Bleu_3', 0.5345), ('Bleu_4', 0.4086), ('ROUGE_L', 0.6084), ('SPICE', 0.2469), ('METEOR', 0.306)])
l_step.append([('CIDEr', 1.3832), ('Bleu_1', 0.8348), ('Bleu_2', 0.6833), ('Bleu_3', 0.5349), ('Bleu_4', 0.4096), ('ROUGE_L', 0.6085), ('SPICE', 0.2469), ('METEOR', 0.3063)])
l_step.append([('CIDEr', 1.3788), ('Bleu_1', 0.8342), ('Bleu_2', 0.6823), ('Bleu_3', 0.5335), ('Bleu_4', 0.408), ('ROUGE_L', 0.6081), ('SPICE', 0.2464), ('METEOR', 0.3063)])
l_step.append([('CIDEr', 1.3768), ('Bleu_1', 0.8349), ('Bleu_2', 0.6829), ('Bleu_3', 0.5337), ('Bleu_4', 0.4077), ('ROUGE_L', 0.6076), ('SPICE', 0.2464), ('METEOR', 0.3058)])
l_step.append([('CIDEr', 1.3784), ('Bleu_1', 0.8346), ('Bleu_2', 0.6826), ('Bleu_3', 0.5338), ('Bleu_4', 0.4082), ('ROUGE_L', 0.6074), ('SPICE', 0.2464), ('METEOR', 0.3056)])
l_step.append([('CIDEr', 1.3789), ('Bleu_1', 0.8349), ('Bleu_2', 0.6828), ('Bleu_3', 0.534), ('Bleu_4', 0.4081), ('ROUGE_L', 0.6076), ('SPICE', 0.2464), ('METEOR', 0.3058)])
l_step.append([('CIDEr', 1.3792), ('Bleu_1', 0.8349), ('Bleu_2', 0.6829), ('Bleu_3', 0.5341), ('Bleu_4', 0.4083), ('ROUGE_L', 0.6079), ('SPICE', 0.2464), ('METEOR', 0.3058)])


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

l_wa.append([('CIDEr', 0.6485), ('Bleu_1', 0.3892), ('Bleu_2', 0.2956), ('Bleu_3', 0.2164), ('Bleu_4', 0.1551), ('ROUGE_L', 0.4803), ('SPICE', 0.2496), ('METEOR', 0.2725), ('lambda', [2.0, -1.0])])
l_wa.append([('CIDEr', 1.0931), ('Bleu_1', 0.6827), ('Bleu_2', 0.5348), ('Bleu_3', 0.4037), ('Bleu_4', 0.2985), ('ROUGE_L', 0.5732), ('SPICE', 0.2537), ('METEOR', 0.3093), ('lambda', [1.5, -0.5])])
l_wa.append([('CIDEr', 1.156), ('Bleu_1', 0.7166), ('Bleu_2', 0.5645), ('Bleu_3', 0.4279), ('Bleu_4', 0.3176), ('ROUGE_L', 0.5816), ('SPICE', 0.2541), ('METEOR', 0.311), ('lambda', [1.4, -0.4])])
l_wa.append([('CIDEr', 1.2086), ('Bleu_1', 0.7416), ('Bleu_2', 0.588), ('Bleu_3', 0.4484), ('Bleu_4', 0.3347), ('ROUGE_L', 0.5881), ('SPICE', 0.2541), ('METEOR', 0.3122), ('lambda', [1.3, -0.3])])
l_wa.append([('CIDEr', 1.2571), ('Bleu_1', 0.7595), ('Bleu_2', 0.6059), ('Bleu_3', 0.4645), ('Bleu_4', 0.3487), ('ROUGE_L', 0.5934), ('SPICE', 0.2545), ('METEOR', 0.3128), ('lambda', [1.2, -0.2])])
l_wa.append([('CIDEr', 1.2939), ('Bleu_1', 0.7735), ('Bleu_2', 0.6199), ('Bleu_3', 0.4774), ('Bleu_4', 0.3599), ('ROUGE_L', 0.597), ('SPICE', 0.2536), ('METEOR', 0.3123), ('lambda', [1.1, -0.1])])
l_wa.append([('CIDEr', 1.3168), ('Bleu_1', 0.7838), ('Bleu_2', 0.6298), ('Bleu_3', 0.4861), ('Bleu_4', 0.3674), ('ROUGE_L', 0.5994), ('SPICE', 0.2532), ('METEOR', 0.3117), ('lambda', [1.0, 0.0])])
l_wa.append([('CIDEr', 1.3381), ('Bleu_1', 0.7938), ('Bleu_2', 0.6399), ('Bleu_3', 0.4956), ('Bleu_4', 0.3758), ('ROUGE_L', 0.6011), ('SPICE', 0.2521), ('METEOR', 0.3106), ('lambda', [0.9, 0.1])])
l_wa.append([('CIDEr', 1.3523), ('Bleu_1', 0.8024), ('Bleu_2', 0.6485), ('Bleu_3', 0.5033), ('Bleu_4', 0.3826), ('ROUGE_L', 0.6034), ('SPICE', 0.2512), ('METEOR', 0.3097), ('lambda', [0.8, 0.2])])
l_wa.append([('CIDEr', 1.3623), ('Bleu_1', 0.8087), ('Bleu_2', 0.6544), ('Bleu_3', 0.5081), ('Bleu_4', 0.3862), ('ROUGE_L', 0.6042), ('SPICE', 0.2498), ('METEOR', 0.3082), ('lambda', [0.7, 0.3])])
l_wa.append([('CIDEr', 1.3683), ('Bleu_1', 0.8133), ('Bleu_2', 0.6597), ('Bleu_3', 0.5125), ('Bleu_4', 0.3898), ('ROUGE_L', 0.6043), ('SPICE', 0.2483), ('METEOR', 0.3066), ('lambda', [0.6, 0.4])])
l_wa.append([('CIDEr', 1.3755), ('Bleu_1', 0.8293), ('Bleu_2', 0.6747), ('Bleu_3', 0.5251), ('Bleu_4', 0.4001), ('ROUGE_L', 0.6064), ('SPICE', 0.2475), ('METEOR', 0.3064), ('lambda', [0.5, 0.5])])
l_wa.append([('CIDEr', 1.379), ('Bleu_1', 0.8344), ('Bleu_2', 0.6805), ('Bleu_3', 0.5304), ('Bleu_4', 0.4044), ('ROUGE_L', 0.6067), ('SPICE', 0.2466), ('METEOR', 0.3053), ('lambda', [0.4, 0.6])])
l_wa.append([('CIDEr', 1.3797), ('Bleu_1', 0.8392), ('Bleu_2', 0.6849), ('Bleu_3', 0.534), ('Bleu_4', 0.4069), ('ROUGE_L', 0.6061), ('SPICE', 0.2456), ('METEOR', 0.304), ('lambda', [0.30000000000000004, 0.7])])
l_wa.append([('CIDEr', 1.3783), ('Bleu_1', 0.8438), ('Bleu_2', 0.6885), ('Bleu_3', 0.5361), ('Bleu_4', 0.4083), ('ROUGE_L', 0.6053), ('SPICE', 0.244), ('METEOR', 0.3025), ('lambda', [0.19999999999999996, 0.8])])
l_wa.append([('CIDEr', 1.3761), ('Bleu_1', 0.8468), ('Bleu_2', 0.6909), ('Bleu_3', 0.5373), ('Bleu_4', 0.4084), ('ROUGE_L', 0.6044), ('SPICE', 0.2428), ('METEOR', 0.3009), ('lambda', [0.09999999999999998, 0.9])])
l_wa.append([('CIDEr', 1.3707), ('Bleu_1', 0.8497), ('Bleu_2', 0.6931), ('Bleu_3', 0.538), ('Bleu_4', 0.4086), ('ROUGE_L', 0.6038), ('SPICE', 0.2413), ('METEOR', 0.299), ('lambda', [0.0, 1.0])])
l_wa.append([('CIDEr', 1.3655), ('Bleu_1', 0.8505), ('Bleu_2', 0.6934), ('Bleu_3', 0.5377), ('Bleu_4', 0.4078), ('ROUGE_L', 0.6027), ('SPICE', 0.2402), ('METEOR', 0.2971), ('lambda', [-0.10000000000000009, 1.1])])
l_wa.append([('CIDEr', 1.3591), ('Bleu_1', 0.851), ('Bleu_2', 0.6933), ('Bleu_3', 0.5364), ('Bleu_4', 0.406), ('ROUGE_L', 0.6008), ('SPICE', 0.2389), ('METEOR', 0.295), ('lambda', [-0.19999999999999996, 1.2])])
l_wa.append([('CIDEr', 1.3495), ('Bleu_1', 0.8509), ('Bleu_2', 0.6919), ('Bleu_3', 0.5336), ('Bleu_4', 0.4027), ('ROUGE_L', 0.5993), ('SPICE', 0.2368), ('METEOR', 0.2922), ('lambda', [-0.30000000000000004, 1.3])])
l_wa.append([('CIDEr', 1.3424), ('Bleu_1', 0.8512), ('Bleu_2', 0.6915), ('Bleu_3', 0.5321), ('Bleu_4', 0.4004), ('ROUGE_L', 0.5988), ('SPICE', 0.2343), ('METEOR', 0.2902), ('lambda', [-0.3999999999999999, 1.4])])
l_wa.append([('CIDEr', 1.3343), ('Bleu_1', 0.8507), ('Bleu_2', 0.6904), ('Bleu_3', 0.5295), ('Bleu_4', 0.3971), ('ROUGE_L', 0.5976), ('SPICE', 0.2317), ('METEOR', 0.2882), ('lambda', [-0.5, 1.5])])
# l_wa.append([('CIDEr', 1.2477), ('Bleu_1', 0.8307), ('Bleu_2', 0.6649), ('Bleu_3', 0.5002), ('Bleu_4', 0.3688), ('ROUGE_L', 0.5819), ('SPICE', 0.2141), ('METEOR', 0.2704), ('lambda', [-1.0, 2.0])])


def todict(l):
    d = {k: v for k, v in l}
    d["lambdas"] = d["lambda"]
    assert 0.9999 < d["lambda"][0] + d["lambda"][1] < 1.001
    d["lambda"] = d["lambda"][1]
    return d


l_wa = [todict(l) for l in l_wa]
l_wa = sorted(l_wa, key=lambda d: d["lambda"], reverse=False)
