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
        ('CIDEr', 1.3638), ('Bleu_1', 0.8442), ('Bleu_2', 0.6893), ('Bleu_3', 0.5349),
        ('Bleu_4', 0.4063), ('ROUGE_L', 0.6026), ('SPICE', 0.2399), ('METEOR', 0.2974),
        ("lambda", 0.75)
    ]
)
l_step.append(
    [
        ('CIDEr', 1.3555), ('Bleu_1', 0.847), ('Bleu_2', 0.6888), ('Bleu_3', 0.533),
        ('Bleu_4', 0.4035), ('ROUGE_L', 0.6003), ('SPICE', 0.2399), ('METEOR', 0.2958),
        ("lambda", 0.5)
    ]
)
l_step.append(
    [
        ('CIDEr', 1.3503), ('Bleu_1', 0.8556), ('Bleu_2', 0.6928), ('Bleu_3', 0.5325),
        ('Bleu_4', 0.4007), ('ROUGE_L', 0.5997), ('SPICE', 0.2327), ('METEOR', 0.2928),
        ("lambda", 0.25)
    ]
)
l_step.append(
    [
        ('CIDEr', 1.3389), ('Bleu_1', 0.8572), ('Bleu_2', 0.6917), ('Bleu_3', 0.5305),
        ('Bleu_4', 0.3982), ('ROUGE_L', 0.5972), ('SPICE', 0.2315), ('METEOR', 0.2913),
        ("lambda", 0.)
    ]
)
l_step = [todict(l) for l in l_step]

l_ens = []
l_ens.append(
    [
        ('CIDEr', 1.3965), ('Bleu_1', 0.8321), ('Bleu_2', 0.6864), ('Bleu_3', 0.5408),
        ('Bleu_4', 0.4168), ('ROUGE_L', 0.6059), ('SPICE', 0.2428), ('METEOR', 0.3036),
        ("lambda", 1.0)
    ]
)
l_ens.append(
    [
        ('CIDEr', 1.3936), ('Bleu_1', 0.8462), ('Bleu_2', 0.6963), ('Bleu_3', 0.5468),
        ('Bleu_4', 0.42), ('ROUGE_L', 0.6081), ('SPICE', 0.2421), ('METEOR', 0.3019),
        ("lambda", 0.5)
    ]
)

l_ens.append(
    [
        ('CIDEr', 1.3389), ('Bleu_1', 0.8572), ('Bleu_2', 0.6917), ('Bleu_3', 0.5305),
        ('Bleu_4', 0.3982), ('ROUGE_L', 0.5972), ('SPICE', 0.2315), ('METEOR', 0.2913),
        ("lambda", 0.)
    ]
)
l_ens = [todict(l) for l in l_ens]

l_wa = []

l_wa.append(
    [
        ('CIDEr', 1.3965), ('Bleu_1', 0.8321), ('Bleu_2', 0.6864), ('Bleu_3', 0.5408),
        ('Bleu_4', 0.4168), ('ROUGE_L', 0.6059), ('SPICE', 0.2428), ('METEOR', 0.3036),
        ("lambda", 1.0)
    ]
)

l_wa.append(
    [
        ('CIDEr', 1.392), ('Bleu_1', 0.8326), ('Bleu_2', 0.6856), ('Bleu_3', 0.5396),
        ('Bleu_4', 0.4159), ('ROUGE_L', 0.6056), ('SPICE', 0.2427), ('METEOR', 0.3032),
        ("lambda", 0.95)
    ]
)

l_wa.append(
    [
        ('CIDEr', 1.3933), ('Bleu_1', 0.8336), ('Bleu_2', 0.6864), ('Bleu_3', 0.5398),
        ('Bleu_4', 0.4155), ('ROUGE_L', 0.6059), ('SPICE', 0.2423), ('METEOR', 0.3031),
        ("lambda", 0.9)
    ]
)
l_wa.append(
    [
        ('CIDEr', 1.3939), ('Bleu_1', 0.8408), ('Bleu_2', 0.692), ('Bleu_3', 0.5434),
        ('Bleu_4', 0.4171), ('ROUGE_L', 0.6069), ('SPICE', 0.2434), ('METEOR', 0.3032),
        ("lambda", 0.7)
    ]
)

l_wa.append(
    [
        ('CIDEr', 1.3874), ('Bleu_1', 0.845), ('Bleu_2', 0.6939), ('Bleu_3', 0.5429),
        ('Bleu_4', 0.4149), ('ROUGE_L', 0.6055), ('SPICE', 0.2436), ('METEOR', 0.3015),
        ("lambda", 0.5)
    ]
)

l_wa.append(
    [
        ('CIDEr', 1.3756), ('Bleu_1', 0.8532), ('Bleu_2', 0.697), ('Bleu_3', 0.5421),
        ('Bleu_4', 0.4122), ('ROUGE_L', 0.604), ('SPICE', 0.2407),
        ('METEOR', 0.2987), ("lambda", 0.3)
    ]
)

l_wa.append(
    [
        ('CIDEr', 1.3656), ('Bleu_1', 0.8551), ('Bleu_2', 0.6961), ('Bleu_3', 0.5393),
        ('Bleu_4', 0.4085), ('ROUGE_L', 0.602), ('SPICE', 0.2384), ('METEOR', 0.2965),
        ("lambda", 0.2)
    ]
)

l_wa.append(
    [
        ('CIDEr', 1.3526), ('Bleu_1', 0.8553), ('Bleu_2', 0.6929), ('Bleu_3', 0.5336),
        ('Bleu_4', 0.4017), ('ROUGE_L', 0.599), ('SPICE', 0.2347), ('METEOR', 0.2937),
        ("lambda", 0.1)
    ]
)

l_wa.append(
    [
        ('CIDEr', 1.3444), ('Bleu_1', 0.8558), ('Bleu_2', 0.6918), ('Bleu_3', 0.5315),
        ('Bleu_4', 0.3998), ('ROUGE_L', 0.5968), ('SPICE', 0.2327), ('METEOR', 0.2923),
        ("lambda", 0.05)
    ]
)

l_wa.append(
    [
        ('CIDEr', 1.3389), ('Bleu_1', 0.8572), ('Bleu_2', 0.6917), ('Bleu_3', 0.5305),
        ('Bleu_4', 0.3982), ('ROUGE_L', 0.5972), ('SPICE', 0.2315), ('METEOR', 0.2913),
        ("lambda", 0.)
    ]
)

l_wa = [todict(l) for l in l_wa]
