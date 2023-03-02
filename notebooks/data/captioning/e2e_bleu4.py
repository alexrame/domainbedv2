def todict(l):
    return {k: v for k, v in l}


l_step4 = []
l_step4.append(
    [
        ('CIDEr', 1.3965), ('Bleu_1', 0.8321), ('Bleu_2', 0.6864), ('Bleu_3', 0.5408),
        ('Bleu_4', 0.4168), ('ROUGE_L', 0.6059), ('SPICE', 0.2428), ('METEOR', 0.3036),
        ("lambda", 1.0)
    ]
)
l_step4.append(
    [
        ('CIDEr', 1.3591), ('Bleu_1', 0.8244), ('Bleu_2', 0.6813), ('Bleu_3', 0.5399),
        ('Bleu_4', 0.4183), ('ROUGE_L', 0.6052), ('SPICE', 0.2384), ('METEOR', 0.3)
    ]
)
l_step4 = [todict(l) for l in l_step4]
