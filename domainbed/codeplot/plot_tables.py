import numpy as np
def clean(l, skip=0, std=0):
    dwa = [float(x.split("/")[0].split("$\pm$")[std].split("&")[0].replace("\\underline{", "").replace("\\textbf{", "").replace("}", "")) for x in l.split("&")[skip:]]
    return dwa

def mean(l, factor=1):
    return np.mean([factor * line for line in l])

def mc(l, factor=1):
    return mean(clean(l), factor=factor)

def mcs(l, factor=1):
    return "{:.1f}".format(mean(clean(l), factor=factor)) + " $\pm$ " + "{:.1f}".format(mean(clean(l, std=1), factor=factor))

def format_val(x, e, add_std=True, prec=1):
    if np.issubdtype(type(x), np.floating):
        if prec == 2:
            x = "{:.2f}".format(x)
        else:
            x = "{:.1f}".format(x)
    if np.issubdtype(type(e), np.floating):
        e = "{:.1f}".format(e)
    if add_std:
        return str(x) + " $\pm$ " + str(e)
    else:
        return str(x)

def mcf(l, factor=1, prec=1, add_std=True, str_join=" & "):
    clean_l = [factor * ll for ll in clean(l)]
    clean_l.append(mean(clean_l))

    if add_std:
        clean_err = [factor * ll for ll in clean(l, std=1)]
        clean_err.append(mean(clean_err))
        return str_join.join(
            [format_val(r, e, prec=prec, add_std=add_std) for r, e in zip(clean_l, clean_err)]
        )
    else:
        return str_join.join(
            [format_val(r, 0, prec=prec, add_std=add_std) for r in clean_l]
        )

def av(l, skip=1, add_m=False, factor=1, prec=1, add_std=True, str_join=" & "):
    l = l.replace("\\", "").strip()
    clean_l = [clean(ll.strip(), skip=skip) for ll in l.split("\n")]
    if add_m:
        for i in range(len(clean_l)):
            clean_l[i].append(mean(clean_l[i]))
        #print(clean_l)
    clean_l = [[factor * ll for ll in l] for l in clean_l]
    row = [mean(ll) for ll in zip(*clean_l)]

    err = [np.std(list(ll) / np.sqrt(len(ll))) for ll in zip(*clean_l)]
    #print(str_join.join([format_val(r, e) for r, e in zip(row, err)]), "\\\\")
    return str_join.join([format_val(r, e, prec=prec, add_std=add_std) for r, e in zip(row, err)])
