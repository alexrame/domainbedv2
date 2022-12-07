import numpy as np
def clean(l, skip=0, std=0):
    dwa = [
        float(
            x.split("/")[0].split("$\pm$")[std].split("$pm$")[0].split("&")
            [0].replace("\\underline{", "").replace("\\textbf{", "").replace("}", "")
        ) for x in l.split("&")[skip:]
    ]
    return dwa

def mean(l, factor=1):
    return np.mean([factor * line for line in l])

def mc(l, factor=1):
    return mean(clean(l), factor=factor)

def mcs(l, factor=1):
    return "{:.1f}".format(mean(clean(l), factor=factor)) + " $\pm$ " + "{:.1f}".format(mean(clean(l, std=1), factor=factor))

def format_val(x, e, add_std=True, prec=1):
    if np.issubdtype(type(x), np.floating):
        if prec == 5:
            x = "{:.5f}".format(x)
        elif prec == 2:
            x = "{:.2f}".format(x)
        else:
            x = "{:.1f}".format(x)
    if np.issubdtype(type(e), np.floating):
        e = "{:.1f}".format(e)
    if add_std:
        return str(x) + " $\pm$ " + str(e)
    else:
        return str(x)


def mcf(l, factor=1, prec=1, add_std=True, str_join=" & ", title=False):
    if title:
        title = l[0]
        l = "&".join(l.split("&")[1:])
    clean_l = [factor * ll for ll in clean(l)]
    clean_l.append(mean(clean_l))

    if add_std:
        clean_err = [factor * ll for ll in clean(l, std=1)]
        clean_err.append(mean(clean_err))
        out = str_join.join(
            [format_val(r, e, prec=prec, add_std=add_std) for r, e in zip(clean_l, clean_err)]
        )
    else:
        out = str_join.join(
            [format_val(r, 0, prec=prec, add_std=add_std) for r in clean_l]
        )
    if title:
        return title + " & " + out
    return out

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

def test():
    # "%load_ext autoreload\n",
    # "%autoreload 2\n",
    # "import sys\n",
    # "import collections\n",
    # "import numpy as np\n",
    # "sys.path.append(\"/private/home/alexandrerame/domainbedv2/\")\n",
    # "sys.path.append(\"/private/home/alexandrerame/slurmconfig/notebook/data\")\n",
    # "from domainbed.codeplot import plot_tables
    # " mcf = plot_tables.mcf\n"
    ls = """0.9051513671875 0.0012088454027473186
    0.899169921875 0.0014333085582868973
    0.8437286689419796 0.001919163086421909
    0.8420861774744027 0.001296991477266239
    new
    0.9860179640718562 0.0006306406098993785
    0.9864970059880239 0.00041594143680988543
    0.8504963094935096 0.0019093916348700624
    0.8467930771188599 0.0021235964188514663
    """
    for l in ls.split("new"):
        clean_l = [plot_tables.clean(ll)[0] for ll in l.split()]
        acc_std = " & ".join([str(acc) + "$\pm$" + str(std) for acc,std in zip(clean_l[::2], clean_l[1::2])])
        print(mcf(acc_std, add_std=True, factor=100, title=False))
