def merge(*ll):
    return [[y for x in l for y in x] for l in zip(*ll)]

import numpy as np
def clean(l):
    dwa = [float(x.split("/")[0].split("$")[0]) for x in l.split("&")]
    return dwa

def average(l):
    l = l.replace("\\", "").strip()
    clean_l = [clean(ll.strip()) for ll in l.split("\n")]
    row = [mean(ll) for ll in zip(*clean_l)]
    def format_val(x, e):
        if np.issubdtype(type(x), np.floating):
            x = "{:.1f}".format(x)
        if np.issubdtype(type(e), np.floating):
            e = "{:.1f}".format(e)
        return str(x) + " $\\pm$ " + str(e)
    err = [np.std(list(ll) / np.sqrt(len(ll))) for ll in zip(*clean_l)]
    print(" & ".join([format_val(r, e) for r, e in zip(row, err)]), "\\\\")



def get_x(l, key):
    if key in l[0]:
        return [(i[key] if key in i else -1000) for i in l if check_condition(i)]
    if key.replace('.', '', 1).isnumeric():
        return [float(key) for i in l]
    if key == "":
        return [0 for i in l]
    if "|" in key:
        if key.split("|")[0] == "abs":
            return [abs(i) for i in get_x(l, key.split("|")[1])]
        raise ValueError("Unknown operator")
    if "%" in key:
        return [(i - j)/j for i, j in zip(get_x(l, key.split("%")[0]), get_x(l, "%".join(key.split("%")[1:])))]
    if "/" in key:
        return [i/j for i, j in zip(get_x(l, key.split("/")[0]), get_x(l, "/".join(key.split("/")[1:])))]
    if "-" in key:
        return [i - j for i, j in zip(get_x(l, key.split("-")[0]), get_x(l, "-".join(key.split("-")[1:])))]
    if "+" in key:
        return [i + j for i, j in zip(get_x(l, key.split("+")[0]), get_x(l, "+".join(key.split("+")[1:])))]
    print(key, "absent")
    return [(i[key] if key in i else -10) for i in l if check_condition(i)]


ENV = None
TOPK = None
DROP = None
STEP = None
THESS = False
EHESS = False
def check_condition(i):
    if (DROP is not None and i.get("drop", DROP) != DROP):
        return False
    if (STEP is not None and i.get("step", STEP) != STEP):
        return False
    if (TOPK is not None and i.get("topk", TOPK) != TOPK):
        return False
    if ENV is not None and i.get("env", ENV) != ENV:
        return False
    if THESS and "thess" not in i:
        return False
    if EHESS and "hess" not in i:
        return False
    return True


import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.ticker import StrMethodFormatter

def concat_lists(l):
    new_l = []
    for ll in l:
        new_l.extend(ll)
    return new_l

def plot_histogram(l, labels, key, limits={}, lambda_filtering=None, list_indexes=None, loc="upper right", size=None, bins=25):
    if list_indexes is not None:
        l = [l[i] for i in list_indexes]
        if labels is not None:
            labels = [labels[i] for i in list_indexes]
    plt.rcParams["figure.figsize"] = (5, 5)
    kwargs = dict(alpha=0.5, bins=bins, density=True, stacked=True)

    def check_line(line):
        if lambda_filtering is not None:
            return lambda_filtering(line)
        return True

    fig = plt.figure()
    data = []

    for c in l:
        data.append(get_x([line for line in c if check_line(line)], key))

    colors = cm.rainbow(np.linspace(0., 1, len(labels)))

    for i in range(len(labels)):
        plt.hist(data[i], **kwargs, color=colors[i], label=labels[i])

    plt.gca().set_xlabel(
        dict_key_to_label.get(key, key), fontsize=SIZE_AXIS)
    # "Prediction diversity",
    # dict_key_to_label.get(key, key),
    # fontsize=SIZE_AXIS)
    plt.gca().set_ylabel('Frequency (%)', fontsize=SIZE_AXIS)
    if key in limits:
        plt.xlim(limits[key][0], limits[key][1])
    if loc:
        plt.legend(loc=loc, fontsize=size or SIZE)
    return fig

from matplotlib import pyplot
from matplotlib.patches import Rectangle

def plot_histogram_keys(ll, list_keys, labels=None, limits={}, lambda_filtering=None, loc="upper right", size=None, title=None, ax1=None):

    if ax1 is None:
        fig, ax1 = plt.subplots()
    else:
        fig = None

    if FORMAT_X:
        assert FORMAT_X == 3
        ax1.xaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}')) # 1 decimal places

    kwargs = dict(alpha=0.5, bins=15, density=True, stacked=True)

    def check_line(line):
        if lambda_filtering is not None:
            return lambda_filtering(line)
        return True

    ax1.set_xlabel(dict_key_to_label.get(list_keys[0], list_keys[0]), fontsize=SIZE_AXIS)
    ax1.set_ylabel('Frequency (%)', fontsize=SIZE_AXIS)

    data = []
    for key in list_keys:
        sublines = [line for line in ll if key.split("-")[0] in line]
        data.append(get_x([line for line in sublines if check_line(line)], key))

    if labels is None:
        labels = list_keys
    colors = cm.rainbow(np.linspace(0., 1, len(labels)))
    plot_lines_1 = []
    for i in range(len(labels)):
        hist_i = ax1.hist(data[i], **kwargs, color=colors[i], label=labels[i])
        plot_lines_1.append(hist_i[-1])
    handles = [Rectangle((0, 0), 1, 1, color=c, alpha=0.5) for c in colors]
    first_legend = ax1.legend(
        handles=handles,
        labels=labels,
        loc=loc,
        fontsize=size or SIZE
    )
    ax1.set_xlabel(dict_key_to_label.get(key, key), fontsize=SIZE_AXIS)
    ax1.set_ylabel('Frequency (%)', fontsize=SIZE_AXIS)
    if key in limits:
        ax1.set_xlim(limits[key][0], limits[key][1])
    if "freq" in limits:
        ax1.set_ylim(limits["freq"][0], limits["freq"][1])
    if loc:
        ax1.add_artist(first_legend)
    if title:
        ax1.set_title(title, fontsize=SIZE)
    return fig


def plot_histogram_two(l, labels, key1, key2, limits={}, lambda_filtering=None, list_indexes=None, loc="upper right"):
    if list_indexes is not None:
        l = [l[i] for i in list_indexes]
        if labels is not None:
            labels = [labels[i] for i in list_indexes]
    plt.rcParams["figure.figsize"] = (5, 5)
    kwargs = dict(alpha=0.5, bins=15, density=True, stacked=True)

    def check_line(line):
        if lambda_filtering is not None:
            return lambda_filtering(line)
        return True

    fig = plt.figure()
    plt.gca().set_xlabel(dict_key_to_label.get(key2, key2), fontsize=SIZE_AXIS)
    plt.gca().set_ylabel('Frequency (\%)', fontsize=SIZE_AXIS)

    data = []
    for c in l:
        data.append(get_x([line for line in c if check_line(line)], key1))

    colors = cm.Blues(np.linspace(0.3, 1, len(labels)))
    plot_lines_1 = []
    for i in range(len(labels)):
        hist_i = plt.hist(data[i], **kwargs, color=colors[i], label=labels[i])
        plot_lines_1.append(hist_i[-1])
    handles = [Rectangle((0, 0), 1, 1, color=c, alpha=0.5) for c in colors]
    first_legend = plt.legend(
        handles=handles,
        labels=labels,
        title="Val ID:",
        loc='upper right',
        bbox_to_anchor=(1.0, 1.0),
        fontsize="small"
    )
    plt.gca().add_artist(first_legend)

    data = []
    for c in l:
        data.append(get_x([line for line in c if check_line(line)], key2))

    colors = cm.Reds(np.linspace(0.3, 1, len(labels)))
    plot_lines_2 = []
    for i in range(len(labels)):
        hist_i = plt.hist(data[i], **kwargs, color=colors[i], label=labels[i])
        plot_lines_2.append(hist_i[-1])
    handles = [Rectangle((0, 0), 1, 1, color=c, alpha=0.5) for c in colors]
    plt.legend(
        handles=handles,
        labels=labels,
        title="Test OOD:",
        loc='upper right',
        bbox_to_anchor=(1.0, 0.75),
        fontsize="small"
    )

    return fig



# dict_key_to_label = {
#     "length": "M (number of networks)",
#     "soupswa": "Acc. sw",
#     "thess": "Train Flatness",
#     "soup-netm": "$Acc(\\frac{1}{M}(\\sum \\theta_m)) - \\frac{1}{M}(\\sum Acc(\\theta_m))$",
#     "df": "Feature diversity ",
#     "dr": "Prediction diversity ",
#     "hess": "Flatness",
#     "netm": "$\\frac{1}{M}(\\sum Acc(\\theta_m))$",
#     "soup": "$Acc(\\frac{1}{M}(\\sum \\theta_m))$",
#     "net": "$Acc(\\{\\theta_m)\\})$"
# }

dict_key_to_label = {
    "length": "$M$",
    "robust": "Robust Coeff",
    "step": "# steps",
    "acc": "$Acc(WA)$",
    # "acc": "OOD test acc.",
    "test_acc": "OOD test acc.",
    "acc_cla": "OOD test acc.",
    "dirslen": "\# auxiliary tasks",
    "lengthf": "\# training runs",
    "length": "# training runs",
    "testin_acc": "OOD train acc.",
    "env_1_out_acc+env_2_out_acc+env_3_out_acc/3": "ID val acc.",
    "train_acc": "ID val acc.",
    "val_acc": "ID val acc.",
    # "soupswa": "Acc. sw",
    # "thess": "Train Flatness",
    # "soup-netm": "$Acc(\\theta_{WA}) - \\frac{1}{M}(\\sum Acc(\\theta_m))$",
    "soup-netm":
    '$Acc(\\frac{\\theta_{1} + \\theta_{2}}{2}) - \\frac{Acc(\\theta_{1}) + Acc(\\theta_{2})}{2}$',
    "lr2-lr1": "Difference in learning rates",
    "acc-acc_netm": "OOD test acc. gain",
    "train_acc-train_acc_netm": "ID val acc. gain",

    "divf_netm": "Feature diversity",
    "dist_lambdas": "$|\kappa_1 - \kappa_0|$",
    "acc-acc_ens":  "Accuracy gain of WA over ENS",
    "divr_netm": "Prediction r-diversity",
    "divr_net": "Prediction r-diversity",
    "divd_netm": "Prediction d-diversity",
    "divp_netm": "Prediction p-diversity",
    "1-divq_net": "Prediction q-diversity",
    "1-divq_netm": "Prediction q-diversity",
    "1-train_divq_netm": "ID val prediction q-diversity",
    "divq_netm": "Prediction similarity",
    # "hess": "Flatness",
    # "acc_netm": "$\\frac{1}{M}(\\sum Acc(\\theta_m))$",
    "acc_netm": "Individual acc.",
    "soup": "$Acc(\\theta_{WA})$",
    # "net": "$Acc(\\{\\theta_m\\}_1^M)$"
    "acc_ens": "$Acc(ENS)$",
    "weighting": '$\lambda$',
    "lambda": '$\lambda$',
}


def plot_slopes_c():
    fig = plt.figure()
    dr = [96, 177, 202, 207, 236, 258, 303, 312]
    dr = [d / 1000 for d in dr]
    df = [52, 105, 135, 134, 158, 158, 203, 229]
    df = [d / 1000 for d in df]
    m = list(range(2, 10))
    plt.scatter(m, dr, label="Prediction diversity", color="blue")
    fit_and_plot_with_value(m, dr, order="log", label=None, color="blue", ax=None)
    plt.scatter(m, df, label="Feature diversity", color="red")
    fit_and_plot_with_value(m, df, order="log", label=None, color="red", ax=None)
    plt.xlabel("M (number of networks)", fontsize=SIZE)
    plt.ylabel(r"Slope", fontsize=SIZE)
    plt.legend(fontsize=SIZE)
    return fig


def fit_and_plot_with_value(val1, val2, order, label, color, ax=None, linestyle="-"):

    # get_x1_sorted = sorted(val1)
    get_x1_sorted = np.linspace(min(val1), max(val1), 500000)

    if ax is None:
        ax = plt
    if order in [1, "1"]:
        m, b = np.polyfit(val1, val2, 1)
        linewidth = 1
        if MUL:
            labelplot = label + " (slope: " + "{:.0f}".format(m * MUL) + ")"
        elif label in ["No2", "No1"] and order == 1:
            labelplot = "y={:.3f}x+{:.3f}".format(m, b)
            linewidth = 5
            # color = "cyan"
        else:
            # pass
            labelplot = label
            # + " (slope: " + "{:.3f}".format(m) + ")"
        ax.plot(
            get_x1_sorted,
            m * np.array(get_x1_sorted) + b,
            color=color,
            label=labelplot,
            linestyle=linestyle,
            linewidth=linewidth
        )
    elif order in [2, "2"]:
        m2, m1, b = np.polyfit(val1, val2, 2)
        preds = m2 * np.array(get_x1_sorted)**2 + m1 * np.array(get_x1_sorted) + b

        ax.plot(
            get_x1_sorted, preds, linestyle=linestyle, color=color, linewidth=3,
            label=label
        )  # label="int."+label)
        # if label == "dnf_0":
        #     import pdb; pdb.set_trace()
    elif order in [3, "3"]:
        m3, m2, m1, b = np.polyfit(val1, val2, 3)

        preds = m3 * np.array(get_x1_sorted)**3 + m2 * np.array(get_x1_sorted)**2 + m1 * np.array(
            get_x1_sorted
        ) + b
        ax.plot(
            get_x1_sorted, preds, color=color, linestyle=linestyle, linewidth=3,
            label=label
        )  # label="int."+label)
    elif order in [4, "4"]:
        m4, m3, m2, m1, b = np.polyfit(val1, val2, 4)

        preds = m4 * np.array(get_x1_sorted)**4 + m3 * np.array(get_x1_sorted)**3 + m2 * np.array(get_x1_sorted)**2 + m1 * np.array(
            get_x1_sorted
        ) + b
        ax.plot(
            get_x1_sorted, preds, color=color, linestyle=linestyle, linewidth=3, label=label
        )  # label="int."+label)
    elif order in [5, "5"]:
        m5, m4, m3, m2, m1, b = np.polyfit(val1, val2, 5)

        preds = m5 * np.array(get_x1_sorted)**5 + m4 * np.array(get_x1_sorted)**4 + m3 * np.array(get_x1_sorted)**3 + m2 * np.array(get_x1_sorted)**2 + m1 * np.array(
            get_x1_sorted
        ) + b
        ax.plot(
            get_x1_sorted, preds, color=color, linestyle=linestyle, linewidth=3, label=label
        )  # label="int."+label)
    elif order == "log":
        m1, b = np.polyfit(np.log(val1), val2, 1)
        log_get_x1_sorted = np.log(get_x1_sorted)
        preds = m1 * np.array(log_get_x1_sorted) + b
        ax.plot(get_x1_sorted, preds, color=color, label=label, linestyle=linestyle)  # label="int."+label)
    elif order == "2log":
        m2, m1, b = np.polyfit(np.log(val1), val2, 2)
        log_get_x1_sorted = np.log(get_x1_sorted)
        preds = m2 * np.array(log_get_x1_sorted)**2 + m1 * np.array(log_get_x1_sorted) + b

        ax.plot(get_x1_sorted, preds, color=color, label=label, linestyle=linestyle)  # label="int."+label)
    elif order == "3log":
        m3, m2, m1, b = np.polyfit(np.log(val1), val2, 3)
        log_get_x1_sorted = np.log(get_x1_sorted)
        preds = m3 * np.array(log_get_x1_sorted)**3 + m2 * np.array(log_get_x1_sorted)**2 + m1 * np.array(
            log_get_x1_sorted
        ) + b
        ax.plot(get_x1_sorted, preds, color=color, label=label, linestyle=linestyle)  # label="int."+label)
    elif order in [0, -1, None, "", "0"]:
        return
    else:
        raise ValueError(order)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (6, 6)
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times Roman'

MUL = 0


def fit_and_plot(key1, key2, l, order, label, color, ax=None, linestyle="-"):
    return fit_and_plot_with_value(
        val1=get_x(l, key1),
        val2=get_x(l, key2),
        order=order,
        label=label,
        color=color,
        ax=ax,
        linestyle=linestyle
    )

SIZE="large"
SIZE_AXIS="xx-large"


def plot_basic_scatter(list_dict_values, key_x, keys_y, _dict_key_to_label="def", colors=None, keycolor=None, order=0, linestyle=None, keys_error=None, loc="upper right", title=None, keyclustering=None,  kwargs={}, _dict_key_to_limit={}, ax1=None, lambda_filtering=None, markers=None):
    if ax1 is None:
        fig, ax1 = plt.subplots()
    else:
        fig = None
    if _dict_key_to_label == "def":
        _dict_key_to_label = dict_key_to_label
    else:
        _dict_key_to_label = {**dict_key_to_label, **_dict_key_to_label}

    if colors is None:
        if keycolor is not None:
            colors = ["Blues", "Reds", "Greens", "Oranges", "Greys", "Purples"][:len(keys_y)]
        else:
            colors = cm.rainbow(np.linspace(0, 1, len(keys_y)))


    if lambda_filtering is not None:
        list_dict_values = [lll for lll in list_dict_values if lambda_filtering(lll)]

    if keyclustering is not None:
        list_dict_values_means = lambda_clustering(list_dict_values, keyclustering)

    if key_x in _dict_key_to_limit:
        ax1.set_xlim(_dict_key_to_limit[key_x])
    if keys_y[0] in _dict_key_to_limit:
        ax1.set_ylim(_dict_key_to_limit[keys_y[0]])

    for i in range(len(keys_y)):
        key_y = keys_y[i]
        color = colors[i]
        label = _dict_key_to_label.get(key_y, key_y)
        label=(label if order != 1 else None)
        if markers is not None:
            marker = markers[i]
        else:
            marker = None
        if keys_error is not None:
            # plt.errorbar(
            #     x,
            #     y,
            #     get_x(list_dict_values, keys_error[i]),
            #     color=color,
            #     label=label)
            ax1.fill_between(
                get_x(list_dict_values_means, key_x),
                get_x(list_dict_values_means, key_y + "-" + keys_error[i]),
                get_x(list_dict_values_means, key_y + "+" + keys_error[i]),
                color=color,
                # label=label,
                **kwargs)

        if keycolor is not None:

            ax1.scatter(
                get_x(list_dict_values, key_x),
                get_x(list_dict_values, key_y),
                c=get_x(list_dict_values, keycolor),
                cmap=color,
                label=label,
                marker=marker,
                **kwargs)
            color = cm.get_cmap(color)(0.5)
        else:
            ax1.scatter(
                get_x(list_dict_values, key_x),
                get_x(list_dict_values, key_y),
                color=color,
                marker=marker,
                label=label
            )
        fit_and_plot(key_x, key_y, list_dict_values, order, label, color=color, linestyle=linestyle, ax=ax1)

    ax1.set_xlabel(_dict_key_to_label.get(key_x, key_x), fontsize=SIZE)
    ax1.set_ylabel("OOD Accuracy", fontsize=SIZE)
    if loc != "no":
        ax1.legend(loc=loc, fontsize=SIZE)
        # if keycolor is not None:
        #     legend = ax1.get_legend()
        #     for i, cmap_name in enumerate(colors[:len(keys_y)]):
        #         cmap = cm.get_cmap(cmap_name)
        #         legend.legendHandles[i].set_color(cmap(0.5))
    if title:
        ax1.set_title(title, fontsize=SIZE)
    return fig

import collections
def lambda_split(l, keysplit, lambda_filtering=None):
    dict_l = collections.defaultdict(list)

    for line, key in zip(l, get_x(l, keysplit)):
        line[keysplit] = key
        if lambda_filtering is None or lambda_filtering(line):
            dict_l[key].append(line)

    for key_uniq, list_dict_values in list(dict_l.items()):
        new_ls = [{} for _ in range(1)]
        for key in list_dict_values[0].keys():
            if isinstance(list_dict_values[0][key], str) or key in ["count", "testenv"]:
                # if "divd" in key or "divp" in key or "divf" in key or 'max' in key or "ens1h" in key or "length" in key or key in ["dirs", "count"]:
                for new_l in new_ls:
                    new_l[key] = list_dict_values[0][key]
            else:
                list_values = [line[key] for line in list_dict_values]
                # new_l[key + "_std"] = np.std(list_values)
                for new_l in new_ls:
                    new_l[key] = np.mean(list_values)# + np.random.normal(0, abs(np.random.normal(0, 0.003)))
        dict_l[key_uniq].extend(new_ls)
        # dict_l[key_uniq] = [new_l]
    new_l = [
        dict_l[key] for key in sorted(dict_l.keys(), reverse=True)
    ]
    return new_l

def lambda_clustering(l, keyclustering):
    dict_l = collections.defaultdict(list)

    for line, key in zip(l, get_x(l, keyclustering)):
        line[keyclustering] = key
        dict_l[key].append(line)

    new_l = []
    for key, list_dict_values in dict_l.items():
        new_l.append({})
        for key in list_dict_values[0].keys():
            if isinstance(list_dict_values[0][key], str) or key in ["count"]:
                # if "divd" in key or "divp" in key or "divf" in key or 'max' in key or "ens1h" in key or "length" in key or key in ["dirs", "count"]:
                continue
            list_values = [line[key] for line in list_dict_values]
            new_l[-1][key + "_std"] = np.std(list_values)
            new_l[-1][key] = np.mean(list_values)
            if key == "step":
                new_l[-1][key] = int(new_l[-1][key])
    return new_l

FORMAT_X=0
FORMAT_Y=0

from matplotlib.colors import ListedColormap,LinearSegmentedColormap

def create_colormaps():
    N = 256
    dict_colormaps = {}
    def create_cmp(r,g,b):
        np_array = np.ones((N, 4))
        np_array[:, 0] = np.linspace(r/256, 1, N)
        np_array[:, 1] = np.linspace(g/256, 1, N)
        np_array[:, 2] = np.linspace(b/256, 1, N)
        return ListedColormap(np_array)

    dict_colormaps["Yellows"] = create_cmp(255, 232, 11)
    dict_colormaps["Light_Yellows"] = create_cmp(255, 232, 200)
    dict_colormaps["Dark_Blues"] = create_cmp(2, 2, 200)
    dict_colormaps["Blues_Greys"] = create_cmp(40, 60, 80)
    dict_colormaps["Dark_Greys"] = create_cmp(40, 60, 200)

    return dict_colormaps

def add_arrow(line, position=None, direction='right', size=15, color=None):

    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )


def plot_key(
    l,
    key_x,
    key_y,
    key_y_2=None,
    keysplit=None,
    keycolor=None,
    keysize=None,
    order=1,
    label="",
    labels=None,
    diag=False,
    markers=None,
    colors=None,
    colormaps=None,
    linestyles=None,
    _dict_key_to_limit={},
    _dict_key_to_label="def",
    loc="upper right",
    legendtitle=None,
    lambda_filtering=None,
    keyclustering=None,
    list_indexes=None,
    keyerror=None,
    title=None,
    connect_points=False,
    connect_endpoints=False,
    kwargs={}
):
    if list_indexes is not None:
        l = [l[i] for i in list_indexes]
        if labels is not None:
            labels = [labels[i] for i in list_indexes]
        if colormaps is not None:
            colormaps = [colormaps[i] for i in list_indexes]
        if linestyles is not None:
            linestyles = [linestyles[i] for i in list_indexes]

    fig, ax1 = plt.subplots()

    if FORMAT_X:
        if FORMAT_X == 1:
            plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}')) # 2 decimal places
        elif isinstance(FORMAT_X, list):
            plt.gca().set_xticks(FORMAT_X)
    if FORMAT_Y:
        assert FORMAT_Y == 3
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}')) # 2 decimal places
    if _dict_key_to_label == "def":
        _dict_key_to_label = dict_key_to_label
    else:
        _dict_key_to_label = {**dict_key_to_label, **_dict_key_to_label}

    if keysplit is not None:
        assert len(l) == 1
        l = lambda_split(l[0], keysplit, lambda_filtering=lambda_filtering)
        if labels == "fromsplit":
            labels = [label.format(ll[0][keysplit]) for ll in l]


    if colormaps is None:
        if keycolor is not None:
            colormaps = [
                "Reds", "Blues", "Greens", "Oranges", "Greys", "Purples", "Reds", "Blues", "Greens",
                "Oranges", "Greys", "Purples"
            ][:len(l)]
    else:
        dict_colormaps = create_colormaps()
        colormaps = [dict_colormaps.get(cmp, cmp) for cmp in colormaps]

    if colors is None:
        if colormaps is not None:
            colors = [cm.get_cmap(cmp)(0.5) for cmp in colormaps]
        else:
            colors = cm.rainbow(np.linspace(0, 1, len(l)))

    if labels is None:
        if label.startswith("."):
            labels = [
                ".".join([str(ll[0][key]) for key in label.split(".")[1:]])
                for ll in l]
        else:
            labels = [label + str(i) for i in range(len(l))]

    plt.xlabel(_dict_key_to_label.get(key_x, key_x), fontsize=SIZE_AXIS)
    plt.ylabel(_dict_key_to_label.get(key_y, key_y), fontsize=SIZE_AXIS)
    if key_y_2:
        ax2 = ax1.twinx()
        ax2.set_ylabel(_dict_key_to_label.get(key_y_2, key_y_2), fontsize=SIZE_AXIS)


    def plot_with_int(ll, color, colormap, label, marker, linestyle, key_y, ax, kwargs):
        ll = [lll for lll in ll if lambda_filtering is None or lambda_filtering(lll)]
        t = get_x(ll, key_x)
        if t == []:
            return
        if keyclustering is not None:
            ll = lambda_clustering(ll, keyclustering)

        newlabel = label if (linestyle is None and not connect_points and label != "no") else None
        # if label !="no" else None#
        if keyerror is not None:
            ax.errorbar(
                get_x(ll, key_x),
                get_x(ll, key_y),
                get_x(ll, keyerror),
                color=color,
                label=newlabel,
                marker=marker,
                **kwargs
            )
            plt.fill_between(get_x(ll, key_x),
                get_x(ll, key_y + "-" + keyerror),
                get_x(ll, key_y + "+" + keyerror),
                alpha=0.5,
                )
            # plt.errorbar(
            #     get_x(ll, key_x),
            #     get_x(ll, key2),
            #     get_x(ll, keyerror),
            #     color=color,
            #     label=newlabel,
            #     marker=marker,
            #     **kwargs)
        else:
            if connect_points and linestyle is not None:
                line_plot = ax.plot(
                    get_x(ll, key_x),
                    get_x(ll, key_y),
                    label=label,
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                )
                # if len(ll) > 1:
                #     add_arrow(line_plot[0], color=color, size=30)

            if keycolor is not None:
                kwargs["c"] = [-x for x in get_x(ll, keycolor)]
                kwargs["cmap"] = colormap
            else:
                kwargs["color"] = color
            if keysize is not None:
                kwargs["s"] = [x for x in get_x(ll, keysize)]
                min_s = min(kwargs["s"])
                max_s = max(kwargs["s"])
                kwargs["s"] = [200 * (x-min_s) / (max_s-min_s) + 5 for x in kwargs["s"]]

            ax.scatter(
                get_x(ll, key_x),
                get_x(ll, key_y),
                label=newlabel,
                marker=marker,
                **kwargs
            )

        if not connect_points:
            fit_and_plot(
                key_x,
                key_y,
                ll,
                order,
                label if linestyle is not None else None,
                color,
                linestyle=linestyle
            )
        if connect_endpoints:
            ax.plot(
                    get_x([ll[0], ll[-1]], key_x),
                    get_x([ll[0], ll[-1]], key_y),
                    label=None,
                    color=color,
                    linestyle="--",
                )

    for index in range(len(l)):
        label = labels[index]
        label = labels[index]
        if linestyles is not None:
            linestyle = linestyles[index%len(linestyles)]
        else:
            linestyle = None
        if colormaps is not None:
            colormap = colormaps[index]
        else:
            colormap = None
        if markers is not None:
            marker = markers[index]
        else:
            dictlinestyle_to_marker = {
                None: "o",
                "solid": "o",
                "dashed": "+",
                "dotted": "*",
                "dashdot": "x",
            }
            marker = dictlinestyle_to_marker[linestyle]
        kwargs_copy = {k:v for k, v in kwargs.items()}
        plot_with_int(l[index], color=colors[index], colormap=colormap, label=label, marker=marker, linestyle=linestyle, key_y=key_y, ax=ax1, kwargs=kwargs_copy)
        if key_y_2:
            plot_with_int(l[index], color=colors[index], colormap=colormap, label=label, marker="*", linestyle="--", key_y=key_y_2, ax=ax2, kwargs=kwargs_copy)

    if diag:
        xpoints = ypoints = plt.xlim()
        ax1.plot(
            xpoints,
            ypoints,
            linestyle='--',
            color='k',
            lw=3,
            scalex=False,
            scaley=False,
            label="y=x"
        )

    if key_x in _dict_key_to_limit:
        ax1.set_xlim(_dict_key_to_limit[key_x])
    if key_y in _dict_key_to_limit:
        ax1.set_ylim(_dict_key_to_limit[key_y])
    if key_y_2 is not None and key_y_2 in _dict_key_to_limit:
        ax2.set_ylim(_dict_key_to_limit[key_y_2])
    if loc != "no":
        ax1.legend(title=legendtitle, loc=loc, fontsize=SIZE)
        # if keycolor is not None:
        #     legend = ax1.get_legend()
        #     for i, cmap_name in enumerate(colormaps[:len(l)]):
        #         cmap = cm.get_cmap(cmap_name)
        #         legend.legendHandles[i].set_color(cmap(0.5))
    if title:
        ax1.title(title, fontsize=SIZE)
    return fig


def plot_iter(key_x, key_y, order=1, dict_key_to_limit={}):
    fig = plt.figure()
    plt.xlabel(dict_key_to_label.get(key_x, key_x), fontsize=SIZE_AXIS)
    plt.ylabel(dict_key_to_label.get(key2, key2), fontsize=SIZE_AXIS)

    def plot_with_int(l, color, label):
        t = get_x(l, key_x)
        if t == []:
            return
        plt.scatter(
            get_x(l, key_x), get_x(l, key2), color=color, label=label if order != 1 else None
        )
        fit_and_plot(key_x, key2, l, order, label, color)

    plot_with_int(
        l2, color="yellow", label="SOUP: $\\{\\theta_m\\}_1^M$ from different runs (HP Standard)"
    )
    plot_with_int(
        leoa, color="grey", label="SOUP: $\\{\\theta_m\\}_1^M$ from different runs (HP=EoA)"
    )
    plot_with_int(l0, color="blue", label="SWA: $\\{\\theta_m\\}_1^M$ from same run")
    if key_x in dict_key_to_limit:
        plt.xlim(dict_key_to_limit[key_x])
    if key2 in dict_key_to_limit:
        plt.ylim(dict_key_to_limit[key2])
    plt.legend(fontsize=SIZE)
    return fig


def process_line(liter):
    for line in liter:
        line["out_acc_soup"] = np.mean(
            [value for key, value in line.items() if key.endswith("out_acc_soup")]
        )
        line["out_acc_soupswa"] = np.mean(
            [value for key, value in line.items() if key.endswith("out_acc_soupswa")]
        )
        line["out_ece_soup"] = np.mean(
            [value for key, value in line.items() if key.endswith("out_ece_soup")]
        )
        line["out_ece_soupswa"] = np.mean(
            [value for key, value in line.items() if key.endswith("out_ece_soupswa")]
        )


def plot_iter_soupacc(key_x, order=1, do_ens=False, do_soup=True, ood=False):
    if ood:
        dict_key_to_limit = {"soup": [0.610, 0.695]}
    else:
        dict_key_to_limit = {"soup": [0.832, 0.874]}

    fig = plt.figure()
    plt.xlabel(dict_key_to_label.get(key_x, key_x), fontsize=SIZE_AXIS)
    plt.ylabel(dict_key_to_label.get("soup", "soup"), fontsize=SIZE_AXIS)

    colors = cm.rainbow(np.linspace(0.2, 1, 3))

    def plot_with_int(l, color, label, key2, marker, linestyle):
        t = get_x(l, key_x)
        if t == []:
            return

        l = [ll for ll in l if key2 in ll]
        plt.scatter(
            get_x(l, key_x),
            get_x(l, key2),
            color=color,
            label=label if order != 1 else None,
            marker=marker
        )
        fit_and_plot(key_x, key2, l, order, label, color, linestyle=linestyle)

    if do_soup:
        plot_with_int(
            l_emvc,
            color=colors[2],
            label="DWA: $\\{\\theta_m\\}_1^M$ from 20 ERM, 20 Mixup and 20 Coral runs",
            key2="soup" if ood else "train_soup",
            marker=".",
            linestyle="-"
        )
    if do_ens:
        plot_with_int(
            l_emvc,
            color=colors[2],
            label="DE: $\\{\\theta_m\\}_1^M$ from 20 ERM, 20 Mixup and 20 Coral runs",
            key2="net" if ood else "train_net",
            marker="x",
            linestyle="--"
        )
    if do_soup:
        plot_with_int(
            l_erm,
            color=colors[1],
            label="DWA: $\\{\\theta_m\\}_1^M$ from 60 ERM runs",
            key2="soup" if ood else "train_soup",
            marker=".",
            linestyle="-"
        )
    if do_ens:
        plot_with_int(
            l_erm,
            color=colors[1],
            label="DE: $\\{\\theta_m\\}_1^M$ from 60 ERM runs",
            key2="net" if ood else "train_net",
            marker="x",
            linestyle="--"
        )

    if do_soup:
        plot_with_int(
            l_swa,
            color=colors[0],
            label="WA: $\\{\\theta_m\\}_1^M$ from a single ERM run",
            key2="soup" if ood else "train_soup",
            marker=".",
            linestyle="-"
        )
    if do_ens:
        plot_with_int(
            l_swa,
            color=colors[0],
            label="DE: $\\{\\theta_m\\}_1^M$ from a single ERM run",
            key2="net" if ood else "train_net",
            marker="x",
            linestyle="--"
        )
    #plot_with_int(liter_hpl.lswa, color=colors[0], label="Ens: $\\{\\theta_m\\}_1^M$ from a single ERM run", key2="net")

    if key_x in dict_key_to_limit:
        plt.xlim(dict_key_to_limit[key_x])
    if "soup" in dict_key_to_limit:
        plt.ylim(dict_key_to_limit["soup"])
    plt.legend(fontsize=SIZE)
    return fig


def plot_iter_dir(
    key_x, key2, list_l, labels, order=1, dict_key_to_limit={}, key3=None, key4=None, xlabel=None
):

    fig, ax1 = plt.subplots()
    if key3:
        ax2 = ax1.twinx()
    if key4:
        ax3 = ax1.twinx()

    if xlabel is None:
        xlabel = dict_key_to_label.get(key_x, key_x)
    ax1.set_xlabel(xlabel, fontsize=SIZE_AXIS)
    ax1.set_ylabel(dict_key_to_label.get(key2, key2), fontsize=SIZE_AXIS)
    if key3:
        ax2.set_ylabel(dict_key_to_label.get(key3, key3), fontsize=SIZE_AXIS)

    def plot_with_int(l, color, label):
        t = get_x(l, key_x)
        if t == []:
            return
        ax1.scatter(
            get_x(l, key_x), get_x(l, key2), color=color, label=label + key2 if order != 1 else None
        )
        fit_and_plot(key_x, key2, l, order, label + key2, color, ax=ax1)

        if key3:
            ax2.scatter(
                get_x(l, key_x),
                get_x(l, key3),
                color="red",
                label=label + key3 if order != 1 else None
            )
            fit_and_plot(key_x, key3, l, order, label + key3, color="red", ax=ax2)
        if key4:
            ax3.scatter(
                get_x(l, key_x),
                get_x(l, key4),
                color="yellow",
                #label=label + key4 if order != 1 else None
            )

    colors = cm.viridis(np.linspace(0, 1, len(list_l)))
    for i, l in enumerate(list_l):
        if labels is None:
            label = l[0]["dir"]
        else:
            label = labels[i]
        plot_with_int(l, color=colors[i], label=str(label))

    if key_x in dict_key_to_limit:
        ax1.set_xlim(dict_key_to_limit[key_x])
    if key2 in dict_key_to_limit:
        ax1.set_ylim(dict_key_to_limit[key2])
    if key3 in dict_key_to_limit:
        ax2.set_ylim(dict_key_to_limit[key3])

    # ax1.legend(loc='lower left')
    # if key3:
    #     ax2.legend(loc='upper right')
    return fig


def plot_soup_soupswa(key_x, keys2, order=1, dict_key_to_limit={}):
    plt.xlabel(dict_key_to_label.get(key_x, key_x), fontsize=SIZE_AXIS)
    plt.ylabel(dict_key_to_label["soup"], fontsize=SIZE_AXIS)

    def plot_with_int(l, color, label, key2):
        t = x(l, key_x)
        if t == []:
            return
        plt.scatter(x(l, key_x), get_x(l, key2), color=color, label=label if order != 1 else None)
        fit_and_plot(key_x, key2, l, order, label, color)

    colors = ["blue", "yellow"]
    labels = [
        "SOUP: $\\{\\theta_m\\}_1^M$ last checkpoints from different runs",
        "SOUPSWA: $\\{\\theta_m\\}_1^M$ SWA from different runs"
    ]
    #plot_with_int(l0, color="grey", key2="soup", label="swa")
    for i, key2 in enumerate(keys2):
        plot_with_int(l2, key2=key2, color=colors[i], label=labels[i])
    if key_x in dict_key_to_limit:
        plt.xlim(dict_key_to_limit[key_x])
    if key2 in dict_key_to_limit:
        plt.ylim(dict_key_to_limit[key2])
    plt.legend(fontsize=SIZE)
    return fig

import os
def save_fig(fig, name, folder="/home/rame/figures/rlwa/", do_save=True, format="png"):
    name = os.path.splitext(name)[0] + "." + format
    if do_save:
        fig.savefig(
            os.path.join(folder, name),
            format='pdf',
            dpi=600,
            bbox_inches='tight'
        )
# def save_fig(fig, name, folder="/private/home/alexandrerame/slurmconfig/notebook/filesdevfair/"):
# def save_fig(fig, name, folder="/Users/alexandrerame/code_repository/tex/model_recycling/images/filesdevfair"):

def get_result(liter, key_x, key2, reverse=True):
    sliter = [(l[key_x], l[key2], l["length"]) for l in liter if l["length"] > 1]
    sliter = sorted(sliter, reverse=True, key=lambda x: x[2])
    r = sorted(sliter, reverse=reverse, key=lambda x: x[1])[0]
    print(r)
    return r[0]


def get_result_oracle(liter, key_x, key2):
    sliter = [(l[key_x], l[key2], l["length"]) for l in liter]
    r = sorted(sliter, reverse=True, key=lambda x: x[0])[0]
    print(r)
    return r[0]


def print_result(l, key_x, key2):
    r = (l[key_x], l[key2], l["length"])
    print(r)
    return r[0]


def get_list_l_full(lib):
    l0 = [l for l in lib.liter if "e0_acc_net" in l]
    l1 = [l for l in lib.liter if "e1_acc_net" in l]
    l2 = [l for l in lib.liter if "e2_acc_net" in l]
    l3 = [l for l in lib.liter if "e3_acc_net" in l]
    return l0, l1, l2, l3


MAGIC_INDEX = 19


def plot_results_for_env(l, env, key="soup", key2="out_acc_soup", reverse=False, filter_=""):
    l0 = l[env]
    print("==")
    a01 = print_result(l0[MAGIC_INDEX], "e" + str(env) + "_" + filter_ + "acc_" + key, key2)
    a11 = get_result(l0, "e" + str(env) + "_" + filter_ + "acc_" + key, key2, reverse=reverse)
    a21 = get_result_oracle(
        l0,
        "e" + str(env) + "_" + filter_ + "acc_" + key,
        key2,
    )

    print(a01)
    print(a11)
    print(a21)


def plot_full_results_calib(l, key="soup", key2="out_acc_soup", reverse=False, filter_=""):
    l0, l1, l2, l3 = l
    print("acc", "hess", "M")
    a00 = print_result(l0[MAGIC_INDEX], "e0_" + filter_ + "acc_" + key, key2)
    a10 = get_result(l0, "e0_" + filter_ + "acc_" + key, key2, reverse=reverse)
    a20 = get_result_oracle(
        l0,
        "e0_" + filter_ + "acc_" + key,
        key2,
    )
    print("==")
    a01 = print_result(l1[MAGIC_INDEX], "e1_" + filter_ + "acc_" + key, key2)
    a11 = get_result(l1, "e1_" + filter_ + "acc_" + key, key2, reverse=reverse)
    a21 = get_result_oracle(
        l1,
        "e1_" + filter_ + "acc_" + key,
        key2,
    )
    print("==")
    a02 = print_result(l2[MAGIC_INDEX], "e2_" + filter_ + "acc_" + key, key2)
    a12 = get_result(l2, "e2_" + filter_ + "acc_" + key, key2, reverse=reverse)
    a22 = get_result_oracle(
        l2,
        "e2_" + filter_ + "acc_" + key,
        key2,
    )
    print("==")
    a03 = print_result(l3[MAGIC_INDEX], "e3_" + filter_ + "acc_" + key, key2)
    a13 = get_result(l3, "e3_" + filter_ + "acc_" + key, key2, reverse=reverse)
    a23 = get_result_oracle(
        l3,
        "e3_" + filter_ + "acc_" + key,
        key2,
    )

    m0 = np.mean([a00, a01, a02, a03])
    m1 = np.mean([a10, a11, a12, a13])
    m2 = np.mean([a20, a21, a22, a23])

    print()
    print(
        f"DWA Top{MAGIC_INDEX+1} & " +
        " & ".join(["{:.1f}".format(x * 100) for x in [a00, a01, a02, a03, m0]]) + " \\\\"
    )
    print(
        "DWA Grow & " + " & ".join(["{:.1f}".format(x * 100) for x in [a10, a11, a12, a13, m1]]) +
        " \\\\"
    )
    print(
        "DWA Oracle & " + " & ".join(["{:.1f}".format(x * 100) for x in [a20, a21, a22, a23, m2]]) +
        " \\\\"
    )


def plot_full_results(l, key="soup", filter_=""):
    l0, l1, l2, l3 = l
    print("acc", "hess", "M")
    a00 = print_result(l0[MAGIC_INDEX], "e0_" + filter_ + "acc_" + key, "e123_out_souphess")
    a10 = get_result(l0, "e0_" + filter_ + "acc_" + key, "e123_out_souphess", reverse=False)
    a20 = get_result_oracle(
        l0,
        "e0_" + filter_ + "acc_" + key,
        "e123_out_souphess",
    )
    print("==")
    a01 = print_result(l1[MAGIC_INDEX], "e1_" + filter_ + "acc_" + key, "e023_out_souphess")
    a11 = get_result(l1, "e1_" + filter_ + "acc_" + key, "e023_out_souphess", reverse=False)
    a21 = get_result_oracle(
        l1,
        "e1_" + filter_ + "acc_" + key,
        "e023_out_souphess",
    )
    print("==")
    a02 = print_result(l2[MAGIC_INDEX], "e2_" + filter_ + "acc_" + key, "e013_out_souphess")
    a12 = get_result(l2, "e2_" + filter_ + "acc_" + key, "e013_out_souphess", reverse=False)
    a22 = get_result_oracle(
        l2,
        "e2_" + filter_ + "acc_" + key,
        "e013_out_souphess",
    )
    print("==")
    a03 = print_result(l3[MAGIC_INDEX], "e3_" + filter_ + "acc_" + key, "e012_out_souphess")
    a13 = get_result(l3, "e3_" + filter_ + "acc_" + key, "e012_out_souphess", reverse=False)
    a23 = get_result_oracle(
        l3,
        "e3_" + filter_ + "acc_" + key,
        "e012_out_souphess",
    )

    m0 = np.mean([a00, a01, a02, a03])
    m1 = np.mean([a10, a11, a12, a13])
    m2 = np.mean([a20, a21, a22, a23])

    print()
    print(
        f"DWA Top{MAGIC_INDEX+1} & " +
        " & ".join(["{:.1f}".format(x * 100) for x in [a00, a01, a02, a03, m0]]) + " \\\\"
    )
    print(
        "DWA Grow & " + " & ".join(["{:.1f}".format(x * 100) for x in [a10, a11, a12, a13, m1]]) +
        " \\\\"
    )
    print(
        "DWA Oracle & " + " & ".join(["{:.1f}".format(x * 100) for x in [a20, a21, a22, a23, m2]]) +
        " \\\\"
    )


# fig = plot_iter_dir("out_acc_soup", "e0_acc_soup", list_l=[get_lv1(0), get_lv1(1), get_lv1(2), get_lv1(-1)], labels=None)

# from data.home import lswa_hpd, l_nodrop, l_drop, l_hpeoa, lswa_hpl, lswa_hpl_v2
# from data.home import lsoup_hpd, lsoup_hpeoa, lsoup_hpl_35, lsoup_hps, lsoup_hpeoa_10003000, lsoup_hpeoa_10002000, lsoup_hpeoa_5000
# from data.home import lsoup_hpd_ehess, lsoup_hpeoa_ehess, lsoup_hpl_ehess, lsoup_hps_203601
# from data.home import lsoup_hpl_topk0_35, lsoup_hpl_topk0_5, lsoup2_hpl_samedata, lsoup_hps_samedata_ermmixupcoral_hess_0413

# l = merge(l_nodrop.l, l_drop.l, l_hpeoa.l, lswa_hpd.l)
# lsoup = merge(
#     lsoup_hpd.lsoup, lsoup_hpeoa_10003000.lsoup, lsoup_hpeoa.lsoup, lsoup_hps.lsoup,
#     lsoup_hpl_35.lsoup, lsoup_hpeoa_10002000.lsoup, lsoup_hpeoa_5000.lsoup, lsoup_hpd_ehess.lsoup,
#     lsoup_hpeoa_ehess.lsoup, lsoup_hpl_ehess.lsoup, lsoup_hps_203601.lsoup_hp203,
#     lsoup_hps_203601.lsoup_hp601,
# )
# lsoupl = merge(lsoup_hpl_topk0_35.l, lsoup_hpl_topk0_5.lsoup, lsoup2_hpl_samedata.lsoup)

# lhesssoup = merge(
#     lsoup_hpd_ehess.lsoup,
#     lsoup_hpeoa_ehess.lsoup,
#     #lsoup_hpl_ehess.lsoup,
#     lsoup_hps_203601.lsoup_hp203,
#     lsoup_hps_203601.lsoup_hp601,
#     lsoup_hps_hessout_ermmixupcoral.lsoup
# )

# lhess = merge(
#     lsoup_hpd.lsoup,  # bottom left
#     lsoup_hpeoa.lsoup,  # up left
#     lsoup_hps.lsoup,  # only 234
#     l_hpeoa.l,
#     lsoup_hpd_ehess.lsoup,
#     lsoup_hpeoa_ehess.lsoup,
#     lsoup_hpl_ehess.lsoup,
#     lsoup_hps_203601.lsoup_hp203,
#     lsoup_hps_203601.lsoup_hp601
# )

# from data.home import lsoup_hps_env0_div2_ermmixupcoral, lsoup_hps_env0_div2, lsoup_hps_hessout_ermmixupcoral
# # divacrossregul, divacrossrefulv2,
# from data6.home.combin import combinhome0_emvc_hpx_0412

# lsoupemc = plot.merge(
#     lsoup_hps_hessout_ermmixupcoral.lsoup,
#     lsoup_hps_samedata_ermmixupcoral_hess_0413.lsoup,
#     lsoup_hps_env0_div2_ermmixupcoral.lsoup, combinhome0_emvc_hpx_0412.lsoup
# )


def merge_list0_list1(list0, list1):

    for i in range(4):
        for j, l in enumerate(list0[i]):
            l.update(list1[i][j])


def compare_203_601():

    key = 3
    trainkeys = "012"
    fig = plot_iter_dir(
        "length",
        "e" + trainkeys + "_out_souphess",
        list_terra_203[key:key + 1],
        labels,
        order="0",
        xlabel="terra_203",
        dict_key_to_limit=dict_key_to_limit,
        key3="e" + str(key) + "_acc_soup"
    )
    fig = plot_iter_dir(
        "length",
        "out_acc_soup",
        list_terra_203[key:key + 1],
        labels,
        order="0",
        xlabel="terra_203",
        dict_key_to_limit=dict_key_to_limit,
        key3="e" + str(key) + "_acc_soup"
    )
    fig = plot_iter_dir(
        "length",
        "train_out_acc_soup",
        list_terra_601[key:key + 1],
        labels,
        order="0",
        xlabel="terra_601",
        dict_key_to_limit=dict_key_to_limit,
        key3="e" + str(key) + "_in_acc_soup"
    )
    fig = plot_iter_dir(
        "length",
        "e" + trainkeys + "_out_souphess",
        list_terra_601[key:key + 1],
        labels,
        order="0",
        xlabel="terra_601",
        dict_key_to_limit=dict_key_to_limit,
        key3="e" + str(key) + "_in_acc_soup"
    )

    fig = plot_iter_dir(
        "length",
        "e" + trainkeys + "_out_souphess",
        list_home_203[key:key + 1],
        labels,
        order="0",
        xlabel="home_203",
        dict_key_to_limit=dict_key_to_limit,
        key3="e" + str(key) + "_acc_soup"
    )
    fig = plot_iter_dir(
        "length",
        "out_acc_soup",
        list_home_203[key:key + 1],
        labels,
        order="0",
        xlabel="home_203",
        dict_key_to_limit=dict_key_to_limit,
        key3="e" + str(key) + "_acc_soup"
    )


def plot_slope():

    fig_dr = plot_key(
        l=merge(l),
        key_x="dr",
        key2="soup-netm",
        order="",
        label="M=",
        _dict_key_to_label=dict_key_to_label
    )
    fig_df = plot_key(
        l=merge(l),
        key_x="df",
        key2="soup-netm",
        order="",
        label="M=",
        _dict_key_to_label=dict_key_to_label
    )
    save_fig(fig_df, "samediffruns_df_soup.pdf")
    save_fig(fig_dr, "samediffruns_dr_soup.pdf")


def plot_large_hp():

    dict_key_to_limit = {
        "soup-netm": [-0.5, 0.12],
        "df": [0.05, 0.80],
        "dr": [0.36, 1.07],
        "soup": [0.05, 0.72]
    }
    fig_dr = plot_key(
        l=merge(l, lsoupall),
        key_x="dr",
        key2="soup-netm",
        order="",
        label="M=",
        _dict_key_to_label=dict_key_to_label,
        _dict_key_to_limit=dict_key_to_limit
    )
    fig_df = plot_key(
        l=merge(l, lsoupall),
        key_x="df",
        key2="soup-netm",
        order="",
        label="M=",
        _dict_key_to_label=dict_key_to_label,
        _dict_key_to_limit=dict_key_to_limit
    )
    save_fig(fig_df, "samediffrunslargehp_df_soup.pdf")
    save_fig(fig_dr, "samediffrunslargehp_dr_soup.pdf")


def plot_large_hp():

    lsoup = merge(lsoup_hps_203601.lsoup_hp601, lsoup_hps_env0_div2.lsamedata)
    lsoupl = merge(lsoup2_hpl_samedata.lsoup, lsoup_hpl_topk0_35.l)

    markers = ["*" * len(lsoup[i]) + "s" * len(lsoupl[i]) for i in range(len(l))]

    blues = cm.Blues(np.linspace(0.5, 1, 2))
    reds = cm.Reds(np.linspace(0.5, 1, 2))
    colors = [[blues[0]] * len(lsoup[i]) + [reds[0]] * len(lsoupl[i]) for i in range(len(l))]

    labels = ["Standard hyperparams", "Extreme hyperparams"]
    plt.rcParams["figure.figsize"] = (5, 5)

    fig_dr = plot_key(
        l=merge(lsoup, lsoupl),
        markers=markers,
        fcard=2,
        key_x="dr",
        key2="soup-netm",
        order="",
        labels=labels,
        colors=colors,
        _dict_key_to_label=dict_key_to_label,
        _dict_key_to_limit={}
    )

    save_fig(fig_dr, "diffrunslargehp_dr_soup-netm.pdf")


def plot_fig_hess():

    THESS = False
    EHESS = True
    fig_ehess_soup = plot_key(
        l=merge(lhesssoup)[:-1],
        key_x="hess",
        key2="soup",
        order="",
        label="M=",
        _dict_key_to_limit={
            "hess": [8000, 40000],
            "soup-netm": [0.02, 0.12]
        }
    )
    save_fig(fig_ehess_soup, "diffruns_ehess_soup.pdf")
    fig_ehess_dr = plot_key(
        l=merge(lhesssoup)[:-1],
        key_x="hess",
        key2="dr",
        order="",
        label="M=",
        _dict_key_to_limit={
            "hess": [8000, 40000],
            "soup-netm": [0.02, 0.12]
        }
    )
    save_fig(fig_ehess_dr, "diffruns_ehess_dr.pdf")


def get_list_l_full(lib_liter):
    l0 = [l for l in lib_liter if "e0_acc_soup" in l]
    l1 = [l for l in lib_liter if "e1_acc_soup" in l]
    l2 = [l for l in lib_liter if "e2_acc_soup" in l]
    l3 = [l for l in lib_liter if "e3_acc_soup" in l]
    return l0, l1, l2, l3




def plot_robust(ll_m, key_x, orders=None, key_axis1="acc", key_axis2=None, labels=None, key_annot=None, legends=None, _dict_key_to_limit={}, title=None, loc="lower left"):

    if labels is None:
        labels = [str(i) for i in range(len(ll_m))]
    if orders is None:
        orders = ["no", "no"]
    if not isinstance(orders, list):
        orders = [orders for i in range(len(ll_m))]
    fig, ax1 = plt.subplots()
    plt.xlabel(dict_key_to_label.get(key_x, key_x), fontsize=SIZE_AXIS)
    colors = cm.rainbow(np.linspace(0.3, 1, len(ll_m)))
    ax1.set_ylabel(dict_key_to_label[key_axis1], fontsize=SIZE_AXIS)

    def plot_with_int(l, ax, color, label, key2, marker, linestyle, order):
        t = get_x(l, key_x)
        if t == []:
            return

        l = [ll for ll in l if key2 in ll]
        ax.scatter(
            get_x(l, key_x),
            get_x(l, key2),
            color=color,
            label=label,
            marker=marker
        )
        if key_annot:
            n = len(get_x(l, key_x))
            for i in range(n):
                ax.annotate(get_x(l, key_annot)[i], (get_x(l, key_x)[i], get_x(l, key2)[i]), color=color)

        if order != "no":
            fit_and_plot(key_x, key2, l, order=order, label=label, color=color, ax=ax, linestyle=linestyle)

    for i, l_m in enumerate(ll_m):
        plot_with_int(
            l_m,
            ax1,
            color=colors[i],
            label=dict_key_to_label.get(i, labels[i]),
            key2=key_axis1,
            marker="." if i %2 else "x",
            order=orders[0],
            linestyle="-"
        )
    if key_axis1 in _dict_key_to_limit:
        ax1.set_ylim(_dict_key_to_limit[key_axis1])

    if key_axis2:
        ax2 = ax1.twinx()
        ax2.set_ylabel(dict_key_to_label[key_axis2], fontsize=SIZE)

        for i, l_m in enumerate(ll_m):
            plot_with_int(
                l_m,
                ax2,
                color=colors[i],
                label=dict_key_to_label.get(i, labels[i]),
                key2=key_axis2,
                marker="x",
                order=orders[-1],
                linestyle="--"
            )
        if key_axis2 in dict_key_to_limit:
            ax2.set_ylim(dict_key_to_limit[key_axis2])
    if key_x in dict_key_to_limit:
        plt.xlim(dict_key_to_limit[key_x])
    if loc != "no":
        legend1 = ax1.legend(fontsize=SIZE, loc=loc + (" left" if key_axis2 else ""))
        title1 = dict_key_to_label.get(key_axis1)
        if legends:
            title1= legends[0]
        legend1.set_title(title1, prop = {'size': 15})
        if key_axis2:
            title2 = dict_key_to_label.get(key_axis2)
            if legends:
                title2 = legends[1]
            legend2 = ax2.legend(fontsize=SIZE, loc=loc + " right")
            legend2.set_title(title2, prop = {'size': 15})
    if title:
        fig.suptitle(title, fontsize=20)
    return fig

# list_home_601_erm = get_list_l_full(home_iter_hps_env0_ermmixupcoral.lerm)
# list_home_601_ermmixupcoral = get_list_l_full(home_iter_hps_env0_ermmixupcoral.lermmixupcoral)


def plot_continual(l, labels, label=None, name="home0", do_iid=False, do_save=False, do_h=True, index_range=3, order=3, linestyles=["solid", "dashed", "dotted"]):

    list_indexes = range(0, 0 + index_range)

    if labels is not None:
        labels = [l.replace("RXRX", "RxRx") for l in labels]

    fig_dr = plot_key(
        l,
        key_x="weighting",
        key2="acc",
        labels=labels,
        label=label,
        order=order,
        loc="lower right",
        _dict_key_to_label={},
        list_indexes=list_indexes,
        linestyles=linestyles
    )
    if do_save:
        save_fig(fig_dr, "lmc/" + name + "_lmc_hyp1_ood.pdf")

    if do_iid:
        fig_dr = plot_key(
            l,
            key_x="weighting",
            key2="train_acc",
            labels=labels,
            label=label,
            order=order,
            loc="lower right",
            _dict_key_to_label={},
            list_indexes=list_indexes,
            linestyles=linestyles
        )
        if do_save:
            save_fig(fig_dr, "lmc/" + name + "_lmc_hyp1_iid.pdf")
    if do_h:
        list_indexes = range(3, 3 + index_range)

        fig_dr = plot_key(
            l,
            key_x="weighting",
            key2="acc",
            labels=labels,
            label=label,
            order=order,
            loc="lower right",
            _dict_key_to_label={},
            list_indexes=list_indexes,
            linestyles=linestyles
        )
        if do_save:
            save_fig(fig_dr, "lmc/" + name + "_lmc_hyp1h_ood.pdf")
        if do_iid:
            fig_dr = plot_key(
                l,
                key_x="weighting",
                key2="train_acc",
                labels=labels,
                label=label,
                order=order,
                loc="lower right",
                _dict_key_to_label={},
                list_indexes=list_indexes,
                linestyles=linestyles
            )
            if do_save:
                save_fig(fig_dr, "lmc/" + name + "_lmc_hyp1h_iid.pdf")

    list_indexes = range(6, 6 + index_range)

    fig_dr = plot_key(
        l,
        key_x="weighting",
        key2="acc",
        labels=labels,
        label=label,
        order=order,
        loc="lower right",
        _dict_key_to_label={},
        list_indexes=list_indexes,
        linestyles=linestyles
    )
    if do_save:
        save_fig(fig_dr, "lmc/" + name + "_lmc_hyp2_ood.pdf")
    if do_iid:
        fig_dr = plot_key(
            l,
            key_x="weighting",
            key2="train_acc",
            labels=labels,
            label=label,
            order=order,
            loc="lower right",
            _dict_key_to_label={},
            list_indexes=list_indexes,
            linestyles=linestyles
        )
        if do_save:
            save_fig(fig_dr, "lmc/" + name + "_lmc_hyp2_iid.pdf")

    if do_h:
        list_indexes = range(9, 9 + index_range)
        fig_dr = plot_key(
            l,
            key_x="weighting",
            key2="acc",
            labels=labels,
            label=label,
            order=order,
            loc="lower right",
            _dict_key_to_label={},
            list_indexes=list_indexes,
            linestyles=linestyles
        )
        if do_save:
            save_fig(fig_dr, "lmc/" + name + "_lmc_hyp2h_ood.pdf")
        if do_iid:
            fig_dr = plot_key(
                l,
                key_x="weighting",
                key2="train_acc",
                labels=labels,
                label=label,
                order=order,
                loc="lower right",
                _dict_key_to_label={},
                list_indexes=list_indexes,
                linestyles=linestyles
            )
            if do_save:
                save_fig(fig_dr, "lmc/" + name + "_lmc_hyp2h_iid.pdf")
