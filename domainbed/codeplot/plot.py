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
        if key.split("|")[0] == "norm":
            vals = [i for i in get_x(l, key.split("|")[1])]
            min_vals = min(vals)
            max_vals = max(vals)
            return [(i-min_vals)/(max_vals-min_vals) for i in vals]
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

def check_condition(i):
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
    "1-lambda": '$\lambda$',
}

import numpy as np
# from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

def interpolate_points(val1, val2, order, label, color, marker=None, ax=None, linestyle="-", linewidth=1):
    if ax is None:
        ax = plt

    if order == "nofit":
        return

    # x, y = np.array(val1), np.array(val2)

    # X_Y_Spline = make_interp_spline(x, y)

    # # Returns evenly spaced numbers
    # # over a specified interval.
    # X_ = np.linspace(x.min(), x.max(), 500)
    # Y_ = X_Y_Spline(X_)
    # plt.plot(X_, Y_)
    # return

    if order in ['slinear', 'quadratic', 'cubic'] or (isinstance(order, str) and order.startswith("savgol")):
        try:
            if order.startswith("savgol"):
                x_smooth = savgol_filter(val1, int(order.split("_")[1]), int(order.split("_")[2]))
                x_smooth[0], x_smooth[-1] = val1[0], val1[-1]
                y_smooth = savgol_filter(val2, int(order.split("_")[1]), int(order.split("_")[2]), mode="nearest")
                y_smooth[0], y_smooth[-1] = val2[0], val2[-1]
            else:
                x_smooth = val1
                y_smooth = val2

            points = np.array([x_smooth, y_smooth]).T  # a (nbre_points x nbre_dim) array

            # Linear length along the line:
            distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
            distance = np.insert(distance, 0, 0) / distance[-1]

            alpha = np.linspace(0, 1, 75)

            interpolator = interp1d(distance, points, kind=order.split("_")[-1], axis=0)
            curve = interpolator(alpha)

            # Graph:
            ax.plot(
                *curve.T,
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=0,
                linewidth=linewidth
            )

            # ax.plot(*points.T, 'ok', label='original points')
            return
        except Exception as exc:
            print(f"Failed for label: {label} because: {exc}")
            order = "connect"

    if order == "inter1":
        x, y = x_smooth, val2
        t = np.arange(len(x))
        ti = np.linspace(0, t.max(), 10 * t.size)

        xi = interp1d(t, x, kind='cubic')(ti)
        yi = interp1d(t, y, kind='cubic')(ti)
        ax.plot(
            xi,
            yi,
            label=label,
            color=color,
            linestyle=linestyle,
            markersize=0,
            linewidth=linewidth
        )
        return

    if order == "connect":
        ax.plot(
            val1,
            val2,
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=0,
            linewidth=linewidth
        )
        return

    get_x1_sorted = np.linspace(min(val1), max(val1), 500000)

    if order in [1, "1"]:
        m, b = np.polyfit(val1, val2, 1)
        preds = m * np.array(get_x1_sorted) + b
    elif order in [2, "2"]:
        m2, m1, b = np.polyfit(val1, val2, 2)
        preds = m2 * np.array(get_x1_sorted)**2 + m1 * np.array(get_x1_sorted) + b
    elif order in [3, "3"]:
        m3, m2, m1, b = np.polyfit(val1, val2, 3)
        preds = m3 * np.array(get_x1_sorted)**3 + m2 * np.array(get_x1_sorted)**2 + m1 * np.array(
            get_x1_sorted
        ) + b
    elif order in [4, "4"]:
        m4, m3, m2, m1, b = np.polyfit(val1, val2, 4)
        preds = m4 * np.array(get_x1_sorted)**4 + m3 * np.array(get_x1_sorted)**3 + m2 * np.array(
            get_x1_sorted
        )**2 + m1 * np.array(get_x1_sorted) + b
    elif order in [5, "5"]:
        m5, m4, m3, m2, m1, b = np.polyfit(val1, val2, 5)
        preds = m5 * np.array(get_x1_sorted)**5 + m4 * np.array(get_x1_sorted)**4 + m3 * np.array(
            get_x1_sorted
        )**3 + m2 * np.array(get_x1_sorted)**2 + m1 * np.array(get_x1_sorted) + b
    elif order == "log":
        m1, b = np.polyfit(np.log(val1), val2, 1)
        log_get_x1_sorted = np.log(get_x1_sorted)
        preds = m1 * np.array(log_get_x1_sorted) + b
    elif order == "2log":
        m2, m1, b = np.polyfit(np.log(val1), val2, 2)
        log_get_x1_sorted = np.log(get_x1_sorted)
        preds = m2 * np.array(log_get_x1_sorted)**2 + m1 * np.array(log_get_x1_sorted) + b
    elif order == "3log":
        m3, m2, m1, b = np.polyfit(np.log(val1), val2, 3)
        log_get_x1_sorted = np.log(get_x1_sorted)
        preds = m3 * np.array(log_get_x1_sorted)**3 + m2 * np.array(
            log_get_x1_sorted
        )**2 + m1 * np.array(log_get_x1_sorted) + b
    else:
        assert order in [0, -1, None, "", "0"]

    ax.plot(
        get_x1_sorted, preds, color=color, linestyle=linestyle, label=label,
        marker=marker,
        markersize=0,
        linewidth=linewidth
    )


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (6, 6)
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times Roman'

MUL = 0


def fit_and_plot(key1, key2, l, order, label, color, ax=None, marker=None, linestyle="-"):
    return interpolate_points(
        val1=get_x(l, key1),
        val2=get_x(l, key2),
        order=order,
        label=label,
        color=color,
        ax=ax,
        marker=marker,
        linestyle=linestyle
    )

SIZE="large"
SIZE_AXIS="xx-large"


def plot_basic_scatter(list_dict_values, key_x, keys_y, labels=None, _dict_key_to_label="def", colors=None, colormaps=None, keycolor=None, order=0, linestyles=None, keys_error=None, loc="best", title=None, keyclustering=None,  kwargs={}, _dict_key_to_limit={}, ax1=None, lambda_filtering=None, markers=None, legendtitle=None, markersize=12):
    if ax1 is None:
        fig, ax1 = plt.subplots()
    else:
        fig = None
    if _dict_key_to_label == "def":
        _dict_key_to_label = dict_key_to_label
    else:
        _dict_key_to_label = {**dict_key_to_label, **_dict_key_to_label}

    if colormaps is None:
        if keycolor is not None:
            colormaps = [
                "Reds", "Blues", "Greens", "Oranges", "Greys", "Purples", "Reds", "Blues", "Greens",
                "Oranges", "Greys", "Purples"
            ][:len(keys_y)]
    else:
        dict_colormaps = create_colormaps()
        colormaps = [dict_colormaps.get(cmp, cmp) for cmp in colormaps]

    if colors is None:
        if colormaps is not None:
            colors = [cm.get_cmap(cmp)(0.5) for cmp in colormaps]
        else:
            colors = cm.rainbow(np.linspace(0, 1, len(keys_y)))

    if lambda_filtering is not None:
        list_dict_values = [lll for lll in list_dict_values if lambda_filtering(lll)]

    if keyclustering is not None:
        list_dict_values_means = lambda_clustering(list_dict_values, keyclustering)

    if "x" in _dict_key_to_limit:
        ax1.set_xlim(_dict_key_to_limit["x"])
    if "y" in _dict_key_to_limit:
        ax1.set_ylim(_dict_key_to_limit["y"])

    for index in range(len(keys_y)):
        key_y = keys_y[index]
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

        color = colors[index]
        if labels is None:
            label = key_y
        else:
            label = labels[index]
        label = _dict_key_to_label.get(label, label)

        if keys_error is not None:
            # plt.errorbar(
            #     x,
            #     y,
            #     get_x(list_dict_values, keys_error[i]),
            #     color=color,
            #     label=label)
            ax1.fill_between(
                get_x(list_dict_values_means, key_x),
                get_x(list_dict_values_means, key_y + "-" + keys_error[index]),
                get_x(list_dict_values_means, key_y + "+" + keys_error[index]),
                color=color,
                # label=label,
                **kwargs)

        if keycolor is not None:
            ax1.scatter(
                get_x(list_dict_values, key_x),
                get_x(list_dict_values, key_y),
                c=get_x(list_dict_values, keycolor),
                cmap=colormap,
                marker=marker,
                label=None if order != "nofit" else label,
                **kwargs
            )
            color = cm.get_cmap(color)(0.5)
        else:
            ax1.scatter(
                get_x(list_dict_values, key_x),
                get_x(list_dict_values, key_y),
                color=color,
                marker=marker,
                s=[markersize for _ in get_x(list_dict_values, key_y)],
                label=None if order != "nofit" else label,
            )
        interpolate_points(
            get_x(list_dict_values, key_x),
            get_x(list_dict_values, key_y),
            order=order,
            ax=ax1,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.5
        )

    ax1.set_xlabel(_dict_key_to_label.get(key_x, key_x), fontsize=SIZE)
    ax1.set_ylabel("Normalized rewards", fontsize=SIZE)
    if loc != "no":
        if isinstance(loc, tuple):
            legend = ax1.legend(title=legendtitle, bbox_to_anchor=loc, fontsize=SIZE)
        else:
            legend = ax1.legend(title=legendtitle, loc=loc, fontsize=SIZE)
        for lgnd in legend.legendHandles:
            lgnd.set_markersize(8)
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

import matplotlib
from matplotlib.colors import ListedColormap,LinearSegmentedColormap

def get_color_from_cmap(cmp, dict_colormaps):
    cmp = dict_colormaps.get(cmp, cmp)
    if isinstance(cmp, str) and cmp.startswith("fake_"):
        return cmp.split("_")[1]
    return cm.get_cmap(cmp)(0.5)


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
    dict_colormaps["Blues_Greens"] = create_cmp(125, 150, 250)
    dict_colormaps["Reds_Greens"] = create_cmp(250, 150, 125)
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
    linewidths=None,
    _dict_key_to_limit={},
    _dict_key_to_label="def",
    loc="upper right",
    legendtitle=None,
    lambda_filtering=None,
    keyclustering=None,
    list_indexes=None,
    keyerror=None,
    title=None,
    connect_endpoints=False,
    fontsize=None,
    kwargs={}
):
    if list_indexes is None:
        list_indexes = [i for i, ll in enumerate(l) if ll is not None]
        if len(list_indexes) == len(l):
            list_indexes = None
    if list_indexes is not None:
        l = [l[i] for i in list_indexes]
        if labels is not None:
            labels = [labels[i] for i in list_indexes]
        if colormaps is not None:
            colormaps = [colormaps[i] for i in list_indexes]
        if colors is not None:
            colors = [colors[i] for i in list_indexes]
        if linestyles is not None:
            linestyles = [linestyles[i] for i in list_indexes]
        if linewidths is not None:
            linewidths = [linewidths[i] for i in list_indexes]
        if markers is not None:
            markers = [markers[i] for i in list_indexes]

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
        colormaps = [
            dict_colormaps.get(cmp, cmp) if isinstance(cmp, str) else cmp for cmp in colormaps
        ]

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


    def plot_with_int(ll, color, colormap, label, marker, linestyle, linewidth, key_y, ax, kwargs):
        ll = [lll for lll in ll if lambda_filtering is None or lambda_filtering(lll)]
        t = get_x(ll, key_x)
        if t == []:
            return
        if keyclustering is not None:
            ll = lambda_clustering(ll, keyclustering)

        # if label !="no" else None#
        if keyerror is not None:
            ax.errorbar(
                get_x(ll, key_x),
                get_x(ll, key_y),
                get_x(ll, keyerror),
                color=color,
                label=label,
                marker=marker,
                **kwargs
            )
            plt.fill_between(get_x(ll, key_x),
                get_x(ll, key_y + "-" + keyerror),
                get_x(ll, key_y + "+" + keyerror),
                alpha=0.5,
                )
        else:
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

        interpolate_points(
            get_x(ll, key_x),
            get_x(ll, key_y),
            order=order,
            ax=ax,
            label=label,
            color=color,
            marker=marker,
            linewidth=linewidth,
            linestyle=linestyle
        )
        ax.scatter(
            get_x(ll, key_x),
            get_x(ll, key_y),
            label=None if order != "nofit" else label,
            marker=marker,
            **kwargs
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
        if linewidths is not None:
            linewidth = linewidths[index]
        else:
            linewidth = None
        kwargs_copy = {k:v for k, v in kwargs.items()}
        plot_with_int(l[index], color=colors[index], colormap=colormap, label=label, marker=marker, linewidth=linewidth, linestyle=linestyle, key_y=key_y, ax=ax1, kwargs=kwargs_copy)
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
        fontsize = fontsize or SIZE
        if isinstance(loc, tuple):
            legend = ax1.legend(title=legendtitle, bbox_to_anchor=loc, fontsize=fontsize)
        else:
            legend = ax1.legend(title=legendtitle, loc=loc, fontsize=fontsize)
        for lgnd in legend.legendHandles:
            lgnd.set_markersize(6)
        # legend = ax1.get_legend()
        # if keycolor is not None:
        #     legend = ax1.get_legend()
        #     for i, cmap_name in enumerate(colormaps[:len(l)]):
        #         cmap = cm.get_cmap(cmap_name)
        #         legend.legendHandles[i].set_color(cmap(0.5))
    if title:
        ax1.title(title, fontsize=SIZE)
    return fig


import os
def save_fig(fig, name, folder="/home/rame/figures/rlwa/", do_save=True, format="pdf"):
    name = os.path.splitext(name)[0] + "." + format
    if do_save:
        fig.savefig(
            os.path.join(folder, name),
            format=format,
            dpi=600,
            bbox_inches='tight'
        )
# def save_fig(fig, name, folder="/private/home/alexandrerame/slurmconfig/notebook/filesdevfair/"):
# def save_fig(fig, name, folder="/Users/alexandrerame/code_repository/tex/model_recycling/images/filesdevfair"):
