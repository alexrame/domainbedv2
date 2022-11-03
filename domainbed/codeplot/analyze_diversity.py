import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt


def merge(*ll):
    return [[y for x in l for y in x] for l in zip(*ll)]


# from data.home import lsoup_hpeoa, lsoup_hpl_35, lsoup_hpl_topk0_5, lsoup_hps, lsoup_hpd

# from data.home import lswa_hpd, l_nodrop, l_drop, l_hpeoa, lswa_hpl, lswa_hpl_v2

# def plot_diversity_all(key="df", keys=["swa", "soup", "same", "diff"]):
#     fig = plt.figure()
#     plt.rcParams["figure.figsize"] = (5, 5)
#     kwargs = dict(alpha=0.5, bins=15, density=True, stacked=False)

#     keyname = "Feature" if key == "df" else "Prediction"

#     if "swa" in keys:
#         colors_reds = cm.Reds(np.linspace(0.3, 1, 3))
#         plt.hist(
#             [line[key] for c in lswa_hpd.l for line in c],
#             **kwargs,
#             color=colors_reds[0],
#             label='One run with default hyperparameters'
#         )

#         plt.hist(
#             [line[key] for c in merge(l_nodrop.l, l_drop.l, l_hpeoa.l) for line in c],
#             **kwargs,
#             color=colors_reds[1],
#             label='One run with restrained hyperparameter ranges'
#         )

#         plt.hist(
#             [line[key] for c in merge(lswa_hpl.l, lswa_hpl_v2.l) for line in c],
#             **kwargs,
#             color=colors_reds[2],
#             label='One run with full hyperparameter ranges'
#         )

#     if "soup" in keys:
#         colors_blues = cm.Blues(np.linspace(0.3, 1, 3))

#         plt.hist(
#             [line[key] for c in merge(lsoup_hpd.lsoup, lsoup_hpd_ehess.lsoup) for line in c],
#             **kwargs,
#             color=colors_blues[0],
#             label='Two runs with default hyperparameters and different data'
#         )

#         # soup_hpeoa3 = [line[key] for c in lsoup_hpl_35.lsoup for line in c if line["step"] == 3000]
#         # plt.hist(soup_hpeoa3, **kwargs, color=colors_blues[2], label='Two runs with dropout and various hyperparameters at epoch 3000')

#         plt.hist(
#             [
#                 line[key] for c in merge(
#                     lsoup_hps_203601.lsoup_hp601
#                     # lsoup_hpeoa.lsoup, lsoup_hpeoa_ehess.lsoup, lsoup_hpeoa_5000.lsoup,
#                 ) for line in c
#             ],
#             **kwargs,
#             color=colors_blues[1],
#             label='Two runs with restrained hyperparameter ranges and same data'
#         )

#         plt.hist(
#             [
#                 line[key] for c in merge(
#                     lsoup_hps.lsoup, lsoup_hps_203601.lsoup_hp203
#                     # lsoup_hpeoa.lsoup, lsoup_hpeoa_ehess.lsoup, lsoup_hpeoa_5000.lsoup,
#                 ) for line in c
#             ],
#             **kwargs,
#             color=colors_blues[1],
#             label='Two runs with restrained hyperparameter ranges and different data'
#         )

#         plt.hist(
#             [
#                 line[key]
#                 for c in merge(lsoup_hpl_topk0_5.lsoup)
#                 #, lsoup_hpl_ehess.lsoup) do not include because topk=20
#                 for line in c
#                 if line.get("step", 5000) == 5000
#             ],
#             **kwargs,
#             color=colors_blues[2],
#             label='Two runs with full hyperparameter ranges and different data'
#         )

#     if "same" in keys:
#         colors_dataaug = cm.YlOrBr(np.linspace(0.3, 1, 3))
#         dataaugs = [["erm", "erm"], ["coral", "coral"], ["mixup", "mixup"]]
#         for i in range(3):
#             add_div_losses(key, dataaugs[i][0], dataaugs[i][1], kwargs, color=colors_dataaug[i])

#     if "diff" in keys:
#         colors_dataaug = cm.Greens(np.linspace(0.3, 1, 3))
#         dataaugs = [
#             ["erm", "mixup"],
#             ["erm", "coral"],
#             ["mixup", "coral"]
#         ]
#         for i in range(3):
#             add_div_losses(key, dataaugs[i][0], dataaugs[i][1], kwargs, color=colors_dataaug[i])

#     plt.gca().set(title=keyname + ' diversity', ylabel='Count')
#     if key == "dr":
#         plt.xlim(0.4, 0.85)
#     else:
#         plt.xlim(0.1, 0.7)
#     plt.ylim(0, 30)
#     plt.legend()
#     return fig

# def add_div_losses(key, dataaug1, dataaug2, kwargs, color):
#     div_dataaug = [l[key] for l in divacrossregul.l + divacrossrefulv2.l if dataaug1 in l["dirs"] and dataaug2 in l["dirs"]]
#     plt.hist(
#         div_dataaug,
#         **kwargs,
#         color=color,
#         label=f'Two restrained runs with dataaug: {dataaug1} and {dataaug2}'
#     )

# from data.home import lswa_hpd, l_nodrop, l_drop, l_hpeoa, lswa_hpl, lswa_hpl_v2
# from data.home import lsoup_hpd, lsoup_hpd_ehess
from data.home import lsoup_hpeoa, lsoup_hps, lsoup_hpeoa_5000, lsoup_hpeoa_ehess
# from data.home import lsoup_hpl_ehess, lsoup_hpl_topk0_5

from data.home import lswa_hpd, l_nodrop, l_drop, l_hpeoa, lswa_hpl, lswa_hpl_v2, lswa_hps_sam_env0
from data.home import lsoup_hpd, lsoup_hpd_ehess
from data.home import lsoup_hps, lsoup_hps_203601, lsoup_hps_diffinits, lsoup_hps_env0_div2, lsoupswa_hps_0412
# lsoup_hpeoa_ehess, lsoup_hpeoa_5000, lsoup_hpeoa, lsoup_hpl_ehess,
from data.home import lsoup_hpl_topk0_5, lsoup2_hpl_samedata

from data.home import lsoup_hps_env0_div2_ermmixupcoral  # divacrossregul, divacrossrefulv2,
# from data6.home.combin import combinhome0_emvc_hpx_0412

from data64.pacs import pacs0_combin_erm_hps_soup_0420, pacs0_combin_erm_hps_swa_0420_v0, pacs0_combin_erm_hps_swa_0420, pacs0_combin_erm_hps_soup_0420_v1, pacs0_combin_erm_hps_swa_0422_579
from data64.pacs import pacs0_combin_erm_hps_swa_0422, pacs0_combin_erm_hps_swa_0423

# def clean_lsoup(lsoup):
#     return [
#         [
#             l
#             for l in m
#             if all(steps not in l["dirs"].split("_")
#                    for steps in ["0", "100", "200", "300"]) and l["netm"] < l["soup"] + 0.002 and l["netm"] > 0.775
#         ]
#         for m in lsoup
#     ]

# from data64.pacs import pacs0_combin_erm_hps_swa_0421
# pacs0_combin_erm_hps_swa_0421_lswa = clean_lsoup(pacs0_combin_erm_hps_swa_0421.lswa)

from data64.home.hessian import home0_combin_trainacc_erm_0426, home0_combin_hess_erm_0426, home0_combin_hess_erm_0428
from data64.home.hessian import home0_combin_trainacchess_samjz_0428, home0_combin_trainacchess_samhac_0428


def enrich():
    for l in home0_combin_trainacc_erm_0426.lsouplast[1]:
        found = False
        for j in home0_combin_hess_erm_0426.lsouplast[1]:
            if l["dirs"] == j["dirs"]:
                l.update(j)
                found = True
        if not found:
            print(f"Not found {l}")


def get_data_label_color_hess(key1, key2):

    data = []
    label = []

    def check_line(line, dl=1):
        if key1 + "soup" + key2 not in line:
            return False
        if dl is None:
            return True
        return line["length"] == dl

    data.append(
        [
            line[key1 + "soup" + key2] for c in merge(
                home0_combin_trainacc_erm_0426.lsouplast,
                home0_combin_trainacc_erm_0426.lsoup,
                home0_combin_hess_erm_0428.lsouplast,
            ) for line in c if check_line(line)
        ]
    )

    label.append('ERM')

    data.append(
        [
            line[key1 + "soupswa" + key2] for c in merge(
                home0_combin_trainacc_erm_0426.lsouplast, home0_combin_trainacc_erm_0426.lsoup,
                home0_combin_hess_erm_0428.lsouplast
            ) for line in c if check_line(line)
        ]
    )

    label.append('WA')

    if False:
        data.append(
            [
                line[key1 + "soup" + key2] for c in merge(
                    home0_combin_trainacchess_samjz_0428.lsouplast,
                    # home0_combin_trainacchess_samhac_0428.lsouplast
                ) for line in c if check_line(line)
            ]
        )
        label.append('SAM')
    if False:
        data.append(
            [
                line[key1 + "soupswa" + key2] for c in merge(
                    home0_combin_trainacchess_samjz_0428.lsouplast,
                    # home0_combin_trainacchess_samhac_0428.lsouplast
                ) for line in c if check_line(line)
            ]
        )
        label.append('SAM + WA')

    colors = cm.rainbow(np.linspace(0.3, 1, 4))[:len(label)]

    return data, label, colors


def plot_boxplot_hess(key1, key2, title="Hessian flatness in training", limits=None):

    enrich()
    fig, ax = plt.subplots(nrows=1, ncols=1)

    data, label, colors = get_data_label_color_hess(key1, key2)

    d_limits = {}
    if limits is not None:
        d_limits = {key2: limits}

    make_boxplot(
        fig, ax, data, colors, title, label, key=key2, limits=d_limits, color_median="black"
    )

    return fig


# def plot_samswa(key1, key2, title=None, limits=None):

#     enrich()
#     plt.rcParams["figure.figsize"] = (5, 5)
#     kwargs = dict(alpha=0.5, bins=25, density=True, stacked=True)

#     fig = plt.figure()

#     data, label, colors = get_data_label_color_hess(key1, key2)

#     for i in range(len(label)):
#         plt.hist(data[i], **kwargs, color=colors[i], label=label[i])

#     plt.gca().set_xlabel(title, fontsize="x-large")
#     plt.gca().set_ylabel('Frequency (%)', fontsize="x-large")
#     if limits:
#         plt.xlim(limits[0], limits[1])
#     plt.legend(loc="upper right", fontsize="x-large")
#     return fig


def plot_diversity_pacs(key="df", length=None, limits={}):

    plt.rcParams["figure.figsize"] = (5, 5)
    kwargs = dict(alpha=0.5, bins=25, density=True, stacked=True)

    def check_line(line, dl=length):
        if dl is None:
            return True
        return line["length"] == dl

    fig = plt.figure()
    keyname = "Feature" if key == "df" else "Prediction"

    data = []
    label = []


    data.append(
        [
            line[key]
            for c in
            merge(pacs0_combin_erm_hps_soup_0420.lsoup, pacs0_combin_erm_hps_soup_0420_v1.lsoup)
            for line in c
            if check_line(line)
        ]
    )
    label.append('Weights from different runs')

    # data.append(
    #     [
    #         line[key]
    #         for c in
    #         merge(
    #             pacs0_combin_erm_hps_swa_0420_v0.lswa, pacs0_combin_erm_hps_swa_0420.lswa, pacs0_combin_erm_hps_swa_0422_579.lswa
    #             )
    #         for line in c
    #         if check_line(line)
    #     ]
    # )
    label.append('Weights from one single run')

    data.append(
        [
            line[key] for c in
            # merge(pacs0_combin_erm_hps_swa_0420_v0.lswa, pacs0_combin_erm_hps_swa_0420.lswa)
            merge(
                # pacs0_combin_erm_hps_swa_0421_lswa
                pacs0_combin_erm_hps_swa_0422.lswa,
                pacs0_combin_erm_hps_swa_0423.lswa
            ) for line in c if check_line(line)
        ]
    )
    # label.append('One single run new')


    colors = cm.rainbow(np.linspace(0.3, 1, 4))
    colors = [colors[0], colors[2],  colors[1]]

    for i in range(len(label)):
        plt.hist(data[i], **kwargs, color=colors[i], label=label[i])

    plt.gca().set_xlabel(keyname + ' diversity', fontsize="x-large")
    plt.gca().set_ylabel('Frequency (%)', fontsize="x-large")
    if key in limits:
        plt.xlim(limits[key][0], limits[key][1])
    plt.legend(loc="upper right", fontsize="x-large")
    return fig



def plot_diversity_hp(key="df", length=None, limits={}, with_sam=False):

    plt.rcParams["figure.figsize"] = (5, 5)
    kwargs = dict(alpha=0.5, bins=25, density=True, stacked=True)

    def check_line(line, dl=length):
        if dl is None:
            return True
        return line["length"] == dl

    fig = plt.figure()
    keyname = "Feature" if key == "df" else "Prediction"

    data = []
    label = []


    if with_sam:
        data.append(
            [line[key] for c in merge(lswa_hps_sam_env0.lswa) for line in c if check_line(line)]
        )
        label.append('Weights from one SAM run')
    else:
        data.append(
            [
                line[key] for c in merge(
                    lsoup_hps_203601.lsoup_hp601,
                    # lsamedata_from,
                    lsoup_hps_env0_div2.lsamedata,
                    lsoup_hps.lsoup,
                    lsoup_hps_203601.lsoup_hp203,
                    lsoup_hps_env0_div2.ldiffdata,
                ) for line in c if check_line(line)
            ]
        )
        label.append('Weights from different runs')


    data.append(
        [
            line[key]
            for c in merge(lswa_hpd.l, l_nodrop.l, l_drop.l, l_hpeoa.l)
            for line in c
            if check_line(line)
        ]
    )
    if with_sam:
        label.append('Weights from one ERM run')
    else:
        label.append('Weights from one single run')


    colors = cm.rainbow(np.linspace(0.3, 1, 4))
    colors = [colors[0], colors[2]]

    for i in range(2):
        plt.hist(data[i], **kwargs, color=colors[i], label=label[i])

    plt.gca().set_xlabel(keyname + ' diversity', fontsize="x-large")
    plt.gca().set_ylabel('Frequency (%)', fontsize="x-large")
    if key in limits:
        plt.xlim(limits[key][0], limits[key][1])
    plt.legend(loc="upper right", fontsize="x-large")
    return fig


def plot_boxplot_full(key="df", keys=["swa", "soup", "same"], length=None, limits={}):

    def check_line(line, dl=length):
        if dl is None:
            return True
        return line["length"] == dl

    fig, ax = plt.subplots(
        nrows=len(keys), ncols=1, gridspec_kw={'height_ratios': [2, 4, 3][:len(keys)]}
    )

    keyname = "Feature" if key == "df" else "Prediction"

    if "swa" in keys:
        data = []
        label = []
        colors = []
        index = keys.index("swa")
        # data.append([line[key] for c in lswa_hpd.l for line in c if check_line(line)])
        # label.append('Default hp.')

        data.append(
            [
                line[key]
                for c in merge(lswa_hpd.l, l_nodrop.l, l_drop.l, l_hpeoa.l)
                for line in c
                if check_line(line)
            ]
        )
        label.append('Standard')

        data.append(
            [line[key] for c in merge(lswa_hpl.l, lswa_hpl_v2.l) for line in c if check_line(line)]
        )
        label.append('Extreme hyperparameters')

        # data.append(
        #     [line[key] for c in merge(lswa_hps_sam_env0.lswa) for line in c if check_line(line)]
        # )
        # label.append('SAM')

        colors.extend(cm.Reds(np.linspace(0.1, 1, len(label))))

        make_boxplot(
            fig,
            ax[index],
            data,
            colors,
            keyname + ' diversity in one single run',
            label,
            key,
            limits=limits
        ),

        ax[index].set_xticks([])

    if "soup" in keys:
        data = []
        label = []
        colors = []
        index = keys.index("soup")

        # data.append(
        #     [line[key] for c in merge(lsoup_hpd.lsoup, lsoup_hpd_ehess.lsoup) for line in c if check_line(line)]
        # )
        # label.append('Default hp.')

        # soup_hpeoa3 = [line[key] for c in lsoup_hpl_35.lsoup for line in c if check_line(line) and line["step"] == 3000]
        # plt.hist(soup_hpeoa3, **kwargs, color=colors_blues[2], label='dropout and various hp.s at epoch 3000')
        data.append(
            [
                line[key] for c in merge(
                    lsoup_hps_203601.lsoup_hp601,
                    # lsamedata_from,
                    lsoup_hps_env0_div2.lsamedata
                ) for line in c if check_line(line)
            ]
        )
        label.append('Standard')

        data.append(
            [
                line[key]
                for c in merge(lsoup2_hpl_samedata.lsoup)
                for line in c
                if line.get("step", 5000) == 5000 and check_line(line)
            ]
        )
        label.append('Extreme hyperparameters')

        data.append(
            [
                line[key]
                for c in merge(lsoup_hps_diffinits.lsoup)
                for line in c
                if line.get("step", 5000) == 5000 and check_line(line)
            ]
        )
        label.append('Different classifier inits')

        data.append(
            [
                line[key] for c in merge(
                    # lsoup_hps.lsoup,
                    # lsoup_hps_203601.lsoup_hp203,
                    lsoup_hps_env0_div2.ldiffdata,
                    # lsoup_hpeoa.lsoup, lsoup_hpeoa_ehess.lsoup, lsoup_hpeoa_5000.lsoup,
                ) for line in c if check_line(line, dl=2)
            ]
        )
        label.append('Different data')

        # data.append(
        #     [
        #         line[key]
        #         for c in merge(lsoupswa_hps_0412.lsoupswa)
        #         for line in c
        #         if line.get("step", 5000) == 5000 and check_line(line)
        #     ]
        # )
        # label.append('Moving averages')

        # data.append(
        #     [
        #         line[key]
        #         for c in merge(lsoup_hpl_topk0_5.lsoup)
        #         for line in c
        #         if line.get("step", 5000) == 5000 and check_line(line)
        #     ]
        # )
        # label.append('Full hp. (different splits)')

        colors.extend(cm.Blues(np.linspace(0.1, 1, len(label))))

        make_boxplot(
            fig,
            ax[index],
            data,
            colors,
            keyname + ' diversity from different runs',
            label,
            key,
            limits=limits
        )
        ax[index].set_xticks([])

    if "same" in keys:
        data = []
        label = []
        colors = []
        index = keys.index("same")

        data.append(
            [
                line[key]
                for line in lsoup_hps_env0_div2_ermmixupcoral.lsoup[2]
                # divacrossregul.l + divacrossrefulv2.l
                if line["dirs"] in ["erm_mixup", "mixup_erm"] and check_line(line)
            ]
        )
        label.append(f'ERM and Mixup')

        # data.append(
        #     [
        #         line[key]
        #         for line in lsoup_hps_env0_div2_ermmixupcoral.lsoup[2]
        #         # divacrossregul.l + divacrossrefulv2.l
        #         if line["dirs"] in ["mixup_mixup", "mixupv_mixupv"] and check_line(line)
        #     ]
        # )
        # label.append(f'Mixup and Mixup')

        data.append(
            [
                line[key]
                for line in lsoup_hps_env0_div2_ermmixupcoral.lsoup[2]
                # divacrossregul.l + divacrossrefulv2.l
                if line["dirs"] in ["erm_coral", "coral_erm"] and check_line(line)
            ]
        )
        label.append(f'ERM and Coral')

        # data.append(
        #     [
        #         line[key]
        #         for line in lsoup_hps_env0_div2_ermmixupcoral.lsoup[2]
        #         # divacrossregul.l + divacrossrefulv2.l
        #         if line["dirs"] in ["coral_coral"] and check_line(line)
        #     ]
        # )
        # label.append(f'Coral and Coral')

        data.append(
            [
                line[key]
                for line in lsoup_hps_env0_div2_ermmixupcoral.lsoup[2]
                # divacrossregul.l + divacrossrefulv2.l
                if line["dirs"] in ["coral_mixup", "mixup_coral", "mixupv_coral", "mixupv_coral"]
                and check_line(line)
            ]
        )
        label.append(f'Mixup and Coral')

        colors.extend(cm.YlOrBr(np.linspace(0.3, 1, len(label))))

        make_boxplot(
            fig,
            ax[index],
            data,
            colors,
            keyname + ' diversity from different runs with different objectives',
            label,
            key,
            limits=limits
        )

    return fig


def make_figs():
    fig_df = plot_boxplot(key="df", keys=["swa", "soup", "same"], length=2)
    save_fig(fig_df, "boxplot_df.png")
    fig_dr = plot_boxplot(key="dr", keys=["swa", "soup", "same"], length=2)
    save_fig(fig_dr, "boxplot_dr.png")


def make_boxplot(fig, ax, data, colors, title, label, key, limits={}, color_median="black"):
    bp = ax.boxplot(data[::-1], vert=0, showmeans=True, widths=0.5, showfliers=False)

    for median in bp['medians']:
        median.set(color=color_median, linewidth=2)

    import matplotlib.patches as mpatches
    for box, color in zip(bp['boxes'], colors[::-1]):
        patch = mpatches.PathPatch(box.get_path(), color=color)
        ax.add_artist(patch)

    ax.set_xlabel(title)
    ax.set_yticklabels(label[::-1])
    if key in limits:
        ax.set_xlim(limits[key][0], limits[key][1])


# def make_boxplot(fig, ax, data, colors, title, label, key):
#     bp = ax.violinplot(data[::-1], vert=0)

#     # for median in bp['medians']:
#     #     median.set(color='red', linewidth=2)

#     # import matplotlib.patches as mpatches
#     # for box, color in zip(bp['boxes'], colors[::-1]):
#     #     patch = mpatches.PathPatch(box.get_path(), color=color)
#     #     ax.add_artist(patch)

#     ax.set_title(title)
#     ax.set_yticklabels(label[::-1])
#     if key == "dr":
#         ax.set_xlim(0.4, 0.8)
#     else:
#         ax.set_xlim(0.05, 0.7)


def add_div_losses(key, dataaug1, dataaug2, kwargs, color):
    div_dataaug = [
        l[key]
        for l in divacrossregul.l + divacrossrefulv2.l
        if dataaug1 in l["dirs"] and dataaug2 in l["dirs"]
    ]
    plt.hist(
        div_dataaug,
        **kwargs,
        color=color,
    )


import seaborn as sns
list_methods = ['erm', 'mixup', 'coral', 'gdro', 'fishr']
from data.home import divacrossregul, divacrossrefulv2
l = divacrossregul.l + divacrossrefulv2.l


def plot_diversity_aux(l, labels, key, length=None, limits={}):

    plt.rcParams["figure.figsize"] = (5, 5)
    kwargs = dict(alpha=0.5, bins=25, density=True, stacked=True)

    def check_line(line):
        if length is None:
            return True
        return line["length"] == length

    fig = plt.figure()
    keyname = "Feature" if "divf" in key else "Prediction"

    data = []

    for c in l:
        data.append(
            [
                line[key] for line in c if check_line(line)
            ]
        )

    colors = cm.rainbow(np.linspace(0.3, 1, len(labels)))

    for i in range(len(labels)):
        plt.hist(data[i], **kwargs, color=colors[i], label=labels[i])

    plt.gca().set_xlabel(keyname + ' diversity', fontsize="x-large")
    plt.gca().set_ylabel('Frequency (%)', fontsize="x-large")
    if key in limits:
        plt.xlim(limits[key][0], limits[key][1])
    plt.legend(loc="upper right", fontsize="x-large")
    return fig


def plot_diversity_reguls(l, key="dr", plot=False):
    if plot:
        plt.rcParams["figure.figsize"] = (5, 5)
    kwargs = dict(alpha=0.5, bins=10, density=True, stacked=True)

    keyname = "Feature" if key == "df" else "Prediction"
    colors_reds = cm.Reds(np.linspace(0, 1, 5 * 5))

    mat = [[] for _ in range(len(list_methods))]
    for i in range(5):
        for j in range(i, 5):
            mi = list_methods[i]
            mj = list_methods[j]
            lines = [line[key] for line in l if mi in line["dirs"] and mj in line["dirs"]]
            mean_lines = np.mean(lines)

            label = mi + "_" + mj + ": " + str(mean_lines)
            if plot:
                plt.hist(lines, **kwargs, color=colors_reds[5 * i + j], label=label)
            mat[i].append(mean_lines)
            if i != j:
                mat[j].append(mean_lines)
    if plot:
        plt.gca().set(title=keyname + ' diversity', ylabel='Count')
        # plt.xlim(50, 75)
        plt.legend()
    return mat


def plot_matrix(mat):
    fig = plt.figure()
    mask = np.ones_like(mat)
    mask[np.triu_indices_from(mask)] = False
    with sns.axes_style("white"):
        ax = sns.heatmap(
            mat,
            mask=mask,
            # vmax=.5,
            square=True,
            cmap="YlGnBu",
            fmt='.3g',
            annot=True,
            xticklabels=list_methods,
            yticklabels=list_methods
        )
    return fig
