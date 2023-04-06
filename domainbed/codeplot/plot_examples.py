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
