from datafb.transfer.fb import home_initdn_1003, home_1003, pacs_1003, terra_1012, terra_1013, terra_1014_iwild_1o, terra_1014_iwild_2k15k, home_ent_1014, terra_iwild_1015, terra_iwild_1015_oracle, terra_iwild_1017_top1, terra_iwildf_1017, home_dnf_1017, homefdnf_terrafiwildf_1018, terrafiwildf_1019
from datafb.transfer import home_fb_1002_top1, pacs_fb_1002_top1, terradn, home_pacs_terra_top1oracle, vlcs_1010
from datafb.transfer import home3dn_tr, pacs_fb_1001, home_fb_1001, home_fb_notransfer, home_fb, home0dn_tr, home2dn_transferrobustv3, home2dn_transferrobustv2, home2dn_transferrobust
from datafb.transfer import home2dn_transfer, homepacs_mixed, home0top1_patch, home3dn_patch, pacs_from_imagenet, home_from_imagenet, home0123dn, pacs0123dn, pacs3, home2dn, pacs1dn, pacs2dn, home3dn, pacs3dn, home0, home0top1, home0few, home0dn, home1, home1dn, pacs3few, pacs0, pacs0dn

from domainbed.codeplot.plot import get_x, fit_and_plot, save_fig
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

import matplotlib as mpl


def plot_robust_transfer(
    ll_m,
    key1,
    fig=None,
    ax1=None,
    orders="2",
    set_x_label=True,
    dict_key_to_label={},
    key_axis1="acc",
    labels=None,
    key_annot=None,
    legends=None,
    title=None,
    dict_key_to_limit={},
    loc="lower right"
):

    if labels is None:
        labels = [str(i) for i in range(len(ll_m))]
    if ax1 is None:
        fig, ax1 = plt.subplots()
    else:
        assert ax1 is not None
    if set_x_label:
        ax1.set_xlabel(dict_key_to_label.get(key1, key1), fontsize="x-large")
    colors = cm.rainbow(np.linspace(0.0, 0.9, len(ll_m)))
    ax1.set_ylabel(dict_key_to_label.get(key_axis1, key_axis1), fontsize="x-large")
    ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.1f}'))

    def plot_with_int(l, ax, color, label, key2, marker, linestyle, order):
        t = get_x(l, key1)
        if t == []:
            return

        l = [ll for ll in l if key2 in ll]
        try:
            ax.scatter(get_x(l, key1), get_x(l, key2), color=color, label=label, marker=marker)
        except:
            print(color)
            ax.scatter(get_x(l, key1), get_x(l, key2), color="red", label=label, marker=marker)
        if key_annot:
            n = len(get_x(l, key1))
            for i in range(n):
                ax.annotate(
                    get_x(l, key_annot)[i], (get_x(l, key1)[i], get_x(l, key2)[i]), color=color
                )

        if order != "no":
            fit_and_plot(
                key1, key2, l, order=order, label=label, color=color, ax=ax, linestyle=linestyle
            )

    for i, l_m in enumerate(ll_m):
        plot_with_int(
            l_m,
            ax1,
            color=colors[i],
            label=dict_key_to_label.get(i, labels[i]),
            key2=key_axis1,
            marker="." if i % 2 else "x",
            order=orders,
            linestyle="-"
        )
    if key_axis1 in dict_key_to_limit:
        ax1.set_ylim(dict_key_to_limit[key_axis1])

    if key1 in dict_key_to_limit:
        ax1.set_xlim(dict_key_to_limit[key1])

    if loc != "no":
        # legend1 = ax1.legend(fontsize="x-large", loc=loc)
        legend1 = ax1.legend(fontsize="small", loc=loc)
        title1 = dict_key_to_label.get(key_axis1)
        if legends:
            title1 = legends[0]
        legend1.set_title(title1, prop={'size': 15})
    if title:
        ax1.suptitle(title, fontsize=20)


dict_dataset_to_domain_names = {
    "terra": ["L100", "L38", "L43", "L46"],
    "vlcs": ["Caltech101", "LabelMe", "SUN09", "VOC2007"],
    "pacs": ["Art", "Cartoon", "Photo", "Sketch"],
    "home": ["Art", "Clipart", "Product", "Photo"]
}


def get_values_from_list(data, y="ood", metric="best", what="acc"):
    labels = list(data.keys())
    l = [data[label] for label in labels]
    if y == "ood":
        key_axis1 = what
    else:
        assert isinstance(y, str)
        key_axis1 = "env_" + y + "_" + what

    dict_best_value = {}
    for i, l_m in enumerate(l):

        def verify_metric(ll):
            if metric in ["best", "bestoracle", "last"]:
                return ll["step"] == metric
            if metric == "lasts":
                try:
                    return ll["step"] > 4500
                except:
                    return False
            if isinstance(metric, int):
                return ll["step"] == int(metric)

        l_best = [
            ll for ll in l_m if verify_metric(ll)
        ]
        max_label = np.mean(sorted(get_x(l_best, key_axis1), reverse=True)) * 100
        dict_best_value[labels[i]] = max_label
    return dict_best_value


def compare_auxiliary(dataset, pts, metric, plot_only_ood=True, orders=2, figsize=(14, 8), dict_key_to_label=None, labels=None, what="acc", list_limits=None, save_path=None, loc="upper right"):
    list_list_l = []
    list_domain_names = dict_dataset_to_domain_names[dataset]

    plt.rcParams["figure.figsize"] = figsize
    kwargs = {"legends": ["Approach:"], "labels": pts if labels is None else labels, "orders": orders, "loc": loc}
    if plot_only_ood:
        fig, axes = plt.subplots(2, 2)
    for domain in range(4):

        def get_dict_cleanx(x):
            lambda_wrt_dn = float(x.split("_")[0])
            norm_lambda = float(lambda_wrt_dn / (1 + lambda_wrt_dn))
            return "{norm_lambda:.2f}".format(norm_lambda=norm_lambda)

        dict_l = [{} for _ in pts]
        list_y = [
            y if y != str(domain) + "_out" else "ood" for y in ["0_out", "1_out", "2_out", "3_out"]
        ] + [str(domain) + "_out"]

        for i, pt in enumerate(pts):
            data = find_data(dataset, domain, pt)
            for y in list_y:
                dict_best_value = get_values_from_list(data, y=y, metric=metric, what=what)
                for x in dict_best_value:
                    if x not in dict_l[i]:
                        dict_l[i][x] = {
                            "lambda": float(get_dict_cleanx(x)),
                            "clean": r"$\lambda=$" + str(get_dict_cleanx(x))
                        }
                    dict_l[i][x][y] = dict_best_value[x]

        list_l = [
            [
                dict_l[i][x]
                for x in sorted(dict_l[i], key=lambda x: dict_l[i][x]["lambda"], reverse=True)
            ]
            for i, pt in enumerate(pts)
        ]
        list_list_l.append(list_l)
        list_domain_names = dict_dataset_to_domain_names[dataset]
        if dict_key_to_label is None:
            dict_key_to_label = {}
        dict_key_to_label.update({
            "lambda": r"$\lambda$",
            "ood": "OOD " + list_domain_names[domain],
            "0_out": "IID " + list_domain_names[0],
            "1_out": "IID " + list_domain_names[1],
            "2_out": "IID " + list_domain_names[2],
            "3_out": "IID " + list_domain_names[3],
        })
        if not plot_only_ood:
            _, axes = plt.subplots(2, 2)

            for i, y in enumerate(list_y[:4]):

                ax = axes[i // 2][i % 2]
                plot_robust_transfer(
                    list_l,
                    "lambda",
                    ax1=ax,
                    set_x_label=i // 2,
                    key_axis1=y,
                    dict_key_to_label=dict_key_to_label,
                    **kwargs
                )
        else:
            y = "ood"
            dict_key_to_limit = {}
            if list_limits is not None:
                dict_key_to_limit[y] = list_limits[domain]
            plot_robust_transfer(
                list_l,
                "lambda",
                set_x_label=domain // 2,
                ax1=axes[domain // 2][domain % 2],
                key_axis1=y,
                dict_key_to_label=dict_key_to_label,
                dict_key_to_limit=dict_key_to_limit,
                **kwargs
            )
    if save_path is not None:
        save_fig(fig, save_path, folder=None)
    return list_list_l


key_acc_val = "env_0_out_acc+env_1_out_acc+env_2_out_acc+env_3_out_acc/4"
key_acc_val0 = "env_1_out_acc+env_2_out_acc+env_3_out_acc/3"
key_acc_val1 = "env_0_out_acc+env_2_out_acc+env_3_out_acc/3"
key_acc_val2 = "env_0_out_acc+env_1_out_acc+env_3_out_acc/3"
key_acc_val3 = "env_0_out_acc+env_1_out_acc+env_2_out_acc/3"

key_acc_train = "env_0_in_acc+env_1_in_acc+env_2_in_acc+env_3_in_acc/4"
key_acc_train0 = "env_1_in_acc+env_2_in_acc+env_3_in_acc/3"
key_acc_train1 = "env_0_in_acc+env_2_in_acc+env_3_in_acc/3"
key_acc_train2 = "env_0_in_acc+env_1_in_acc+env_3_in_acc/3"
key_acc_train3 = "env_0_in_acc+env_1_in_acc+env_2_in_acc/3"

dict_clean_labels = {
    "0_0": "100%",
    "0_20": "100% robust",
    "0top1_0": "100% top1",
    "0top1_20": "100% robust top1",
    "20_0": "50%",
    "20_20": "50% robust",
    "20top1_0": "50% top1",
    "20top1_20": "50% robust top1",
    "40_0": "33%",
    "40_20": "33% robust",
    "im_0": "0%",
    "im_20": "0% robust",
    "im_0_20": "0% patch",
}

pacs0dn.l["im_0"] = pacs0.l["im_0"]
pacs0dn.l["im_20"] = pacs0.l["im_20"]

pacs3dn.l["im_0"] = pacs3.l["im_0"]
pacs3dn.l["im_20"] = pacs3.l["im_20"]
pacs0dn.l["im_0_20"] = pacs_from_imagenet.l["env0_im0_dnf20"]
pacs1dn.l["im_0_20"] = homepacs_mixed.l["penv1_im_0_dnf20"]

pacs2dn.l["im_0_20"] = pacs_from_imagenet.l["env2_im0_dnf20"]
pacs3dn.l["im_0_20"] = pacs_from_imagenet.l["env3_im0_dnf20"]
pacs0123dn.l["im_0_20"] = pacs_from_imagenet.l["env0123_im0_dnf20"]

home0dn.l["im_0"] = home0.l["im_0"]
home0dn.l["im_20"] = home0.l["im_20"]
home1dn.l["im_0"] = home1.l["im_0"]
home1dn.l["im_20"] = home1.l["im_20"]

home0dn.l["im_0_20"] = home_from_imagenet.l["env0_im0_dnf20"]
home0dn.l["im_10_10"] = home_from_imagenet.l["env0_im10_dnf10"]
home0dn.l["im_0_200"] = homepacs_mixed.l["env0_im0_dnf200"]

home1dn.l["im_0_20"] = home_from_imagenet.l["env1_im0_dnf20"]
home1dn.l["im_0_p20"] = home_from_imagenet.l["env1_im0_dn20"]
home1dn.l["im_0_p20r20"] = home_from_imagenet.l["env1_im0_dn20r20"]
home1dn.l["im_0_200"] = homepacs_mixed.l["env1_im0_dnf200"]

home2dn.l["im_0_20"] = homepacs_mixed.l["env2_im0_dnf20"]

home3dn.l["im_0_20"] = home_from_imagenet.l["env3_im0_dnf20"]
home3dn.l["im_10_20"] = home3dn_patch.l["env0_im10_dnf20"]
home3dn.l["im_10_20"] = home3dn_patch.l["env0_im10_dnf20"]
home3dn.l["im_20_10"] = home3dn_patch.l["env0_im20_dnf10"]
home3dn.l["im_10_10"] = home3dn_patch.l["env0_im10_dnf10"]
home3dn.l["im_20_20"] = home3dn_patch.l["env0_im20_dnf20"]
# dict_keys(['env0_im10_dnf20', 'env0_im20_dnf10', 'env0_im10_dnf10', 'env0_im20_dnf20'])
home0123dn.l["im_0_20"] = home_from_imagenet.l["env0123_im0_dnf20"]

home2dn_transfer.l["im_tdn0"] = home2dn.l["im_0"]
home2dn_transferrobust.l["im_tdn0"] = home2dn.l["im_0"]

for key, value in home2dn_transferrobustv3.l.items():
    home2dn_transferrobustv2.l[key] = value


def create_from_fb(fb_l):
    list_clean = [{}, {}, {}, {}]
    for key, value in fb_l.items():
        if value == []:
            continue
        env = int(key.split("_")[0])
        list_clean[env][str(float(key.split("_")[1]) / 100)] = value
    return list_clean


list_pacsfb = create_from_fb(pacs_fb_1001.l)
list_pacstop1fb = create_from_fb(pacs_fb_1002_top1.l)



list_homefb = create_from_fb(home_fb_1001.l)
list_hometop1fb = create_from_fb(home_fb_1002_top1.l)
for key, value in home_fb.l.items():
    if value == []:
        continue
    env = int(key.split("_")[0])
    list_homefb[env][key.split("_")[1]] = value
for i in range(4):
    list_homefb[i]["0.0"] = home_fb_notransfer.l[str(i)]

def processnew_file(fb_l):
    output_dict = {}
    for key, value in fb_l.items():
        if len(key.split("_"))==5:
            env, lam, rft, topk, pt = key.split("_")
        else:
            env, lam, topk, pt = key.split("_")
            rft = "0"
        env = int(env)
        lam = str(float(lam))
        if topk == "1o":
            topk = "1"
        if topk not in output_dict:
            output_dict[topk] = [{}, {}, {}, {}]
        if lam not in output_dict[topk][int(env)]:
            output_dict[topk][int(env)][lam] = []
        output_dict[topk][int(env)][lam].extend(value)
    return output_dict

dict_homednf = processnew_file(home_dnf_1017.l)
dict_homefdnf = processnew_file(homefdnf_terrafiwildf_1018.dict_l["home"])

# dict_terrafiwildf = processnew_file(homefdnf_terrafiwildf_1018.dict_l["terra"])
dict_terrafiwildf = processnew_file(terrafiwildf_1019.l)



for key, value in home_1003.l.items():
    env, lam, topk = key.split("_")
    env = int(env)
    lam = str(int(lam) / 100)
    if topk == "0":
        assert lam not in list_homefb[env]
        list_homefb[int(env)][lam] = value
    else:
        assert topk == "1"
        assert lam not in list_hometop1fb[env]
        list_hometop1fb[int(env)][lam] = value

for key, value in pacs_1003.l.items():
    env, lam, topk = key.split("_")
    env = int(env)
    lam = str(int(lam) / 100)
    if topk == "0":
        assert lam not in list_pacsfb[env]
        list_pacsfb[int(env)][lam] = value
    else:
        assert topk == "1"
        assert lam not in list_pacstop1fb[env]
        list_pacstop1fb[int(env)][lam] = value

list_terratop1fb = [{}, {}, {}, {}]
list_terrafb = [{}, {}, {}, {}]
for key, value in terradn.l.items():
    env, lam, topk = key.split("_")
    env = int(env)
    lam = str(int(lam) / 100)
    if topk == "0":
        assert lam not in list_terrafb[env]
        list_terrafb[int(env)][lam] = value
    else:
        assert topk == "1"
        assert lam not in list_terratop1fb[env]
        list_terratop1fb[int(env)][lam] = value


list_terratop1fbnatu = [{}, {}, {}, {}]
list_terrafbnatu = [{}, {}, {}, {}]
list_terratop1fbiwild = [{}, {}, {}, {}]
list_terrafbiwild = [{}, {}, {}, {}]
for key, value in terra_1012.l.items():
    env, lam, topk, pt = key.split("_")
    env = int(env)
    if lam == "0005":
        lam = "0.005"
    else:
        lam = str(int(lam) / 100)
    if pt == "iwild":
        if topk == "0":
            assert lam not in list_terrafbiwild[env]
            list_terrafbiwild[int(env)][lam] = value
        else:
            assert topk == "1"
            assert lam not in list_terratop1fbiwild[env]
            list_terratop1fbiwild[int(env)][lam] = value
    else:
        if topk == "0":
            assert lam not in list_terrafbnatu[env]
            list_terrafbnatu[int(env)][lam] = value
        else:
            assert topk == "1"
            assert lam not in list_terratop1fbnatu[env]
            list_terratop1fbnatu[int(env)][lam] = value

list_homedn_ent = [{}, {}, {}, {}]
list_homedn_ent_top1 = [{}, {}, {}, {}]
for key, value in home_ent_1014.l.items():
    env, lam, topk = key.split("_")
    if topk == "1":
        list_homedn_ent_top1[int(env)][lam] = value
    else:
        list_homedn_ent[int(env)][lam] = value

dict_list_terrafbwildrft = {}
for key, value in list(terra_iwild_1015.l.items()) + list(terra_iwild_1017_top1.l.items()):
    env, lam, rft, topk, pt = key.split("_")
    env = int(env)
    lam = float(lam)
    if (1 + lam - float(rft)) <= 0 or float(rft)>1.5:
        continue
    else:
        keyrft = str(float(rft)) + "t" + topk
        if keyrft not in dict_list_terrafbwildrft:
            dict_list_terrafbwildrft[keyrft] = [{}, {}, {}, {}]
        key = (1 + lam - float(rft)) * lam / (1 + lam)
        if str(key) not in dict_list_terrafbwildrft[keyrft][int(env)]:
            dict_list_terrafbwildrft[keyrft][int(env)][str(key)]  = []
        dict_list_terrafbwildrft[keyrft][int(env)][str(key)].extend(value)

for key, value in terra_iwild_1015_oracle.l.items():
    env, lam, rft, topk, pt = key.split("_")
    env = int(env)
    lam = float(lam)
    if (1 + lam - float(rft)) <= 0 or float(rft)>1.5:
        continue
    else:
        keyrft = rft + "t" + topk
        if keyrft not in dict_list_terrafbwildrft:
            dict_list_terrafbwildrft[keyrft] = [{}, {}, {}, {}]
        key = str(float((1 + lam - float(rft)) * lam / (1 + lam)))
        if key not in dict_list_terrafbwildrft[keyrft][int(env)]:
            dict_list_terrafbwildrft[keyrft][int(env)][str(key)] = []
        dict_list_terrafbwildrft[keyrft][int(env)][str(key)].extend(value)


dict_list_terraiwildf = {}
for key, value in list(terra_iwildf_1017.l.items()):
    env, lam, rft, topk, pt = key.split("_")
    env = int(env)
    lam = float(lam)
    if (1 + lam - float(rft)) <= 0 or float(rft) > 1.5:
        continue
    else:
        assert float(rft) == 0
        #  str(float(rft)) + "t" +
        keyrft = topk
        if keyrft not in dict_list_terraiwildf:
            dict_list_terraiwildf[keyrft] = [{}, {}, {}, {}]
        key = (1 + lam - float(rft)) * lam / (1 + lam)
        # key = str(lam)
        # key = lam
        dict_list_terraiwildf[keyrft][int(env)][str(key)] = value



dict_list_terrafbiwild = {}
for key, value in list(terra_1014_iwild_1o.l.items()) + list(terra_1014_iwild_2k15k.l.items()):
    env, lam, rft, topk, pt = key.split("_")
    env = int(env)
    if lam == "0005":
        lam = "0.005"
    else:
        lam = str(int(lam) / 100)
    if rft == "000":
        rft = ""
    elif rft == "0005":
        rft = "0.005"
    else:
        rft = str(int(rft) / 100)
    key = pt + topk + rft

    if key not in dict_list_terrafbiwild:
        dict_list_terrafbiwild[key] = [{}, {}, {}, {}]
    lam = float(lam)
    # lam = (1 + lam - float(rft)) * lam / (1 + lam)
    dict_list_terrafbiwild[key][int(env)][str(lam)] = value



dict_list_terrafbiwildrft = {}
for key, value in terra_1013.l.items():
    env, lam, rft, topk, pt = key.split("_")
    env = int(env)
    if lam == "0005":
        lam = "0.005"
    else:
        lam = str(int(lam) / 100)
    if rft == "0005":
        rft = "0.005"
    else:
        rft = str(int(rft) / 100)
    if rft not in dict_list_terrafbiwildrft:
        dict_list_terrafbiwildrft[rft] = [{}, {}, {}, {}]
    lam = float(lam)
    key = (1+lam-float(rft)) * lam/(1+lam)
    dict_list_terrafbiwildrft[rft][int(env)][str(key)] = value

# oracle selection
list_hometop1ofb = [{}, {}, {}, {}]

list_pacstop1ofb = [{}, {}, {}, {}]

list_terratop1ofb = [{}, {}, {}, {}]

for key, value in home_pacs_terra_top1oracle.l.items():
    dataset, env, lam = key.split("_")
    env = int(env)
    lam = str(int(lam) / 100)
    if dataset == "home":
        list_hometop1ofb[int(env)][lam] = value
    elif dataset == "pacs":
        list_pacstop1ofb[int(env)][lam] = value
    elif dataset == "terra":
        list_terratop1ofb[int(env)][lam] = value
    else:
        raise ValueError("invalid dataset")

list_vlcstop1fb = [{}, {}, {}, {}]
list_vlcsfb = [{}, {}, {}, {}]
list_vlcstop1ofb = [{}, {}, {}, {}]
list_vlcstop1fbnatu = [{}, {}, {}, {}]
list_vlcsfbnatu = [{}, {}, {}, {}]
list_vlcstop1ofbnatu = [{}, {}, {}, {}]
for key, value in vlcs_1010.l.items():
    env, lam, topk, pt = key.split("_")
    env = int(env)
    lam = str(int(lam) / 100)
    if pt == "dn":
        if topk == "0":
            assert lam not in list_vlcsfb[env]
            list_vlcsfb[int(env)][lam] = value
        elif topk == "1":
            assert lam not in list_vlcstop1fb[env]
            list_vlcstop1fb[int(env)][lam] = value
        else:
            list_vlcstop1ofb[int(env)][lam] = value
    elif pt == "natu":
        if topk == "0":
            assert lam not in list_vlcsfbnatu[env]
            list_vlcsfbnatu[int(env)][lam] = value
        elif topk == "1":
            assert lam not in list_vlcstop1fbnatu[env]
            list_vlcstop1fbnatu[int(env)][lam] = value
        else:
            list_vlcstop1ofbnatu[int(env)][lam] = value
    else:
        print(pt)


list_home_idn = [
    {
        key.split("_")[0]: value
        for key, value in home_initdn_1003.l.items()
        if key.split("_")[-1] == "0"
    }, None, None, None
]
list_home_idn[0]["100000"] = list_homefb[0]["0.0"]
list_home_idntop1 = [
    {
        key.split("_")[0]: value
        for key, value in home_initdn_1003.l.items()
        if key.split("_")[-1] == "1"
    }, None, None, None
]
list_home_idntop1[0]["100000"] = list_hometop1fb[0]["0.0"]



list_home = [home0, home1, None, None]
list_homedn = [home0dn, home1dn, home2dn, home3dn, home0123dn]
list_homedntop1 = [home0top1_patch, None, None, None]
list_homednt = [None, None, home2dn_transfer, None]
list_homedntr = [None, None, home2dn_transferrobust, None]
list_homedntr2 = [home0dn_tr, None, home2dn_transferrobustv2, home3dn_tr]

list_pacs = [pacs0, None, None, pacs3]
list_pacsdn = [pacs0dn, pacs1dn, pacs2dn, pacs3dn, pacs0123dn]


def find_data(dataset, domain, pt):
    if dataset == "pacs":
        # if pt == "dn":
        #     return list_pacsdn[domain]
        if pt == "dn":
            return list_pacsfb[domain]
        elif pt == "dntop1":
            return list_pacstop1fb[domain]
        elif pt == "dntop1o":
            return list_pacstop1ofb[domain]
        else:
            raise ValueError(pt)
    elif dataset == "home":
        # if pt == "dn":
        #     return list_homedn[domain].l
        # elif pt == "dnt":
        #     return list_homednt[domain].l
        # elif pt == "dntop1":
        #     return list_homedntop1[domain].l
        # elif pt == "dntr":
        #     return list_homedntr[domain].l
        # elif pt == "dntrv2":
        #     return list_homedntr2[domain].l
        if pt == "dn":
            return list_homefb[domain]
        elif pt == "dnent":
            return list_homedn_ent[domain]
        elif pt == "dnent1":
            return list_homedn_ent_top1[domain]
        elif pt == "dntop1":
            return list_hometop1fb[domain]
        elif pt == "dntop1o":
            return list_hometop1ofb[domain]
        elif pt == "idn":
            # for inter-training
            return list_home_idn[domain]
        elif pt == "idntop1":
            # for inter-training
            return list_home_idntop1[domain]
        elif pt.startswith("fdnf"):
            # when trained with frozen classifier, on home and domainent
            topk = pt.split("_")[1]
            return dict_homefdnf[topk][domain]
        elif pt.startswith("dnf"):
            # when trained with frozen classifier on domainnet
            topk = pt.split("_")[1]
            return dict_homednf[topk][domain]
        else:
            # return list_home[domain]
            raise ValueError(pt)
    elif dataset == "terra":
        if pt == "dn":
            return list_terrafb[domain]
        elif pt == "dntop1":
            return list_terratop1fb[domain]
        elif pt == "dntop1o":
            return list_terratop1ofb[domain]
        elif pt == "natu":
            return list_terrafbnatu[domain]
        elif pt == "natutop1":
            return list_terratop1fbnatu[domain]
        elif pt == "natutop1o":
            raise ValueError(pt)
            return list_terratop1ofbnatu[domain]
        elif pt == "iwild":
            # first experiments with few data
            return list_terrafbiwild[domain]
        elif pt == "iwildtop1":
            # first experiments with few data
            return list_terratop1fbiwild[domain]
        elif pt.startswith("iwildf"):
            # when trained with frozen classifier on iwild
            topk = pt.split("_")[1]
            return dict_list_terraiwildf[topk][domain]
        elif pt.startswith("fiwildf"):
            # when trained with frozen classifier, on home and iwild
            topk = pt.split("_")[1]
            return dict_terrafiwildf[topk][domain]
        elif pt.startswith("wildrft"):
            rft = pt.split("_")[1]
            return dict_list_terrafbwildrft[rft][domain]
        elif pt.startswith("wild"):
            # varying number of steps in iwild
            return dict_list_terrafbiwild[pt][domain]
        elif pt.startswith("iwildrft"):
            # only last step
            rft = pt.split("_")[1]
            assert rft in dict_list_terrafbiwildrft
            return dict_list_terrafbiwildrft[rft][domain]
        else:
            raise ValueError(pt)
    elif dataset == "vlcs":
        if pt == "dn":
            return list_vlcsfb[domain]
        elif pt == "dntop1":
            return list_vlcstop1fb[domain]
        elif pt == "dntop1o":
            return list_vlcstop1ofb[domain]
        elif pt == "natu":
            return list_vlcsfbnatu[domain]
        elif pt == "natutop1":
            return list_vlcstop1fbnatu[domain]
        elif pt == "natutop1o":
            return list_vlcstop1ofbnatu[domain]
        else:
            raise ValueError(pt)
    else:
        raise ValueError()
