dict_key_to_label = {"df": "Div. feats", "dr": "Div. preds.", "hess": "Flatness", "soup": "Acc.", "net": "Ens Acc."}



def plot_key(key1, key2, order=1):

    plt.xlabel(dict_key_to_label.get(key1, key1))
    plt.ylabel(dict_key_to_label.get(key2, key2))

    def plot_with_int(l, color, label):
        t = get_x(l, key1)
        if t == []:
            return
        if order == 1:
            m, b = np.polyfit(get_x(l, key1), get_x(l, key2), 1)
            plt.plot(get_x(l, key1), m * np.array(get_x(l, key1)) + b, color=color, label=label +": " + "{:.0f}".format(m*1000))
            plt.scatter(get_x(l, key1), get_x(l, key2), color=color)
        elif order == 2:
            m2, m1, b = np.polyfit(get_x(l, key1), get_x(l, key2), 2)
            get_x1_sorted = sorted(get_x(l, key1))
            preds = m2 * np.array(get_x1_sorted)**2 + m1 * np.array(get_x1_sorted) + b
            plt.plot(get_x1_sorted, preds, color=color)# label="int."+label)
            plt.scatter(get_x(l, key1), get_x(l, key2), color=color, label=label)
        elif order == 3:
            m3, m2, m1, b = np.polyfit(get_x(l, key1), get_x(l, key2), 3)
            get_x1_sorted = sorted(get_x(l, key1))
            preds = m3 * np.array(get_x1_sorted)**3 + m2 * np.array(get_x1_sorted)**2 + m1 * np.array(get_x1_sorted) + b
            plt.plot(get_x1_sorted, preds, color=color)# label="int."+label)
            plt.scatter(get_x(l, key1), get_x(l, key2), color=color, label=label)
        elif order == "2log":
            m2, m1, b = np.polyfit(np.log(get_x(l, key1)), get_x(l, key2), 2)
            get_x1_sorted = np.log(sorted(get_x(l, key1)))
            preds = m2 * np.array(get_x1_sorted)**2 + m1 * np.array(get_x1_sorted) + b
            plt.plot(sorted(get_x(l, key1)), preds, color=color)# label="int."+label)
            plt.scatter(get_x(l, key1), get_x(l, key2), color=color, label=label)

    colors = cm.rainbow(np.linspace(0, 1, 12))
    for card in range(2, 10):
        #print(card, l[card])
        plot_with_int(l[card], color=colors[card], label="swa" + str(card))

    plt.legend()


def plot_key_all(l, key1, key2, order=1):

    plt.xlabel(dict_key_to_label.get(key1, key1))
    plt.ylabel(dict_key_to_label.get(key2, key2))

    def plot_with_int(l, color, label):
        t = get_x(l, key1)
        if t == []:
            return
        if order == 1:
            m, b = np.polyfit(get_x(l, key1), get_x(l, key2), 1)
            plt.plot(get_x(l, key1), m * np.array(get_x(l, key1)) + b, color=color, label=label +": " + "{:.0f}".format(m*1000))
            plt.scatter(get_x(l, key1), get_x(l, key2), color=color)
        elif order == 2:
            m2, m1, b = np.polyfit(get_x(l, key1), get_x(l, key2), 2)
            get_x1_sorted = sorted(get_x(l, key1))
            preds = m2 * np.array(get_x1_sorted)**2 + m1 * np.array(get_x1_sorted) + b
            plt.plot(get_x1_sorted, preds, color=color)# label="int."+label)
            plt.scatter(get_x(l, key1), get_x(l, key2), color=color, label=label)
        elif order == 3:
            m3, m2, m1, b = np.polyfit(get_x(l, key1), get_x(l, key2), 3)
            get_x1_sorted = sorted(get_x(l, key1))
            preds = m3 * np.array(get_x1_sorted)**3 + m2 * np.array(get_x1_sorted)**2 + m1 * np.array(get_x1_sorted) + b
            plt.plot(get_x1_sorted, preds, color=color)# label="int."+label)
            plt.scatter(get_x(l, key1), get_x(l, key2), color=color, label=label)
        elif order == "2log":
            m2, m1, b = np.polyfit(np.log(get_x(l, key1)), get_x(l, key2), 2)
            get_x1_sorted = np.log(sorted(get_x(l, key1)))
            preds = m2 * np.array(get_x1_sorted)**2 + m1 * np.array(get_x1_sorted) + b
            plt.plot(sorted(get_x(l, key1)), preds, color=color)# label="int."+label)
            plt.scatter(get_x(l, key1), get_x(l, key2), color=color, label=label)

    all_l = [x for ll in l for x in ll]
    plot_with_int(all_l, color="blue", label="all")

    plt.legend()


def plot_markers(l1, l2, key1, key2, order=1, diag=False):

    plt.xlabel(dict_key_to_label.get(key1, key1))
    plt.ylabel(dict_key_to_label.get(key2, key2))

    def plot_without_int(l, color, label, marker):
        t = get_x(l, key1)
        if t == []:
            return
        plt.scatter(get_x(l, key1), get_x(l, key2), color=color, label=label, marker=marker)

    def plot_with_int(l, color):
        if order == 1:
            m, b = np.polyfit(get_x(l, key1), get_x(l, key2), 1)
            plt.plot(get_x(l, key1), m * np.array(get_x(l, key1)) + b, color=color)#, label=label +": " + "{:.0f}".format(m*1000))
        elif order == 2:
            m2, m1, b = np.polyfit(get_x(l, key1), get_x(l, key2), 2)
            get_x1_sorted = sorted(get_x(l, key1))
            preds = m2 * np.array(get_x1_sorted)**2 + m1 * np.array(get_x1_sorted) + b
            plt.plot(get_x1_sorted, preds, color=color)# label="int."+label)
        elif order == 3:
            m3, m2, m1, b = np.polyfit(get_x(l, key1), get_x(l, key2), 3)
            get_x1_sorted = sorted(get_x(l, key1))
            preds = m3 * np.array(get_x1_sorted)**3 + m2 * np.array(get_x1_sorted)**2 + m1 * np.array(get_x1_sorted) + b
            plt.plot(get_x1_sorted, preds, color=color)# label="int."+label)
        elif order == "2log":
            m2, m1, b = np.polyfit(np.log(get_x(l, key1)), get_x(l, key2), 2)
            get_x1_sorted = np.log(sorted(get_x(l, key1)))
            preds = m2 * np.array(get_x1_sorted)**2 + m1 * np.array(get_x1_sorted) + b
            plt.plot(sorted(get_x(l, key1)), preds, color=color)# label="int."+label)

    colors1 = cm.Blues(np.linspace(0, 1, len(l1)))
    colors2 = cm.Reds(np.linspace(0, 1, len(l2)))
    labels1 = ["swa" + str(i) for i in range(len(l1))]
    labels2 = ["soup" + str(i) for i in range(len(l1))]
    for card in range(len(l1)):
        if l1[card] == []:
            continue
        plot_without_int(l1[card], color=colors1[card], label=labels1[card], marker=".")
        plot_with_int(l1[card] + l2[card], color=colors1[card])
    for card in range(len(l2)):
        if l2[card] == []:
            continue
        plot_without_int(l2[card], color=colors2[card], label=labels2[card], marker="*")
    if diag:
        xpoints = ypoints = plt.xlim()
        plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False, label="y=x")

    plt.legend()
