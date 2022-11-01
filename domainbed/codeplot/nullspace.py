import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from codeplot import plot

erm = [
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 0.9980,
    [0.7988, 0.9844, 0.9961, 1., 1., 1., 1., 1., 1., 0.9980], 0.9746,
    [0.7988, 0.9336, 0.9453, 0.9531, 0.9512, 0.9590, 0.9707, 0.9629, 0.9727, 0.9746], 0.8906,
    [0.4141, 0.7090, 0.7266, 0.7305, 0.7578, 0.7930, 0.7734, 0.8145, 0.8301, 0.8906]
]

transfer = [
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    0.9805,
    [0.7891, 0.9766, 0.9941, 0.9961, 0.9980, 0.9980, 1., 1., 0.9980, 0.9805],
    0.9570,
    [0.8008, 0.9336, 0.9434, 0.9434, 0.9492, 0.9609, 0.9473, 0.9609, 0.9590, 0.9570],
    0.9121,
    [0.5371, 0.7363, 0.7734, 0.7930, 0.8516, 0.8691, 0.8672, 0.8965, 0.8984, 0.9121],
]


def plot_slopes_c(key, label):
    fig = plt.figure()
    m = erm[0]

    colors = cm.rainbow(np.linspace(0.2, 1, 2))
    plt.scatter(m, erm[key], label="ERM", color=colors[0])

    #plot.fit_and_plot_with_value(m, erm[key], order="2", label=None, color=colors[0], ax=None)
    plt.scatter(m, transfer[key], label="Improved ERM", color=colors[1])
    if True:
        plt.axhline(y=erm[key][-1], color=colors[0], linestyle="-.")
        plt.axhline(y=transfer[key][-1], color=colors[1], linestyle="-.")
    #plot.fit_and_plot_with_value(m, transfer[key], order="2", label=None, color=colors[1], ax=None)

    plt.xlabel("Fraction", fontsize="x-large")
    plt.ylabel(label, fontsize="x-large")
    plt.legend(fontsize="x-large")
    return fig


plot.plt.rcParams["figure.figsize"] = (8, 5)
fig_trainiid = plot_slopes_c(4, "IID val acc.")
fig_valiid = plot_slopes_c(4, "IID val acc.")
fig_testood = plot_slopes_c(6, "OOD test acc.")

plot.save_fig(fig_trainiid, "pacs0_frac_trainiid.png", folder="/Users/alexandrerame/Documents/deeplearning/writing/fb/from_fb_to_perso")
plot.save_fig(
    fig_trainiid,
    "pacs0_frac_valiid.png",
    folder="/Users/alexandrerame/Documents/deeplearning/writing/fb/from_fb_to_perso"
)
plot.save_fig(
    fig_testood,
    "pacs0_frac_testood.png",
    folder="/Users/alexandrerame/Documents/deeplearning/writing/fb/from_fb_to_perso"
)
