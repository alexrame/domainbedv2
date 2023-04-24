import textwrap
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (6, 6)
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times Roman'

SIZE = "large"
class ComplexRadar():
    """
    Create a complex radar chart with different scales for each variable
    Parameters
    ----------
    fig : figure object
        A matplotlib figure object to add the axes on
    variables : list
        A list of variables
    ranges : list
        A list of tuples (min, max) for each variable
    n_ring_levels: int, defaults to 5
        Number of ordinate or ring levels to draw
    show_scales: bool, defaults to True
        Indicates if we the ranges for each variable are plotted
    """
    def __init__(self, fig, variables, ranges, n_ring_levels=5, show_scales=True):
        # Calculate angles and create for each variable an axes
        # Consider here the trick with having the first axes element twice (len+1)
        angles = np.arange(0, 360, 360./len(variables))
        axes = [fig.add_axes([0.1,0.1,0.9,0.9], polar=True, label = "axes{}".format(i)) for i in range(len(variables)+1)]

        # Ensure clockwise rotation (first variable at the top N)
        for ax in axes:
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_axisbelow(True)

        # Writing the ranges on each axes
        for i, ax in enumerate(axes):

            # Here we do the trick by repeating the first iteration
            j = 0 if (i==0 or i==1) else i-1
            ax.set_ylim(*ranges[j])
            # Set endpoint to True if you like to have values right before the last circle
            grid = np.linspace(*ranges[j], num=n_ring_levels,
                               endpoint=True)
            gridlabel = [
                "{}".format(round(x, 2 if (variables[j] == '$BLEU4$' and x == 0.4238) else 3))
                for x in grid
            ]
            gridlabel[0] = "" # remove values from the center
            lines, labels = ax.set_rgrids(grid, labels=gridlabel, angle=angles[j])

            ax.set_ylim(*ranges[j])
            ax.spines["polar"].set_visible(False)
            ax.grid(visible=False)

            if show_scales == False:
                ax.set_yticklabels([])

        # Set all axes except the first one unvisible
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.xaxis.set_visible(False)

        # Setting the attributes
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
        self.ax1 = axes[1]
        self.plot_counter = 0

        # Draw (inner) circles and lines
        self.ax.yaxis.grid()
        self.ax.xaxis.grid()

        # Draw outer circle
        self.ax.spines['polar'].set_visible(True)

        # ax1 is the duplicate of axes[0] (self.ax)
        # Remove everything from ax1 except the plot itself
        self.ax1.axis('off')
        self.ax1.set_zorder(9)

        # Create the outer labels for each variable
        # print(angles)
        angles_fixed = [angles[0], angles[1]]
        angles_fixed.extend(angles[2:])
        print(angles_fixed)
        l, text = self.ax.set_thetagrids(
            angles_fixed, labels=variables,
            color="black")

        # Beautify them
        labels = [t.get_text() for t in self.ax.get_xticklabels()]
        labels = ['\n'.join(textwrap.wrap(l, 1,
                                          break_long_words=False)) for l in labels]
        self.ax.set_xticklabels(labels, size=SIZE)

        for t,a in zip(self.ax.get_xticklabels(),angles):
            if a == 0:
                t.set_ha('center')
            elif a > 0 and a < 180:
                t.set_ha('left')
            elif a == 180:
                t.set_ha('center')
            else:
                t.set_ha('right')

        self.ax.tick_params(axis='both', pad=10)


    def _scale_data(self, data, ranges):
        """Scales data[1:] to ranges[0]"""
        for d, (y1, y2) in zip(data[1:], ranges[1:]):
            assert (y1 <= d <= y2) or (y2 <= d <= y1)
        x1, x2 = ranges[0]
        d = data[0]
        sdata = [d]
        for d, (y1, y2) in zip(data[1:], ranges[1:]):
            sdata.append((d-y1) / (y2-y1) * (x2 - x1) + x1)
        return sdata

    def plot(self, data, *args, **kwargs):
        """Plots a line"""
        sdata = self._scale_data(data, self.ranges)
        self.ax1.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kwargs)
        self.plot_counter = self.plot_counter+1

    def fill(self, data, *args, **kwargs):
        """Plots an area"""
        sdata = self._scale_data(data, self.ranges)
        self.ax1.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kwargs)

    def use_legend(self, *args, **kwargs):
        """Shows a legend"""
        self.ax1.legend(*args, **kwargs)

    def set_title(self, title, pad=25, **kwargs):
        """Set a title"""
        self.ax.set_title(title,pad=pad, **kwargs)



from matplotlib.transforms import Affine2D
from matplotlib.projections.polar import PolarTransform


def normalize_data(data, dict_index_to_minmax):
    normalized_data = []
    for curve_data in data:
        normalized_curve_data = []
        for i, value in enumerate(curve_data):
            min_val, max_val = dict_index_to_minmax[i]
            normalized_value = (value - min_val) / (max_val - min_val)
            normalized_curve_data.append(normalized_value)
        normalized_data.append(normalized_curve_data)
    return normalized_data


def create_spider_chart(
    data,
    labels,
    line_labels,
    colors,
    markers,
    title,
    dict_key_to_minmax,
    num_ticks=5,
    dict_clean_labels={},
    legendtitle=None,
    legend_loc="upper right"
):
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    dict_index_to_minmax = {i: dict_key_to_minmax[labels[i]] for i in range(len(labels))}
    normalized_data = normalize_data(data, dict_index_to_minmax)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for i, curve_data in enumerate(normalized_data):
        is_main = (i == len(normalized_data) - 1)
        new_curve_data = np.concatenate((curve_data, [curve_data[0]]))
        new_angles = np.concatenate((angles, [angles[0]]))
        ax.plot(
            new_angles, new_curve_data, color=colors[i], marker=markers[i], label=line_labels[i],
            linewidth=3 if is_main else 1)
        ax.fill(new_angles, new_curve_data, color=colors[i], alpha=0.5 if is_main else 0.25)

        if is_main and False:
            # Display real values as text labels
            for j in range(len(data[i])):
                angle_rad = new_angles[j]
                real_value = data[i][j]
                trans_offset = PolarTransform().transform((angle_rad, new_curve_data[j]))
                trans = Affine2D().translate(*trans_offset) + ax.transData
                ax.text(angle_rad, curve_data[j], f"{real_value:.2f} for " + labels[j], transform=trans, ha='center', fontsize=10, color="green")


    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles)
    ax.set_xticklabels([dict_clean_labels.get(label, label) for label in labels])

    ax.set_ylim(0, 1)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)

    y_ticks = np.linspace(0, 1, num_ticks)
    ax.set_yticks(y_ticks)

    if isinstance(legend_loc, tuple):
        ax.legend(title=legendtitle, bbox_to_anchor=legend_loc)
    else:
        ax.legend(title=legendtitle, loc=legend_loc)
    plt.title(title, size=20, color="navy", y=1.1)

    return fig
