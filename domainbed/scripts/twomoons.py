import torch
# import pezdata
from sklearn.datasets import make_moons

# def problem_1(train=True):
#     num_samples = 1000
#     num_dimensions = 1200
#     maj_perc = 0.9 if train else 0.1
#     y = torch.zeros(num_samples, 1).bernoulli_(0.5)
#     x1 = (torch.randn(num_samples, 1) * 0.1 - 1) * (2 * y - 1)
#     x2 = (torch.randn(num_samples, 1) * 0.1 - 1) * (2 * y - 1)
#     x2[torch.randperm(len(x2))[:int((1 - maj_perc) * len(x2))]] *= -1
#     noise = torch.randn(num_samples, num_dimensions - 2)
#     x = torch.cat((x1 * 1, x2 * 5, noise), -1)
#     is_maj = x[:, 0] * x[:, 1] > 0
#     return x, y.view(-1).long(), is_maj


def problem_moons(train=True):
    num_samples = 1000 if train else 1000 * 100
    num_dimensions = 50
    maj_perc = 0.5 if train else 0.1
    x_, y_ = make_moons(n_samples=num_samples, noise=0.07)
    x_[:, [0, 1]] = x_[:, [1, 0]]
    x_[:, 0] *= -1
    x_ *= 1.1
    x_ = x_ - x_.mean(0, keepdims=True)
    x_[:, 0] *= 1.2
    x = torch.FloatTensor(x_)
    y = torch.FloatTensor(y_).view(-1, 1)
    is_maj = torch.zeros(num_samples)
    is_maj = torch.logical_or(
        ((2 * y - 1) * torch.sign(x[:, 0:1])) > 0,
        (torch.abs(x[:, 1:2]) > 1))
    if not train:
        i_min = torch.where(is_maj == 0)[0][:900]
        i_maj = torch.where(is_maj == 1)[0][:100]
        i_all = torch.cat((i_min, i_maj))[torch.randperm(1000)]
        x, y, is_maj = x[i_all], y[i_all], is_maj[i_all]
    noise = torch.randn(len(x), num_dimensions - 2)
    x = torch.cat((x, noise), 1)
    return x, y.view(-1).long(), is_maj.view(-1)

def build_network(x_tr, y_tr):
    return torch.nn.Sequential(
        torch.nn.Linear(x_tr.size(1), 200),
        torch.nn.ReLU(),
        torch.nn.Linear(200, 200),
        torch.nn.ReLU(),
        torch.nn.Linear(200, y_tr.size(1)))



def train_network(network, x_tr, y_tr, n_iterations=1000):
    optimizer = torch.optim.SGD(network.parameters(), lr=1e-1)
    loss = torch.nn.BCEWithLogitsLoss()
    for iteration in range(n_iterations):
        optimizer.zero_grad()
        loss(network(x_tr), y_tr).backward()
        optimizer.step()
    return network

def accuracy(network, x, y):
    return network(x).gt(0).eq(y).float().mean().item()


from pezdata import problem_1, problem_moons
import torch
from matplotlib import pyplot as plt
import numpy as np
def plot_v2(x_tr, y_tr, model=None, draw_boundary=True):
    x_min = x_tr[:, :2].min(0).values.tolist()
    x_max = x_tr[:, :2].max(0).values.tolist()
    x1_min = x_min[0] - 0.5 * abs(x_min[0])
    x1_max = x_max[0] + 0.5 * abs(x_max[0])
    x2_min = x_min[1] - 0.5 * abs(x_min[1])
    x2_max = x_max[1] + 0.5 * abs(x_max[1])

    def generate_heatmap_plane(x, m=150):
        x = x.data.numpy()
        d1, d2 = np.meshgrid(
            np.linspace(x1_min, x1_max, m),
            np.linspace(x2_min, x2_max, m))
        heatmap_plane = np.stack((d1.flatten(), d2.flatten()), axis=1)
        # Below, we compute the distance of each point to the training
        # datapoints. If the distance is less than 1e-3, that point
        # uses the noise features of the closest training point.
        # Otherwise a random but constant noise is added.
        dists = ((heatmap_plane[:, 0:1] - x[:, 0:1].T) ** 2 +
                 (heatmap_plane[:, 1:2] - x[:, 1:2].T) ** 2)
        noise_dims = x[np.argmin(dists, 1)][:, 2:]
        noise_dims[dists.min(1) > 0.001] = np.random.randn(
            1, noise_dims.shape[1]) * 1.0
        return np.concatenate([heatmap_plane, noise_dims], 1)
    cm = plt.cm.RdBu.copy()
    cm.set_under("#C82506")
    cm.set_over("#0365C0")
    if model is not None:
        heatmap_plane = [generate_heatmap_plane(x_tr) for i in range(5)]
        heatmap_x = heatmap_plane[0][:, 0].reshape(150, 150)
        heatmap_y = heatmap_plane[0][:, 1].reshape(150, 150)
        y_hat = [model(torch.FloatTensor(hp)).data.cpu().numpy()[None]
                 for hp in heatmap_plane]
        y_hat = np.concatenate(y_hat).mean(0)
        if y_hat.shape[1] == 2:
            y_hat = y_hat[:, 1:2] - y_hat[:, 0:1]
        heatmap_preds = 1.0 / (1.0 + np.exp(-y_hat))
        heatmap_preds = heatmap_preds.reshape(150, 150)
        plt.contourf(
            heatmap_x,
            heatmap_y,
            heatmap_preds,
            np.linspace(0, 1, 20),
            cmap=cm,
            alpha=0.5,
            vmin=0,
            vmax=1,
            extend="both")

        if draw_boundary:
            plt.contour(
                heatmap_x,
                heatmap_y,
                heatmap_preds,
                [0.5],
                antialiased=True,
                linewidths=1.0,
                colors="k")

    plt.scatter(
        x_tr[:, 0],
        x_tr[:, 1],
        c=(y_tr > 0).int() - (y_tr <= 0).int(),
        cmap=cm,
        edgecolors='none',
        s=20,
        alpha=0.25)

    plt.axhline(
        y=0,
        ls="--",
        lw=0.7,
        color="k",
        alpha=0.5)

    plt.axvline(
        x=0,
        ls="--",
        lw=0.7,
        color="k",
        alpha=0.5)

    plt.grid(
        color="k",
        linestyle="--",
        linewidth=0.5,
        alpha=0.25)

    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    return plt
x_tr, y_tr, m_tr = problem_moons()
model = torch.nn.Sequential(
    torch.nn.Linear(x_tr.size(1), 200),
    torch.nn.ReLU(),
    torch.nn.Linear(200, 200),
    torch.nn.ReLU(),
    torch.nn.Linear(200, 1))
optimizer = torch.optim.SGD(model.parameters(), 0.1)
loss = torch.nn.BCEWithLogitsLoss()
for _ in range(1000):
    optimizer.zero_grad()
    loss(model(x_tr), y_tr.view(-1, 1).float()).backward()
    optimizer.step()

plot_v2(x_tr, y_tr, model)

if __name__ == "__main__":
    problem = problem_moons
    # problem = problem_1
    x_tr, y_tr, m_tr = problem(train=True)
    y_tr = y_tr.view(-1, 1).float()
    x_te, y_te, m_te = problem(train=False)
    y_te = y_te.view(-1, 1).float()
    network = build_network(x_tr, y_tr)
    train_network(network, x_tr, y_tr)
    print(accuracy(network, x_tr, y_tr), accuracy(network, x_te, y_te))
