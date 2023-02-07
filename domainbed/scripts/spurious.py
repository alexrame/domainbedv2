import torch
import pezdata
from sklearn.datasets import make_moons
def problem_1(train=True):
    num_samples = 1000
    num_dimensions = 1200
    maj_perc = 0.9 if train else 0.1
    y = torch.zeros(num_samples, 1).bernoulli_(0.5)
    x1 = (torch.randn(num_samples, 1) * 0.1 - 1) * (2 * y - 1)
    x2 = (torch.randn(num_samples, 1) * 0.1 - 1) * (2 * y - 1)
    x2[torch.randperm(len(x2))[:int((1 - maj_perc) * len(x2))]] *= -1
    noise = torch.randn(num_samples, num_dimensions - 2)
    x = torch.cat((x1 * 1, x2 * 5, noise), -1)
    is_maj = x[:, 0] * x[:, 1] > 0
    return x, y.view(-1).long(), is_maj
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
def accuracy(network, x, y):
    return network(x).gt(0).eq(y).float().mean().item()
def train_network(x_tr, y_tr, n_iterations=1000):
    network = torch.nn.Sequential(
        torch.nn.Linear(x_tr.size(1), 200),
        torch.nn.ReLU(),
        torch.nn.Linear(200, 200),
        torch.nn.ReLU(),
        torch.nn.Linear(200, y_tr.size(1)))
    optimizer = torch.optim.SGD(network.parameters(), lr=1e-1)
    loss = torch.nn.BCEWithLogitsLoss()
    for iteration in range(n_iterations):
        optimizer.zero_grad()
        loss(network(x_tr), y_tr).backward()
        optimizer.step()
    return network
if __name__ == "__main__":
    problem = pezdata.problem_moons
    # problem = pezdata.problem_1
    x_tr, y_tr, m_tr = problem(train=True)
    y_tr = y_tr.view(-1, 1).float()
    x_te, y_te, m_te = problem(train=False)
    y_te = y_te.view(-1, 1).float()
    net = train_network(x_tr, y_tr)
    print(accuracy(net, x_tr, y_tr), accuracy(net, x_te, y_te))
