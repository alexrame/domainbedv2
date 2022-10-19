# import torch.nn as nn
# net_wa = torch.nn.Linear(2, 3)
# net_0 = torch.nn.Linear(2, 3)
# net_1 = torch.nn.Linear(2, 3)
# lambda_0 = torch.tensor(0.5, requires_grad=True)
# lambda_1 = torch.tensor(0.5, requires_grad=True)

# for param_wa, param_0, param_1 in zip(wa.parameters(), net_0.parameters(), net_1.parameters()):
#     param_wa = (param_0 * lambda_0 + param_1 * lambda_1) / (lambda_0 + lambda_1)

# input_tensor = torch.rand(3, 2)
# out = torch.mean(net_wa(input_tensor))
# print("out:", out, "\n")
# out.backward()

# print("net_wa: ", net_wa.weight, "\nnet_wa.grad: ", net_wa.weight.grad, "\n")
# print("net_0: ", net_0.weight, "\nnet_0.grad:", net_0.weight.grad, "\n")
# print("lambda_0: ", lambda_0, "\nlambda_0.grad:", lambda_0.grad)

import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np

input_dim = 784
classes = 10

layer_shapes = [
    [(64, input_dim), (64,)],  # (w0, b0)
    [(classes, 64), (classes,)]  # (w1, b1)
]

# num_weights_to_generate = (classes * 64 + classes) + (64 * input_dim + 64)

# hypernetwork = nn.Sequential(
#     nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, num_weights_to_generate), nn.Sigmoid()
# )

lambda_0 = torch.tensor(0.5, requires_grad=True)
lambda_1 = torch.tensor(0.5, requires_grad=True)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam([lambda_0, lambda_1], lr=1e-4)


class MainNet(torch.nn.Module):

    def __init__(self):
        super(MainNet, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, 64)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(64, classes)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


net_wa = MainNet()
net_0 = MainNet()
net_1 = MainNet()
for param in net_wa.parameters():
    param.requires_grad = False
for param in net_0.parameters():
    param.requires_grad = False
for param in net_1.parameters():
    param.requires_grad = False

def train_step(x, y):
    optimizer.zero_grad()

    weights = {}
    for (name_0, param_0), (name_1, param_1) in zip(net_0.named_parameters(), net_1.named_parameters()):
        assert name_0 == name_1
        weights[name_0] = (param_0 * lambda_0 + param_1 * lambda_1) / (lambda_0 + lambda_1)

    preds = torch.nn.utils.stateless.functional_call(net_wa, weights, x)
    loss = loss_fn(preds, y)
    loss.backward(retain_graph=True)
    optimizer.step()
    return loss.item()

bs = 3
x = torch.rand(bs, input_dim)
y = torch.rand(bs, classes)

for i in range(100):
    l = train_step(x, y)
    print(i, l, lambda_0, lambda_1)
