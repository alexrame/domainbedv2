# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import hashlib
import json
import math
import os
import copy
import itertools
import sys
from shutil import copyfile
from collections import OrderedDict, defaultdict
from numbers import Number
import operator

import numpy as np
import torch
import tqdm
import socket
from collections import Counter


def is_float(element):
    try:
        float(element)
        return True
    except:
        return False

def is_none(arg):
    return arg in [None, "", "none", "None", "0", "False", "false", False, 0]

def is_not_none(arg):
    return not is_none(arg)


def get_machine_name():
    return socket.gethostname()

## DiWA ##
def get_score(results, test_envs, metric_key="out_acc", model_selection="train"):
    val_env_keys = []
    for i in itertools.count():
        acc_key = f'env{i}_' + metric_key
        if acc_key in results:
            if model_selection == "train":
                if i not in test_envs:
                    val_env_keys.append(acc_key)
            else:
                assert model_selection == "oracle"
                if i in test_envs:
                    val_env_keys.append(acc_key)
        else:
            break
    assert i > 0, results
    return np.mean([results[key] for key in val_env_keys])

## DiWA ##
class MergeDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        super(MergeDataset, self).__init__()
        self.datasets = datasets

    def __getitem__(self, key):
        count = 0
        for d in self.datasets:
            if key - count >= len(d):
                count += len(d)
            else:
                return d[key - count]
        raise ValueError(key)

    def __len__(self):
        return sum([len(d) for d in self.datasets])


def l2_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
        torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
        torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    ).pow(2).mean()

class MovingAverage:

    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.ema_data = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.ema_data[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                # correction by 1/(1 - self.ema)
                # so that the gradients amplitude backpropagated in data is independent of self.ema
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.ema_data[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data



def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights

def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_separator():
    print("="*80)

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)

def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def split_meta_train_test(minibatches, num_meta_test=1):
    n_domains = len(minibatches)
    perm = torch.randperm(n_domains).tolist()
    pairs = []
    meta_train = perm[:(n_domains-num_meta_test)]
    meta_test = perm[-num_meta_test:]

    for i,j in zip(meta_train, cycle(meta_test)):
        xi, yi = minibatches[i][0], minibatches[i][1]
        xj, yj = minibatches[j][0], minibatches[j][1]

        min_n = min(len(xi), len(xj))
        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


def _print_dict(_dict):
    """
    function that print args dictionaries in a beautiful way
    """
    print("\n" + "#" * 40)
    col_width = max(len(str(word)) for word in _dict) + 2
    for arg in sorted(list(_dict.keys())):
        str_print = str(_dict[arg])
        _str = "".join([str(arg).ljust(col_width), str_print])
        print(_str)
    print("#" * 40 + "\n")


def print_args(args):
    _dict = args if isinstance(args, dict) else args.__dict__
    _print_dict(_dict)


def results_ensembling(algorithm, loader, device):
    algorithm.eval()
    dict_stats, batch_classes = algorithm.get_dict_prediction_stats(loader, device)
    dict_results = {}
    for key in dict_stats:
        dict_results[("acc_" + key if key != "" else "acc")] = sum(
            dict_stats[key]["correct"].numpy()) / len(dict_stats[key]["correct"].numpy())
    if len(algorithm.networks):
        dict_results["acc_netm"] = np.mean(
            [
                dict_results[f"acc_net{key}"]
                for key in range(
                    int(min(len(algorithm.networks), float(os.environ.get("MAXM", math.inf)))))
            ]
        )
        if os.environ.get("DELETE_NETM"):
            for key in range(int(min(len(algorithm.networks), float(os.environ.get("MAXM", math.inf))))):
                del dict_results[f"acc_net{key}"]

    targets = torch.cat(batch_classes).cpu().numpy()
    # print("Compute diversity across different networks")
    dict_diversity = algorithm.get_dict_diversity(dict_stats, targets, device)
    dict_results.update(dict_diversity)
    return dict_results


def accuracy(algorithm, loader, weights, device):
    dict_key_to_correct = {}
    weights_offset = 0
    total = 0

    algorithm.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            prediction = algorithm.predict(x)
            if not isinstance(prediction, dict):
                prediction = {"": prediction}
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset:weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            total += batch_weights.sum().item()

            for key_p, p in prediction.items():
                if key_p not in dict_key_to_correct:
                    dict_key_to_correct[key_p] = 0
                if p.size(1) == 1:
                    dict_key_to_correct[key_p] += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
                else:
                    dict_key_to_correct[key_p] += (p.argmax(1).eq(y).float() * batch_weights).sum().item()

    algorithm.train()

    return {("acc_" + key_p if key_p != "" else "acc"): (correct / total)
            for key_p, correct in dict_key_to_correct.items()}


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)
