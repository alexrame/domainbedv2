# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import hashlib
import json
import math
import re
import os
import copy
import random
import itertools
import sys
from shutil import copyfile
from collections import OrderedDict, defaultdict
from numbers import Number
import operator

import numpy as np
import torch
import torch.nn as nn
import tqdm
import socket
from collections import Counter

def set_weights(wa_weights, featurizer):
    for name, param in featurizer.named_parameters():
        param.data = wa_weights[name]

def get_wa_weights(lambda_interpolation, featurizers):
    weights = {}
    list_gen_named_params = [featurizer.named_parameters() for featurizer in featurizers]
    for name_0, param_0 in featurizers[0].named_parameters():
        named_params = [next(gen_named_params) for gen_named_params in list_gen_named_params]
        new_data = torch.zeros_like(param_0.data)
        sum_lambdas = 0.
        for i in range(len(featurizers)):
            lambda_i = lambda_interpolation[i]
            name_i, param_i = named_params[i]
            assert name_0 == name_i
            lambda_i = lambda_interpolation[i]
            new_data = new_data + lambda_i * param_i
            sum_lambdas += lambda_i
        assert sum_lambdas > 0
        weights[name_0] = new_data / sum_lambdas
    return weights

def load_featurizer(featurizer, save_dict):
    try:
        featurizer.load_state_dict(save_dict, strict=True)
    except Exception as exc:
        print(f"Had an issue when loading weights. Try with some renaming.")
        new_save_dict = {
            re.sub("^0.network", "network", key): value
            # key.replace("0.network", "network"): value
            for key, value in save_dict.items()
            if key not in ["1.weight", "1.bias"]
        }

        featurizer.load_state_dict(new_save_dict, strict=True)


def process_save_path(_subpath_for_init, hparams):
    subpath_for_init = _subpath_for_init.split("!")[0].split("?")[0].split("*")[0]
    # ugly fix
    if "!u" in _subpath_for_init:
        # u for uniq
        hparams_seed = hparams.get("hparams_seed", None)
    else:
        hparams_seed = None
    if "?b" in _subpath_for_init:
        # b for best
        model_path = "model_bestma.pkl"
    else:
        model_path = "model_best.pkl"
    subpath_for_init = get_save_path(
        subpath_for_init,
        hparams_seed=hparams_seed,
        model_path=model_path)
    return subpath_for_init


def clean_state_dict(state_dict, _subpath_for_init):
    if "model_dict" in state_dict:
        state_dict = state_dict["model_dict"]
    keynetwork = "network"
    if "*" in _subpath_for_init:
        keynetwork += "_" + _subpath_for_init.split("*")[1]

    keys = [k for k in state_dict.keys() if k.startswith(keynetwork + ".0")]
    state_dict = {
        re.sub("^" + keynetwork + ".0.network", "network", k): state_dict[k]
        for k in keys
    }
    return state_dict

def get_save_path(save_path, hparams_seed=None, model_path="model_best.pkl"):
    dict_pkls = {
        "terra": "/private/home/alexandrerame/dataplace/data/domainbed/inits/terra/transfer/terra_erm_lp_t0_0926.pkl",
        "vlcs": "/private/home/alexandrerame/dataplace/data/domainbed/inits/vlcs/transfer/vlcs_erm_lp_t0_0926.pkl",
        "home": "/private/home/alexandrerame/dataplace/data/domainbed/inits/home/transfer/home_erm_lp_r0_t0_0926.pkl",
        "pacs":
            "/private/home/alexandrerame/dataplace/data/domainbed/inits/pacs/transfer/pacs_erm0123_lp_0916_r0.pkl",
        "dn0":
            "/private/home/alexandrerame/dataplace/data/domainbed/inits/dn/transfer/dn_erm0_lp0_r0_t0_0926.pkl",
        "dn1":
            "/private/home/alexandrerame/dataplace/data/domainbed/inits/dn/transfer/dn_erm1_lp1_r0_t0_0926.pkl",
        "dn2":
            "/private/home/alexandrerame/dataplace/data/domainbed/inits/dn/transfer/dn_erm2_lp2_r0_t0_0926.pkl",
        "dn3":
            "/private/home/alexandrerame/dataplace/data/domainbed/inits/dn/transfer/dn_erm3_lp3_r0_t0_0926.pkl",
        "dn4":
            "/private/home/alexandrerame/dataplace/data/domainbed/inits/dn/transfer/dn_erm4_lp4_r0_t0_0926.pkl",
        "dn5":
            "/private/home/alexandrerame/dataplace/data/domainbed/inits/dn/transfer/dn_erm5_lp5_r0_t0_0926.pkl",
        "dn":
            "/private/home/alexandrerame/dataplace/data/domainbed/inits/dn/transfer/dn_erm_lp_r0_t0_0926.pkl",
        "dnma": "/private/home/alexandrerame/dataplace/data/domainbed/inits/dn/continual/dn_ma_lp_0926.pkl",
        "dnerm":
            "/private/home/alexandrerame/dataplace/experiments/domainbed/dn/dn_erm_lp_0926/439fe416014ec6fbf6e8bf8e01119e90/model_feats.pkl",
        "vlcserm": "/private/home/alexandrerame/dataplace/experiments/domainbed/vlcs/vlcs_ma0123_lp_0926/a2bf23072bc96618e252d022f80dee7b/model.pkl",
        "rxrxerm": "/private/home/alexandrerame/dataplace/experiments/domainbed/rxrx/rxrx_erm_lp_0926/97424abf45c621f833fc33f6a6c39925/model_feats.pkl",
        "cameerm": "/private/home/alexandrerame/dataplace/experiments/domainbed/came/came0_erm_lp_0926/7a8760d390c2056f28cd0372d7685220/model_feats.pkl",
        "dnf":
            "/private/home/alexandrerame/dataplace/data/domainbed/inits/dn/transfer/dn_ermf_lp_r0_t0_0926.pkl",
        "iwildnof":
            "/private/home/alexandrerame/dataplace/data/domainbed/inits/iwild/transfer/iwild_erm_lp_r0_t0_0926.pkl",
        "iwild":
            "/private/home/alexandrerame/dataplace/data/domainbed/inits/iwild/transfer/iwild_ermf_lp_r0_t0_0926.pkl",
        "wildma":
            "/private/home/alexandrerame/dataplace/data/domainbed/inits/iwild/transfer/iwild_maf_lp_r0_t0_0926.pkl",
        "natu":
            "/private/home/alexandrerame/dataplace/data/domainbed/inits/natu/transfer/natu_erm_lp_r0_t0_0926.pkl",
        "rxrx":
            "/private/home/alexandrerame/dataplace/data/domainbed/inits/rxrx/transfer/rxrx_erm_lp_r0_t0_0926.pkl"
    }
    if save_path in dict_pkls:
        return dict_pkls[save_path]

    dict_folders = {
        "fdn": "/private/home/alexandrerame/dataplace/experiments/domainbed/dn/dn_erm012345_lp_0926",
        "fdnma": "/private/home/alexandrerame/dataplace/experiments/domainbed/dn/dn_ma012345_lp_0926",
        "fhome": "/private/home/alexandrerame/dataplace/experiments/domainbed/home/home_erm0123_lp_0926",
        "fhomema": "/private/home/alexandrerame/dataplace/experiments/domainbed/home/home_ma0123_lp_0926",
        "fpacs": "/private/home/alexandrerame/dataplace/experiments/domainbed/pacs/pacs_erm0123_lp_0926",
        "fpacsma": "/private/home/alexandrerame/dataplace/experiments/domainbed/pacs/pacs_ma0123_lp_0926",
        "fvlcs": "/private/home/alexandrerame/dataplace/experiments/domainbed/vlcs/vlcs_ma0123_lp_0926",
        "fvlcsma": "/private/home/alexandrerame/dataplace/experiments/domainbed/vlcs/vlcs_ma0123_lp_0926",
        "fterra": "/private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra_ma0123_lp_0926",
        "fterrama": "/private/home/alexandrerame/dataplace/experiments/domainbed/terra/terra_ma0123_lp_0926",

        "fiwild": "/private/home/alexandrerame/dataplace/experiments/domainbed/iwild/iwild_erm_lp_0926",
        "fiwildma": "/private/home/alexandrerame/dataplace/experiments/domainbed/iwild/iwild_ma_lp_0926",
        "fnatu": "/private/home/alexandrerame/dataplace/experiments/domainbed/natu/natu_erm_lp_0926",
        "fnatuma": "notdoneyet"
    }
    if save_path in dict_folders:
        folder = dict_folders[save_path]
        subfolders = [os.path.join(folder, path) for path in os.listdir(folder)]
        subfolders = sorted([subfolder for subfolder in subfolders if os.path.isdir(subfolder) and model_path in os.listdir(subfolder) and "done" in os.listdir(subfolder)])
        if len(subfolders) != 20:
            print("Surprising count of subfolders")
        if hparams_seed is None:
            selected_init = os.path.join(random.choice(subfolders), model_path)
        else:
            print(f"Take {hparams_seed}")
            hparams_seed = hparams_seed % len(subfolders)
            selected_init = os.path.join(subfolders[hparams_seed], model_path)
        print("select", selected_init, " from subfolders", subfolders, len(subfolders))
        return selected_init

    return save_path

def get_batchdiversity_loss(logits):
    msoftmax = nn.Softmax(dim=1)(logits).mean(dim=0)
    return torch.sum(msoftmax * torch.log(msoftmax + 1e-5))

def get_entropy_loss(logits):
    """
    Entropy loss for probabilistic prediction vectors
    """
    return torch.mean(
        torch.sum(
            -nn.functional.softmax(logits, dim=1) * nn.functional.log_softmax(logits, dim=1), 1
        )
    )
# def compute_entropy_predictions(x):
#     #print(x)
#     entropy = x * torch.log(x + 1e-10)  #bs * num_classes
#     return -1. * entropy.sum() / entropy.size(0)


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
    if test_envs == [-1] and model_selection == "oracle":
        return 0.
    val_env_keys = []
    if metric_key == "out_acc" and f'env0_' + metric_key not in results:
        metric_key="out_Accuracies/acc_net"

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
    if i == 0:
        assert "ma" in metric_key, results
        return 0
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

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

def get_dict_entropy(dict_stats, device):
    dict_results = {}

    for key, value in dict_stats.items():
        if "logits" not in value:
            print(value.keys())
            continue
        logits = value["logits"]
        entropy = get_entropy_loss(logits)
        batchdiv = get_batchdiversity_loss(logits)

        dict_results["ent_" + key] = np.mean(
            entropy.float().cpu().numpy()
        )  # mean just to get rid of array
        dict_results["bdi_" + key] = np.mean(
            batchdiv.float().cpu().numpy()
        )  # mean just to get rid of array
    return dict_results


def results_ensembling(algorithm, loader, device, what=[], do_div=True, do_ent=False):
    algorithm.eval()
    dict_stats, aux_dict_stats = algorithm.get_dict_prediction_stats(
        loader, device, what=what + ["classes"]
    )
    dict_results = {}
    for key in dict_stats:
        dict_results[("acc_" + key if key != "" else "acc")] = dict_stats[key]["acc"]

    if len(algorithm.networks):
        num_networks = int(min(len(algorithm.networks), float(os.environ.get("MAXM", math.inf))))
        dict_results["acc_netm"] = np.mean(
            [
                dict_results[f"acc_net{key}"]
                for key in range(num_networks)
            ]
        )
        dict_results["acc_netmax"] = max(
            [
                dict_results[f"acc_net{key}"]
                for key in range(num_networks)
            ]
        )
        if os.environ.get("SUBWA", "0") != "0":
            for subwa in [("0", "1"), ("1", "2"), ("2", "3")]:
                dict_results[f"acc_net{''.join(subwa)}"] = np.mean(
                    [
                        dict_results[f"acc_net{key}"]
                        for key in range(num_networks)
                        if str(key) in subwa
                    ]
                )
        if os.environ.get("DELETE_NETM", "1") != "0":
            for key in range(num_networks):
                del dict_results[f"acc_net{key}"]

    if do_div:
        print("Compute prediction diversity")
        targets = torch.cat(aux_dict_stats["batch_classes"]).cpu().numpy()
        dict_diversity = algorithm.get_dict_diversity(dict_stats, targets, device, divregex=do_div)
        dict_results.update(dict_diversity)
    if do_ent:
        print("Compute prediction entropy")
        dict_entropy = get_dict_entropy(dict_stats, device)
        dict_results.update(dict_entropy)
    return dict_results, aux_dict_stats


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
