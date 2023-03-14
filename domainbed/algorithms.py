# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import os
import pdb
import random
import copy
import re
from torch.distributions.normal import Normal
import numpy as np
from collections import defaultdict, OrderedDict
try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

from domainbed import networks
from domainbed.lib import misc

ALGORITHMS = [
    'ERM', "ERMG", "TWA", "TWAMA", "ERMLasso", "MA", 'Fish', 'IRM', 'GroupDRO', 'Mixup', 'CORAL', 'MMD', 'VREx',
    "Fishr", "DARE", "DARESWAP"
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(
        self,
        input_shape,
        num_classes,
        num_domains,
        hparams,
        what_is_trainable=False,
        path_for_init=None
    ):
        super(ERM, self).__init__(input_shape, num_classes, num_domains, hparams)

        self._what_is_trainable = {
            "0": "all",
            "1": "cla",
            "clafrozen": "feat"}.get(what_is_trainable, what_is_trainable)
        self._create_network()
        self._load_network(path_for_init)
        self.update_count = 0
        self._init_optimizer()

    def _create_network(self):
        self.featurizer = networks.Featurizer(self.input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs, self.num_classes, self.hparams['nonlinear_classifier']
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)

    def get_network_state_dict(self):
        return self.network.state_dict()

    def _load_network(self, path_for_init):
        ## DiWA load shared initialization ##
        if misc.is_not_none(path_for_init):
            for i, _subpath_for_init in enumerate(path_for_init.split(",")):
                subpath_for_init = misc.process_save_path(_subpath_for_init, self.hparams)

                if not os.path.exists(subpath_for_init):
                    raise ValueError(f"Your initialization {subpath_for_init} has not been saved yet")

                state_dict = torch.load(subpath_for_init)
                if os.environ.get("LOAD_ONLY_FEATURES", "0") != "0" or i > 0:
                    if "model_dict" in state_dict:
                        state_dict = misc.clean_state_dict(state_dict, _subpath_for_init)
                    print(f"Load featurizer from: {subpath_for_init} at i: {i}")
                    misc.load_featurizer(self.featurizer, state_dict)
                elif "network_dict" in state_dict:
                    print(f"Load network from: {subpath_for_init}")
                    self.network.load_state_dict(state_dict["network_dict"])
                elif "model_dict" in state_dict:
                    print(f"Load model from: {subpath_for_init}")
                    self.load_state_dict(state_dict["model_dict"])
                else:
                    print(f"Load network from: {subpath_for_init}")
                    self.network.load_state_dict(state_dict)

            if os.environ.get("RESET_CLASSIFIER"):
                print("Reset random classifier")
                self.classifier.reset_parameters()

    def _get_training_parameters(self):
        if self._what_is_trainable in ["warmupnet", "warmupff"]:
            if self.update_count == self.hparams["warmup"]:
                if self._what_is_trainable == "warmupff":
                    what_is_trainable = "firstfrozen"
                else:
                    what_is_trainable = "all"
            else:
                assert self.update_count == 0
                what_is_trainable = "cla"
        else:
            what_is_trainable = self._what_is_trainable

        ## DiWA choose weights to be optimized ##
        if what_is_trainable in ["all", ]:
            print("Learn featurizer and classifier")
            training_parameters = self.network.parameters()
        elif what_is_trainable in ["firstfrozen", ]:
            # import pdb; pdb.set_trace()
            print("Learn featurizer except first and classifier")
            training_parameters = [p for n,p in list(self.network.named_parameters()) if n not in ["0.network.conv1.weight", "0.network.bn1.weight", "0.network.bn1.bias"]]
        elif what_is_trainable in ["feat"]:
            # useful for linear probing
            print("Learn only featurizer")
            training_parameters = self.featurizer.parameters()
        elif what_is_trainable.startswith("frozen_"):
            layer = int(what_is_trainable.split("_")[1])
            training_parameters = [
                p for n, p in list(self.featurizer.named_parameters())
                if n.startswith("network.layer") and int(n.split(".")[1][-1]) >= layer
            ]
            training_parameters += list(self.classifier.parameters())
        else:
            assert what_is_trainable in ["cla"]
            # useful when learning with fixed vocabulary
            print("Learn only classifier")
            training_parameters = self.classifier.parameters()
        return training_parameters

    def _init_optimizer(self):
        training_parameters = self._get_training_parameters()
        self.optimizer = torch.optim.Adam(
            training_parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_features = self.featurizer(all_x)
        if self._what_is_trainable in ["cla"]:
            all_features = all_features.detach()
        all_features = self.modify_features(all_features)
        loss = F.cross_entropy(self.classifier(all_features), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.update_count == self.hparams["warmup"] and self._what_is_trainable in [
            "warmupnet", "warmupff"
        ]:
            self._init_optimizer()
        self.update_count += 1

        return {'loss': loss.item()}

    def modify_features(self, feats):
        return feats

    def predict(self, x):
        return {"": self.network(x)}

    ## DiWA for saving initialization ##
    def save_path_for_future_init(self, path_for_save):
        assert not os.path.exists(path_for_save), f"The initialization: {path_for_save} has already been saved"
        if os.environ.get('SAVE_FEATURES_CLASSIFIERS', "0") != "0":
            # for algorithms inference with decouplage of classifiers and network
            print(f"Save features extractor and classifier at {path_for_save}")
            network = nn.Sequential(self.featurizer, self.classifier)
            torch.save(network.state_dict(), path_for_save)
        elif os.environ.get("SAVE_ONLY_FEATURES", "0") != "0":
            print(f"Save only features extractor at {path_for_save}")
            torch.save(self.featurizer.state_dict(), path_for_save)
        else:
            print(f"Save whole network at {path_for_save}")
            torch.save(self.network.state_dict(), path_for_save)

class TWA(ERM):
    """
    Trainable weight average
    """

    def _create_network(self):
        ERM._create_network(self,)
        self.update_count = 0
        self._use_lambdas = True
        _featurizers_aux_paths = self.hparams["featurizers_aux"].split(" ")
        self.featurizers_aux = [
            self._load_featurizer_aux(aux_path)
            for aux_path in _featurizers_aux_paths
            if aux_path != ""
        ]
        if self.hparams["featurizers_lambdas"] == 0:
            self.featurizers_lambdas = [0. for _ in range(len(self.featurizers_aux) + 1)]
        elif self.hparams["featurizers_lambdas"].split("_")[0] == "rand":
            mutiplier = float(self.hparams["featurizers_lambdas"].split("_")[1])
            self.featurizers_lambdas = [
                (mutiplier * (random.random() - 0.5))
                for _ in range(len(self.featurizers_aux) + 1)
            ]
        else:
            self.featurizers_lambdas = [
                float(l)
                for l in self.hparams["featurizers_lambdas"].split(" ")
                ]
        self.hparams["featurizers_lambdas_begin"] = " ".join(
            ["{:.4f}".format(l) for l in self.featurizers_lambdas]
        )
        assert len(self.featurizers_lambdas) == len(self.featurizers_aux) + 1

        self.featurizers = [self.featurizer] + self.featurizers_aux
        self.num_featurizers = len(self.featurizers)
        self.lambdas = torch.tensor(
            [float(self.featurizers_lambdas[i]) for i in range(self.num_featurizers)], requires_grad=True)

    def _load_featurizer_aux(self, aux_path):
        print(f"Load auxiliary featurizer from: {aux_path}")
        featurizer = networks.Featurizer(self.input_shape, self.hparams)
        _aux_path = aux_path
        aux_path = misc.process_save_path(aux_path, self.hparams)
        if aux_path != 'imagenet':
            state_dict = torch.load(aux_path)
            state_dict = misc.clean_state_dict(state_dict, _aux_path)
            misc.load_featurizer(featurizer, state_dict)

        return featurizer

    def to(self, device):
        ERM.to(self, device)
        for featurizer in self.featurizers_aux:
            featurizer.to(device)

    def train(self, *args):
        ERM.train(self, *args)
        for featurizer in self.featurizers_aux:
            for param in featurizer.parameters():
                param.requires_grad = False

    def _get_training_parameters(self):

        if self._what_is_trainable in ["warmupnet", "warmupcla"]:
            print("No longer using lambdas, back to ERM")
            if self.update_count == self.hparams["warmup"]:
                what_is_trainable = "all"
                misc.set_weights(self.get_featurizer_wa_weights(), self.featurizer)
                self._use_lambdas = False
            else:
                assert self.update_count == 0
                what_is_trainable = "lambdascla" if self._what_is_trainable in ["warmupnet"] else "cla"
        else:
            what_is_trainable = self._what_is_trainable

        training_parameters = []

        if what_is_trainable in ["lambdas", "lambdascla"]:
            if len(self.lambdas) > 1:
                print("Learn lambdas")
                training_parameters.append({"params": [self.lambdas], "lr": self.hparams["lrl"]})
            else:
                print(f"Skip learning lambdas of len {len(self.lambdas)}")
                raise ValueError("Unexpected")

        if what_is_trainable in ["cla", "lambdascla"]:
            print("Learn classifier")
            training_parameters.append({"params": self.classifier.parameters()})
        elif what_is_trainable in ["all"]:
            print("Learn all network")
            assert not self._use_lambdas
            training_parameters.append({"params": self.network.parameters()})

        assert len(training_parameters)

        return training_parameters

    def compute_loss(self, logits, y):
        dict_loss = {}
        dict_loss["ce"] = nn.CrossEntropyLoss()(logits, y)
        # dict_loss["ent"] = misc.get_entropy_loss(logits)
        # dict_loss["bdi"] = misc.get_batchdiversity_loss(logits)
        return dict_loss

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, _ in minibatches])
        all_y = torch.cat([y for _, y in minibatches])
        all_logits = self.predict_in_train(all_x)
        dict_loss = self.compute_loss(all_logits, all_y)
        objective = (
            float(self.hparams["lossce"]) * dict_loss["ce"]
            # + float(self.hparams["lossent"]) * dict_loss["ent"]
            # + float(self.hparams["lossbdi"]) * dict_loss["bdi"]
            # float(self.hparams.get("coralloss", 0.)) * dict_loss.get("coral", 0.)
        )
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        if self.update_count == self.hparams["warmup"] and self._what_is_trainable in [
            "warmupnet", "warmupcla"
        ]:
            self._init_optimizer()

        self.update_count += 1

        results = {key: float(value.item()) for key, value in dict_loss.items()}

        if self._use_lambdas:
            for i in range(self.num_featurizers):
                results[f"lambda_{i}"] = float(self.lambdas[i].detach().float().cpu().numpy())
        return results

    def get_featurizer_wa_weights(self):
        return misc.get_wa_weights(
            torch.exp(self.lambdas),
            self.featurizers)

    def predict_in_train(self, x):
        if not self._use_lambdas:
            return self.network(x)
        wa_weights = self.get_featurizer_wa_weights()
        features_wa = torch.nn.utils.stateless.functional_call(
                self.featurizer, wa_weights, x)
        return self.classifier(features_wa)

    def predict(self, x):
        dict_preds = {"": self.predict_in_train(x)}
        return dict_preds

    def get_network_state_dict(self):
        featurizer = copy.deepcopy(self.featurizer)
        misc.set_weights(self.get_featurizer_wa_weights(), featurizer=featurizer)
        network = nn.Sequential(featurizer, self.classifier)
        return network.state_dict()

    def save_path_for_future_init(self, path_for_save):
        if not self._use_lambdas:
            return ERM.save_path_for_future_init(self, path_for_save)

        assert not os.path.exists(path_for_save), f"The initialization: {path_for_save} has already been saved"
        assert os.environ.get('SAVE_FEATURES_CLASSIFIERS', "0") == "0"
        assert os.environ.get("SAVE_ONLY_FEATURES", "0") == "0"
        state_dict = self.get_network_state_dict()
        print(f"Save wa network at {path_for_save}")
        torch.save(state_dict, path_for_save)


class TWAMA(TWA):
    """
    Empirical Risk Minimization (ERM) with Moving Average (MA) prediction model
    from https://arxiv.org/abs/2110.10832
    """

    def __init__(self, *args, **kwargs):
        TWA.__init__(self, *args, **kwargs)

        self.network_ma = copy.deepcopy(self.network)
        self.network_ma.eval()
        self.ma_start_iter = 100
        if self._what_is_trainable in ["warmupnet", "warmupcla"]:
            self.ma_start_iter += self.hparams["warmup"]
        self.ma_count = 0

    def update(self, *args, **kwargs):
        results = TWA.update(self, *args, **kwargs)
        MA.update_ma(self)
        return results

    def predict(self, x):
        dict_preds = {"": self.predict_in_train(x)}
        dict_preds["ma"] = self.network_ma(x)
        return dict_preds
    # def update_ma(self):
    #     # do something about lambdas

    def save_path_for_future_init(self, path_for_save):
        if not self._use_lambdas:
            return MA.save_path_for_future_init(self, path_for_save)

        assert not os.path.exists(path_for_save), f"The initialization: {path_for_save} has already been saved"
        assert os.environ.get('SAVE_FEATURES_CLASSIFIERS', "0") == "0"
        assert os.environ.get("SAVE_ONLY_FEATURES", "0") == "0"
        state_dict = self.get_network_state_dict()
        print(f"Save wa network at {path_for_save}")
        torch.save(state_dict, path_for_save)


## DiWA to reproduce moving average baseline ##
class MA(ERM):
    """
    Empirical Risk Minimization (ERM) with Moving Average (MA) prediction model
    from https://arxiv.org/abs/2110.10832
    """

    def __init__(self, *args, **kwargs):
        ERM.__init__(self, *args, **kwargs)

        self.network_ma = copy.deepcopy(self.network)
        self.network_ma.eval()
        self.ma_start_iter = 100
        if self._what_is_trainable in ["warmupnet", "warmupff", "warmupcla"]:
            self.ma_start_iter += self.hparams["warmup"]
        self.ma_count = 0

    def update(self, *args, **kwargs):
        results = ERM.update(self, *args, **kwargs)
        self.update_ma()
        return results

    def predict(self, x):
        # self.network_ma.eval()
        # I think this is not necessary
        dict_preds = {"ma": self.network_ma(x), "": self.network(x)}
        return dict_preds

    def update_ma(self):
        if self.update_count >= self.ma_start_iter:
            if self.update_count == self.ma_start_iter:
                print("Begin ma update at step: ", self.update_count)
            self.ma_count += 1
            for param_q, param_k in zip(self.network.parameters(), self.network_ma.parameters()):
                param_k.data = (param_k.data * self.ma_count + param_q.data) / (1. + self.ma_count)
        else:
            for param_q, param_k in zip(self.network.parameters(), self.network_ma.parameters()):
                param_k.data = param_q.data

    ## DiWA for saving initialization ##
    def save_path_for_future_init(self, path_for_save):
        assert not os.path.exists(path_for_save), f"The initialization: {path_for_save} has already been saved"
        assert os.environ.get('SAVE_FEATURES_CLASSIFIERS', "0") == "0"

        print(f"Save wa network at {path_for_save}")
        state_dict = self.network_ma.state_dict()

        if os.environ.get("SAVE_ONLY_FEATURES", "0") != "0":
            state_dict = {key.replace("0.network", "network"): value for key, value in state_dict.items() if key not in ["1.weight", "1.bias"]}

        torch.save(state_dict, path_for_save)


class ERMG(ERM):
    """
    Empirical Risk Minimization (ERM) games
    """

    def _init_optimizer(self):
        self.optimizers = []
        for _ in range(self.num_domains):
            training_parameters = self._get_training_parameters()
            self.optimizers.append(
                torch.optim.Adam(
                    training_parameters,
                    lr=self.hparams["lr"],
                    weight_decay=self.hparams['weight_decay']
                )
            )

    def update(self, minibatches, unlabeled=None):
        losses = {}
        for domain in np.random.permutation(range(self.num_domains)):
            x = minibatches[domain][0]
            y = minibatches[domain][1]
            features = self.featurizer(x)
            if self._what_is_trainable in ["cla"]:
                features = features.detach()
            features = self.modify_features(features)
            loss = F.cross_entropy(self.classifier(features), y)

            self.optimizers[domain].zero_grad()
            loss.backward()
            self.optimizers[domain].step()
            losses["loss" + str(domain)] = loss.item()

        return losses


class ERMLasso(ERM):

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_features = self.featurizer(all_x)
        if self._what_is_trainable in ["cla"]:
            all_features = all_features.detach()
        all_features = self.modify_features(all_features)
        objective = F.cross_entropy(self.classifier(all_features), all_y)

        l1_reg = torch.tensor(0., requires_grad=True)

        for name, param in self.classifier.named_parameters():
            if 'weight' in name:
                l1_reg = l1_reg + torch.norm(param, 1)

        loss = objective + self.hparams["l1_reg"] * l1_reg

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item(), "objective": objective.item(), "l1_reg": l1_reg.item()}


class SD(ERM):
    """
    Gradient Starvation: A Learning Proclivity in Neural Networks
    Equation 25 from [https://arxiv.org/pdf/2011.09468.pdf]
    """

    def __init__(self, *args, **kwargs):
        ERM.__init__(self, *args, **kwargs)
        self.sd_reg = self.hparams["sd_reg"]

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        all_x = torch.cat([x for x, y in minibatches])
        all_features = self.featurizer(all_x)
        if self._what_is_trainable in ["cla"]:
            all_features = all_features.detach()
        all_features = self.modify_features(all_features)

        all_p = self.classifier(all_features)

        loss = F.cross_entropy(all_p, all_y)
        penalty = (all_p**2).mean()
        objective = loss + self.sd_reg * penalty

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'penalty': penalty.item()}



class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, *args, **kwargs):
        ERM.__init__(self, *args, **kwargs)

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        penalty_weight = (
            self.hparams['irm_lambda']
            if self.update_count >= self.hparams['irm_penalty_anneal_iters'] else 1.0
        )
        nll = 0.
        penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_features = self.featurizer(all_x)
        if self._what_is_trainable in ["cla"]:
            all_features = all_features.detach()
        all_logits = self.classifier(all_features)

        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(), 'penalty': penalty.item()}


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""

    def __init__(self, *args, **kwargs):
        ERM.__init__(self, *args, **kwargs)

    def update(self, minibatches, unlabeled=None):
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0

        nll = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_features = self.featurizer(all_x)
        if self._what_is_trainable in ["cla"]:
            all_features = all_features.detach()
        all_logits = self.classifier(all_features)

        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean)**2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams['vrex_penalty_anneal_iters']:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(), 'penalty': penalty.item()}


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """

    def update(self, minibatches, unlabeled=None):
        objective = 0

        for (xi, yi), (xj, yj) in misc.random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"], self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            features = self.featurizer(x)
            if self._what_is_trainable in ["cla"]:
                features = features.detach()
            predictions = self.classifier(features)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """

    def __init__(self, *args, **kwargs):
        ERM.__init__(self, *args, **kwargs)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            features = self.featurizer(x)
            if self._what_is_trainable in ["cla"]:
                features = features.detach()
            losses[m] = F.cross_entropy(self.classifier(features), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class DARE(ERM):
    """
    Perform ERM while removing mean covariance
    """
    norm_or_swap = "norm"

    def __init__(self, *args, **kwargs):
        ERM.__init__(self, *args, **kwargs)
        assert self._what_is_trainable

        self.register_buffer(
            "mean_per_domain", torch.zeros(self.num_domains, self.featurizer.n_outputs)
        )
        self.register_buffer(
            "var_per_domain", torch.ones(self.num_domains, self.featurizer.n_outputs)
        )
        # self.register_buffer(
        #     "cov_per_domain",
        #     torch.ones(self.num_domains, self.featurizer.n_outputs, self.featurizer.n_outputs)
        # )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_features = self.featurizer(all_x)
        if self._what_is_trainable in ["cla"]:
            all_features = all_features.detach()

        penalty = torch.tensor(0.)
        idx = 0
        list_normalized_features = []
        for i, (x, y) in enumerate(minibatches):
            features_i = all_features[idx:idx + x.shape[0]]
            idx += x.shape[0]
            mean_features_i = torch.mean(features_i, dim=0)
            self.mean_per_domain.data[i] = (
                self.hparams['ema'] * self.mean_per_domain[i] +
                (1 - self.hparams['ema']) * mean_features_i.detach()
            )
            centered_features_i = features_i - self.mean_per_domain[i]
            var_features_i = torch.mean(centered_features_i**2, dim=0)

            self.var_per_domain.data[i] = (
                self.hparams['ema'] * self.var_per_domain[i] +
                (1 - self.hparams['ema']) * var_features_i.detach()
            )
            var_domain = 0.9 * self.var_per_domain[i] + 0.1 * torch.ones_like(
                self.var_per_domain[i]
            )
            normalized_features_i = centered_features_i * torch.pow(var_domain, -1 / 2)

            list_normalized_features.append(normalized_features_i)

            # for covariance: left todo
            # torch.diag_embed(
            # normalized_features_i = torch.matmul(centered_features_i, var_domain)
            # transforms.Normalize(2, 0.5)(t)

        all_normalized_features = torch.cat(list_normalized_features)
        if self.norm_or_swap == "swap":
            dist = Normal(
                torch.mean(self.mean_per_domain, dim=0),
                scale=torch.pow(torch.mean(self.var_per_domain, dim=0), 1 / 2)
            )
            all_normalized_features = all_normalized_features + dist.sample(
                torch.Size([all_normalized_features.shape[0]])
            )

        all_logits = self.classifier(all_normalized_features)

        nll = F.cross_entropy(all_logits, all_y)

        loss = nll + self.hparams['lambda'] * penalty
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        dict_output = {'loss': loss.item(), 'nll': nll.item(), 'penalty': penalty.item()}
        # cos = lambda m: F.normalize(m) @ F.normalize(m).t()
        # dict_output["cosine_var"] = cos(self.var_per_domain.data).flatten()
        # dict_output["cosine_mean"] = cos(self.mean_per_domain.data).flatten()
        return dict_output

    def predict(self, x):
        dict_predictions = {}
        features = self.featurizer(x)
        dict_predictions[""] = self.classifier(features)
        centered_features = features - torch.mean(self.mean_per_domain, dim=0)
        var = 0.9 * torch.mean(self.var_per_domain,
                               dim=0) + 0.1 * torch.ones_like(self.var_per_domain[0])
        normalized_features = centered_features * torch.pow(var, -1 / 2)
        dict_predictions["dare"] = self.classifier(normalized_features)
        return dict_predictions


class DARESWAP(DARE):
    norm_or_swap = "swap"


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """

    def __init__(self, gaussian, *args, **kwargs):
        ERM.__init__(self, *args, **kwargs)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1),
                          alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma'] * penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, *args, **kwargs):
        AbstractMMD.__init__(self, True, *args, **kwargs)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, *args, **kwargs):
        AbstractMMD.__init__(self, False, *args, **kwargs)


class Fishr(Algorithm):
    "Invariant Gradients variances for Out-of-distribution Generalization"

    def __init__(
        self,
        input_shape,
        num_classes,
        num_domains,
        hparams,
        what_is_trainable=False,
        path_for_init=None
    ):
        assert backpack is not None, "Install backpack with: 'pip install backpack-for-pytorch==1.3.0'"
        super(Fishr, self).__init__(input_shape, num_classes, num_domains, hparams)

        self._what_is_trainable = what_is_trainable
        self._create_network()
        self._load_network(path_for_init)

        self.update_count = 0
        self.bce_extended = extend(nn.CrossEntropyLoss(reduction='none'))
        self.ema_per_domain = [
            misc.MovingAverage(ema=self.hparams["ema"], oneminusema_correction=True)
            for _ in range(self.num_domains)
        ]
        self._init_optimizer()

    def _create_network(self):
        self.featurizer = networks.Featurizer(self.input_shape, self.hparams)
        self.classifier = extend(
            networks.Classifier(
                self.featurizer.n_outputs,
                self.num_classes,
                self.hparams['nonlinear_classifier'],
            )
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)

    def _load_network(self, path_for_init):
        return ERM._load_network(self, path_for_init)

    def _init_optimizer(self):
        ERM._init_optimizer(self)

    def _get_training_parameters(self):
        return ERM._get_training_parameters(self)

    def update(self, minibatches, unlabeled=False):
        assert len(minibatches) == self.num_domains
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        len_minibatches = [x.shape[0] for x, y in minibatches]

        all_z = self.featurizer(all_x)
        if self._what_is_trainable in ["cla"]:
            all_z = all_z.detach()
        all_logits = self.classifier(all_z)

        penalty = self.compute_fishr_penalty(all_logits, all_y, len_minibatches)
        all_nll = F.cross_entropy(all_logits, all_y)

        penalty_weight = 0
        if self.update_count >= self.hparams["penalty_anneal_iters"]:
            penalty_weight = self.hparams["lambda"]
            if self.update_count == self.hparams["penalty_anneal_iters"] != 0:
                # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
                # gradient magnitudes that happens at this step.
                self._init_optimizer()
        self.update_count += 1

        objective = all_nll + penalty_weight * penalty
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item(), 'nll': all_nll.item(), 'penalty': penalty.item()}

    def compute_fishr_penalty(self, all_logits, all_y, len_minibatches):
        dict_grads = self._get_grads(all_logits, all_y)
        grads_var_per_domain = self._get_grads_var_per_domain(dict_grads, len_minibatches)
        return self._compute_distance_grads_var(grads_var_per_domain)

    def _get_grads(self, logits, y):
        self.optimizer.zero_grad()
        loss = self.bce_extended(logits, y).sum()
        with backpack(BatchGrad()):
            loss.backward(
                inputs=list(self.classifier.parameters()), retain_graph=True, create_graph=True
            )

        # compute individual grads for all samples across all domains simultaneously
        dict_grads = OrderedDict(
            [
                (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                for name, weights in self.classifier.named_parameters()
            ]
        )
        return dict_grads

    def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
        # grads var per domain
        grads_var_per_domain = [{} for _ in range(self.num_domains)]
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id, bsize in enumerate(len_minibatches):
                env_grads = _grads[all_idx:all_idx + bsize]
                all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)

        # moving average
        for domain_id in range(self.num_domains):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
                grads_var_per_domain[domain_id]
            )

        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):

        # compute gradient variances averaged across domains
        grads_var = OrderedDict(
            [
                (
                    name,
                    torch.stack(
                        [
                            grads_var_per_domain[domain_id][name]
                            for domain_id in range(self.num_domains)
                        ],
                        dim=0
                    ).mean(dim=0)
                )
                for name in grads_var_per_domain[0].keys()
            ]
        )

        penalty = 0
        for domain_id in range(self.num_domains):
            penalty += misc.l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_domains

    def predict(self, x):
        return {"": self.network(x)}
