import os
import copy
import numpy as np
import math
import torch
import torch.nn as nn
import collections
from domainbed import networks, algorithms
from domainbed.lib import diversity_metrics


class ERM(algorithms.ERM):

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        algorithms.Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs, num_classes, self.hparams.get('nonlinear_classifier', False)
        )
        self.num_classes = num_classes
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.network_ma = copy.deepcopy(self.network)


class DiWA(algorithms.ERM):

    def __init__(self, input_shape, num_classes, num_domains):
        """
        """
        algorithms.Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams={})
        self.network = None
        self.network_ma = None
        self.featurizer = None
        self.var_network = None
        self.networks = []
        self.classifiers = []
        self.global_count = 0
        self.global_count_feat = 0
        self.global_count_ma = 0
        self.var_global_count = 0

    def update_mean_featurizer(self, featurizer, weight=1.):

        if self.featurizer is None:
            self.featurizer = copy.deepcopy(featurizer)

        for param_n, param_m in zip(featurizer.parameters(), self.featurizer.parameters()):
            param_m.data = (param_m.data * self.global_count_feat +
                            param_n.data * weight) / (weight + self.global_count_feat)

        self.global_count_feat += weight

    def update_mean_network(self, network, weight=1.):

        if self.network is None:
            self.network = copy.deepcopy(network)

        for param_n, param_m in zip(network.parameters(), self.network.parameters()):
            param_m.data = (param_m.data * self.global_count +
                            param_n.data * weight) / (weight + self.global_count)

        self.global_count += weight

    def update_mean_network_ma(self, network, weight=1.):

        if self.network_ma is None:
            self.network_ma = copy.deepcopy(network)

        for param_n, param_m in zip(network.parameters(), self.network_ma.parameters()):
            param_m.data = (param_m.data * self.global_count_ma +
                            param_n.data * weight) / (weight + self.global_count_ma)

        self.global_count_ma += weight

    def update_var_network(self, network):
        if self.var_network is None:
            self.var_network = copy.deepcopy(network)
        for param_n, param_m, param_v in zip(
            network.parameters(), self.network.parameters(), self.var_network.parameters()
        ):
            l2_network_meannetwork = (param_n.data - param_m.data)**2
            param_v.data = (param_v.data * self.var_global_count +
                            l2_network_meannetwork) / (1. + self.var_global_count)
        self.var_global_count += 1

    def create_stochastic_networks(self):
        assert len(self.networks) == 0
        multiplier = float(os.environ.get("VARM", 1.))
        for i in range(int(os.environ.get("MAXM", 3))):
            network = copy.deepcopy(self.network)
            for param_n, param_m, param_v in zip(
                network.parameters(), self.network.parameters(), self.var_network.parameters()
            ):
                param_n.data = torch.normal(
                    mean=param_m.data,
                    std=multiplier *
                    torch.sqrt(self.var_global_count / (self.var_global_count - 1) * param_v.data)
                )
                # param_n.data = param_m.data + * gaussian_noise
            self.add_network(network)

    def add_network(self, network):
        self.networks.append(network)

    def add_classifier(self, classifier):
        self.classifiers.append(classifier)
        if len(self.classifiers) != 2:
            return
        mask_classifier = copy.deepcopy(classifier)
        for param_m, param_0, param_1 in zip(
            mask_classifier.parameters(), self.classifiers[0].parameters(), self.classifiers[1].parameters()
        ):
            #  (param_0 + param_1)/2
            param_m.data = torch.minimum(param_0, param_1) * (param_0 * param_1 > 0).float()
        self.classifiers.append(mask_classifier)

    def predict(self, x):
        if self.network_ma is not None:
            dict_predictions = {"": self.network_ma(x)}
        else:
            dict_predictions = {"": self.network(x)}

        if len(self.networks) != 0:
            logits_ens = []
            for i, network in enumerate(self.networks):
                _logits_i = network(x)
                logits_ens.append(_logits_i)
                if i < float(os.environ.get("MAXM", math.inf)):
                    dict_predictions["net" + str(i)] = _logits_i
            dict_predictions["ens"] = torch.mean(torch.stack(logits_ens, dim=0), 0)

        if len(self.classifiers) != 0:
            logits_enscla = []
            # conflogits_enscla_minus1 = []
            # conflogits_enscla_minus2 = []
            features = self.featurizer(x)

            # def get_conf(logits, power=-1):
            #     probs = torch.softmax(logits, dim=1)
            #     ent = torch.sum(-probs * torch.log(probs), dim=1)
            #     conf = torch.reshape(torch.pow(ent, power), (-1, 1))
            #     return conf

            for i, classifier in enumerate(self.classifiers):
                _logits_i = classifier(features)
                logits_enscla.append(_logits_i)
                dict_predictions["cla" + str(i)] = _logits_i
            #     conflogits_enscla_minus1.append(get_conf(_logits_i, power=-1) * _logits_i)
            #     conflogits_enscla_minus2.append(get_conf(_logits_i, power=-2) * _logits_i)

            # dict_predictions["ensclaminus1"] = torch.mean(
            #     torch.stack(conflogits_enscla_minus1, dim=0), 0
            # )
            # dict_predictions["ensclaminus2"] = torch.mean(
            #     torch.stack(conflogits_enscla_minus2, dim=0), 0
            # )

        return dict_predictions

    def train(self, *args):
        algorithms.ERM.train(self, *args)
        for network in self.networks:
            network.train(*args)
        for classifier in self.classifiers:
            classifier.train(*args)

    def to(self, device):
        algorithms.ERM.to(self, device)
        for network in self.networks:
            network.to(device)
        for classifier in self.classifiers:
            classifier.to(device)

    def get_dict_prediction_stats(
        self,
        loader,
        device,
    ):
        batch_classes = []
        dict_stats = {}
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                prediction = self.predict(x)
                y = y.to(device)
                batch_classes.append(y)
                for key in prediction.keys():
                    logits = prediction[key]
                    if key not in dict_stats:
                        dict_stats[key] = {
                            # "logits": [],
                            # "probs": [],
                            "preds": [],
                            "correct": [],
                            "tcp": []
                            # "confs": [],
                        }
                    preds = logits.argmax(1)
                    probs = torch.softmax(logits, dim=1)
                    # dict_stats[key]["logits"].append(logits.cpu())
                    # dict_stats[key]["probs"].append(probs.cpu())
                    dict_stats[key]["preds"].append(preds.cpu())
                    dict_stats[key]["correct"].append(preds.eq(y).float().cpu())

                    dict_stats[key]["tcp"].append(
                        probs[range(len(torch.flatten(y))),
                              torch.flatten(y)].flatten().cpu()
                    )
                    # dict_stats[key]["confs"].append(probs.max(dim=1)[0].cpu())
        for key0 in dict_stats:
            for key1 in dict_stats[key0]:
                dict_stats[key0][key1] = torch.cat(dict_stats[key0][key1])
        return dict_stats, batch_classes

    def get_dict_diversity(self, dict_stats, targets, device):
        dict_diversity = collections.defaultdict(list)
        num_classifiers = min(len(self.classifiers), float(os.environ.get("MAXM", math.inf)))
        num_members = min(len(self.networks), float(os.environ.get("MAXM", math.inf)))
        # regexes = [("waens", "wa_ens")]
        # regexes = [("netcla0", "net0_cla0"), ("netcla1", "net1_cla1"), ("netcla2", "net2_cla2")]
        regexes = [
            (f"cla{i}{j}", f"cla{i}_cla{j}")
            for i in range(num_classifiers)
            for j in range(i + 1, num_classifiers)
        ]
        # regexes += [("netm", f"net{i}_net{j}") for i in range(num_members) for j in range(i + 1, num_members)]
        for regexname, regex in regexes:
            key0, key1 = regex.split("_")

            if key0 not in dict_stats:
                if key0 == "wa":
                    key0 = ""
                else:
                    print(f"Skip {regex}")
                    continue

            if key1 not in dict_stats:
                print(f"Skip {regex}")
                continue

            preds0 = dict_stats[key0]["preds"].numpy()
            preds1 = dict_stats[key1]["preds"].numpy()
            dict_diversity[f"divr_{regexname}"].append(
                diversity_metrics.ratio_errors(targets, preds0, preds1)
            )
            # dict_diversity[f"divq_{regexname}"].append(diversity_metrics.Q_statistic(
            #     targets, preds0, preds1
            # ))

        dict_results = {key: np.mean(value) for key, value in dict_diversity.items()}

        if num_members > 0 and False:
            # cf https://arxiv.org/abs/2110.13786
            tcps = [dict_stats[f"net{i}"]["tcp"].numpy() for i in range(num_members)]
            tcps = np.stack(tcps, axis=1)

            def div_pac(row):
                max_row = np.max(row)
                normalized_row = [r / (math.sqrt(2) * max_row + 1e-7) for r in row]
                return np.var(normalized_row)

            dict_results["divp_netm"] = np.mean(np.apply_along_axis(div_pac, 1, tcps))

        return dict_results
