import copy
import torch
import torch.nn as nn
from domainbed import networks, algorithms
from domainbed.lib import diversity_metrics

class ERM(algorithms.ERM):

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        algorithms.Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs, num_classes, self.hparams['nonlinear_classifier']
        )
        self.num_classes = num_classes
        self.network = nn.Sequential(self.featurizer, self.classifier)

class DiWA(algorithms.ERM):

    def __init__(self, input_shape, num_classes, num_domains):
        """
        """
        algorithms.Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams={})
        self.network = None
        self.networks = []
        self.global_count = 0

    def add_weights(self, network):
        if self.network is None:
            self.network = copy.deepcopy(network)
        else:
            for param_q, param_k in zip(network.parameters(), self.network.parameters()):
                param_k.data = (param_k.data * self.global_count + param_q.data) / (1. + self.global_count)
        self.global_count += 1

    def add_network(self, network):
        self.networks.append(network)

    def predict(self, x):
        dict_predictions = {"": self.network(x)}
        if len(self.networks) == 0:
            return dict_predictions

        logits_ens = []
        for i, network in enumerate(self.networks):
            _logits_i = network(x)
            logits_ens.append(_logits_i)
            dict_predictions["net" + str(i)] = _logits_i
        dict_predictions["ens"] = torch.mean(torch.stack(logits_ens, dim=0), 0)
        return dict_predictions

    def train(self, *args):
        algorithms.ERM.train(self, *args)
        for network in self.networks:
            network.train(*args)

    def to(self, device):
        algorithms.ERM.to(self, device)
        for network in self.networks:
            network.to(device)

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
                            # "confs": [],
                        }
                    preds = logits.argmax(1)
                    # probs = torch.softmax(logits, dim=1)
                    # dict_stats[key]["logits"].append(logits.cpu())
                    # dict_stats[key]["probs"].append(probs.cpu())
                    dict_stats[key]["preds"].append(preds.cpu())
                    dict_stats[key]["correct"].append(preds.eq(y).float().cpu())
                    # dict_stats[key]["confs"].append(probs.max(dim=1)[0].cpu())
        for key0 in dict_stats:
            for key1 in dict_stats[key0]:
                dict_stats[key0][key1] = torch.cat(dict_stats[key0][key1])
        return dict_stats, batch_classes

    def get_dict_diversity(self, dict_stats, targets, device):
        results = {}
        for regex in ["wa_ens", "net0_net1",]:
            key0, key1 = regex.split("_")

            if key0 not in dict_stats:
                if key0 == "wa":
                    key0 = ""
                else:
                    continue

            if key1 not in dict_stats:
                continue

            preds0 = dict_stats[key0]["preds"].numpy()
            preds1 = dict_stats[key1]["preds"].numpy()
            results[f"divr_{regex}"] = diversity_metrics.ratio_errors(
                targets, preds0, preds1
            )
            results[f"divq_{regex}"] = diversity_metrics.Q_statistic(
                targets, preds0, preds1
            )
        return results
