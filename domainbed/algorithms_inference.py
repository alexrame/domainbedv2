import os
import copy
import pdb
import numpy as np
import math
import torch
import torch.nn as nn
import collections
from domainbed import networks, algorithms
from domainbed.lib import misc, diversity_metrics


class ERM(algorithms.ERM):

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        algorithms.Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams)
        self.num_classes = num_classes
        self._create_network()

    def _create_network(self):
        self.featurizer = networks.Featurizer(self.input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs, self.num_classes,
            self.hparams.get('nonlinear_classifier', False)
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.network_ma = copy.deepcopy(self.network)
        self.network_ma2 = copy.deepcopy(self.network)


class DiWA(algorithms.ERM):

    def __init__(self, input_shape, num_classes, num_domains, hparams={}):
        """
        """
        algorithms.Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams=hparams)
        self._init_counts()
        self._create_network()
        self.domain_to_mean_feats = {}
        self.domain_to_cov_feats = {}

    def _init_counts(self):
        self.global_count = 0
        self.global_count_feat = 0
        self.global_count_cla = 0

        self.global_count_product = 0
        self.global_count_feat_product = 0
        self.global_count_cla_product = 0

        self.global_count_ma = 0
        self.global_count_var = 0

    def _create_network(self):
        self.network = None
        self.featurizer = None
        self.classifier = None

        self.network_product = None
        self.featurizer_product = None
        self.classifier_product = None

        self.network_ma = None
        self.network_var = None
        self.networks = []
        self.networks_wa = []

        self.featurizers = []
        self.featurizers_weights = []
        self.classifiers = []
        self.classifiers_weights = []

    @staticmethod
    def _update_product(net0, net1, weight0, weight1):
        if weight0 + weight1 == 0:
            return

        merging_method_all = os.environ.get("MERGINGMETHOD", "product")
        if merging_method_all == "sample":
            merging_method_all = np.random.choice([
                "mean",
                "product",
                "max",
                "min",
                "rand",
            ])

        for param_0, param_1 in zip(net0.parameters(), net1.parameters()):
            if merging_method_all == "sampleperlayer":
                merging_method = np.random.choice([
                    "mean",
                    "product",
                    "max",
                    "min",
                    "rand",
                ])
            else:
                merging_method = merging_method_all

            if merging_method == "mean":
                param_1.data = (param_1.data * weight1 +
                                param_0.data * weight0) / (weight0 + weight1)
            else:
                mask = (torch.sign(param_0.data) == torch.sign(param_1.data))
                if merging_method == "product":
                    product = torch.pow(torch.abs(param_0.data),
                                        weight0) * torch.pow(torch.abs(param_1.data), weight1)
                    new_data = mask * torch.sign(param_1.data
                                                ) * torch.pow(product, 1 / (weight0 + weight1))
                elif merging_method == "max":
                    new_data = torch.max(torch.abs(param_0.data), torch.abs(param_1.data))
                elif merging_method == "min":
                    new_data = torch.min(torch.abs(param_0.data), torch.abs(param_1.data))
                elif merging_method == "rand":
                    if np.random.random() < weight0 / (weight0 + weight1):
                        new_data = torch.abs(param_0.data)
                    else:
                        new_data = torch.abs(param_1.data)
                else:
                    raise ValueError("Unknown merging type: " + merging_method)

                param_1.data = mask * torch.sign(param_1.data) * new_data

    @staticmethod
    def _update_mean(net0, net1, weight0, weight1):
        if weight0 + weight1 == 0:
            return
        for param_0, param_1 in zip(net0.parameters(), net1.parameters()):
            param_1.data = (param_1.data * weight1 + param_0.data * weight0) / (weight0 + weight1)

    def update_mean_network(self, network, weight=1.):
        if self.network is None:
            self.network = copy.deepcopy(network)
        self._update_mean(network, self.network, weight, self.global_count)
        self.global_count += weight

    def update_mean_featurizer(self, network, weight=1.):
        if self.featurizer is None:
            self.featurizer = copy.deepcopy(network)
        self._update_mean(network, self.featurizer, weight, self.global_count_feat)
        self.global_count_feat += weight

    def update_mean_classifier(self, classifier, weight=1.):
        if self.classifier is None:
            self.classifier = copy.deepcopy(classifier)
        self._update_mean(classifier, self.classifier, weight, self.global_count_cla)
        self.global_count_cla += weight

    def update_product_network(self, network, weight=1.):
        if self.network_product is None:
            self.network_product = copy.deepcopy(network)
        self._update_product(network, self.network_product, weight, self.global_count_product)
        self.global_count_product += weight

    def update_product_featurizer(self, network, weight=1.):
        if self.featurizer_product is None:
            self.featurizer_product = copy.deepcopy(network)
        self._update_product(
            network, self.featurizer_product, weight, self.global_count_feat_product
        )
        self.global_count_feat_product += weight

    def update_product_classifier(self, classifier, weight=1.):

        if self.classifier_product is None:
            self.classifier_product = copy.deepcopy(classifier)
        self._update_product(
            classifier, self.classifier_product, weight, self.global_count_cla_product
        )
        self.global_count_cla_product += weight

    def update_mean_network_ma(self, network, weight=1.):
        if self.network_ma is None:
            self.network_ma = copy.deepcopy(network)

        self._update_mean(network, self.network_ma, weight, self.global_count_ma)
        self.global_count_ma += weight

    def update_var_network(self, network):
        if self.network_var is None:
            self.network_var = copy.deepcopy(network)
        for param_n, param_m, param_v in zip(
            network.parameters(), self.network.parameters(), self.network_var.parameters()
        ):
            l2_network_meannetwork = (param_n.data - param_m.data)**2
            param_v.data = (param_v.data * self.global_count_var +
                            l2_network_meannetwork) / (1. + self.global_count_var)
        self.global_count_var += 1

    def create_stochastic_networks(self):
        assert len(self.networks) == 0
        multiplier = float(os.environ.get("VARM", 1.))
        for i in range(int(os.environ.get("MAXM", 3))):
            network = copy.deepcopy(self.network)
            for param_n, param_m, param_v in zip(
                network.parameters(), self.network.parameters(), self.network_var.parameters()
            ):
                param_n.data = torch.normal(
                    mean=param_m.data,
                    std=multiplier *
                    torch.sqrt(self.global_count_var / (self.global_count_var - 1) * param_v.data)
                )
                # param_n.data = param_m.data + * gaussian_noise
            self.add_network(network)

    def add_network(self, network):
        self.networks.append(network)

    def add_featurizer(self, network, weight):
        self.featurizers.append(network)
        self.featurizers_weights.append(weight)

    def add_classifier(self, classifier, weight=1):
        self.classifiers.append(classifier)
        self.classifiers_weights.append(weight)

    def predict_feat(self, x):
        dict_features = {}
        if len(self.featurizers) != 0:
            for i, featurizer in enumerate(self.featurizers):
                if i < float(os.environ.get("MAXM", math.inf)):
                    dict_features["net" + str(i)] = featurizer(x)
        return dict_features

    def predict(self, x, **kwargs):
        dict_predictions = {}
        if (self.classifier is None or os.environ.get("NETWORKINFERENCE", "0") == "1") and self.network is not None:
            dict_predictions[""] = self.network(x)
        if self.network_ma is not None:
            dict_predictions["ma"] = self.network_ma(x)
        if self.network_product is not None:
            dict_predictions["prod"] = self.network_product(x)
            if "" in dict_predictions:
                dict_predictions["ensprod"] = torch.mean(
                    torch.stack([dict_predictions[""], dict_predictions["prod"]], dim=0), 0
                )

        if len(self.networks) != 0:
            logits_ens = []
            len_network = len(self.networks)
            for i in range(len_network):
                _logits_i = self.networks[i](x)
                logits_ens.append(_logits_i)
                if i < float(os.environ.get("MAXM", math.inf)):
                    dict_predictions["net" + str(i)] = _logits_i
            dict_predictions["ens"] = torch.mean(torch.stack(logits_ens, dim=0), 0)
            if os.environ.get("SUBWA", "0") != "0":
                dict_predictions["ens01"] = torch.mean(torch.stack([logits_ens[0], logits_ens[1]], dim=0), 0)
                dict_predictions["ens12"] = torch.mean(torch.stack([logits_ens[1], logits_ens[2]], dim=0), 0)
                dict_predictions["ens23"] = torch.mean(torch.stack([logits_ens[2], logits_ens[3]], dim=0), 0)

        if os.environ.get("SUBWA", "0") != "0":
            assert len(self.networks) == 4
            for subwa in [("0", "1"), ("1", "2"), ("2", "3")]:
                lambdas_subwa = [
                    1. if str(key) in subwa else 0. for key in range(len(self.networks))
                ]
                wa_weights = misc.get_wa_weights(lambdas_subwa, featurizers=self.networks)
                dict_predictions[f"wa{''.join(subwa)}"] = torch.nn.utils.stateless.functional_call(
                    self.network, wa_weights, x
                )

        if self.featurizer is not None:
            logits_enscla = []
            features = self.featurizer(x)
            dict_predictions["feats"] = features
            if self.classifier is not None:
                dict_predictions["cla"] = self.classifier(features)

            if self.classifier_product is not None:
                dict_predictions["clameanprod"] = self.classifier_product(features)

            if len(self.classifiers) != 0:
                for i, classifier in enumerate(self.classifiers):
                    _logits_i = classifier(features)
                    logits_enscla.append(_logits_i * self.classifiers_weights[i])
                    # dict_predictions["cla" + str(i)] = _logits_i
                sum_weights = np.sum(self.classifiers_weights)
                dict_predictions["enscla"
                                ] = torch.sum(torch.stack(logits_enscla, dim=0), 0) / sum_weights

        if self.featurizer_product is not None:
            features_product = self.featurizer_product(x)
            if self.classifier is not None:
                dict_predictions["claprodmean"] = self.classifier(features_product)
            if self.classifier_product is not None:
                dict_predictions["claprod"] = self.classifier_product(features_product)

        if kwargs.get("lambdas") is not None:
            wa_weights = misc.get_wa_weights(
                lambda_interpolation=[1.] + kwargs.get("lambdas"),
                featurizers=[self.featurizer] + self.featurizers
            )
            features_wa = torch.nn.utils.stateless.functional_call(self.featurizer, wa_weights, x)
            dict_predictions["feats"] = features_wa
            dict_predictions["clawa"] = self.classifier(features_wa)

        return dict_predictions

    def train(self, *args):
        algorithms.ERM.train(self, *args)
        for network in self.networks:
            network.train(*args)
        for featurizer in self.featurizers:
            featurizer.train(*args)
        for classifier in self.classifiers:
            classifier.train(*args)

    def to(self, device):
        algorithms.ERM.to(self, device)
        for network in self.networks:
            network.to(device)
        for featurizer in self.featurizers:
            featurizer.to(device)
        for classifier in self.classifiers:
            classifier.to(device)

    def get_dict_prediction_stats(
        self,
        loader,
        device,
        what=[],
        predict_kwargs={},
    ):

        dict_stats = {}
        aux_dict_stats = {}
        with torch.no_grad():
            i = 0.
            for count_batch, (x, y) in enumerate(loader):
                x = x.to(device)
                bs = x.size(0)
                prediction = self.predict(x, **predict_kwargs)

                if "mean_feats" in what:
                    feats = prediction["feats"]
                    mean_feats, cov_feats = self.get_mean_cov_feats(
                        feats,
                        true_mean=self.domain_to_mean_feats.get(predict_kwargs.get("domain"))
                    )
                    if "mean_feats" not in aux_dict_stats:
                        aux_dict_stats["mean_feats"] = torch.zeros_like(mean_feats)
                    aux_dict_stats["mean_feats"] = (
                        aux_dict_stats["mean_feats"] * i + mean_feats * bs
                    ) / (i + bs)

                    if "cov_feats" in what:
                        if "cov_feats" not in aux_dict_stats:
                            aux_dict_stats["cov_feats"] = torch.zeros_like(cov_feats)
                        aux_dict_stats["cov_feats"] = (
                            aux_dict_stats["cov_feats"] * i + cov_feats * bs
                        ) / (i + bs)

                    if "l2_feats" in what:
                        for domain in self.domain_to_mean_feats.keys():
                            domain_feats = self.domain_to_mean_feats[domain].reshape(1, -1).tile(
                                (bs, 1)
                            )
                            l2_feats = (feats - domain_feats).pow(2).mean()
                            distkey = "l2_" + domain
                            if distkey not in aux_dict_stats:
                                aux_dict_stats[distkey] = 0.
                            aux_dict_stats[distkey] = (aux_dict_stats[distkey] * i +
                                                       l2_feats * bs) / (i + bs)

                    if "l2var_feats" in what:
                        var_feats = torch.diag(
                            self.domain_to_cov_feats[predict_kwargs.get("domain")]
                        ).reshape(1, -1).tile((bs, 1))
                        for domain in self.domain_to_mean_feats.keys():
                            domain_feats = self.domain_to_mean_feats[domain].reshape(1, -1).tile(
                                (bs, 1)
                            )
                            l2_feats = torch.div((feats - domain_feats).pow(2), var_feats).mean()
                            distkey = "l2var_" + domain
                            if distkey not in aux_dict_stats:
                                aux_dict_stats[distkey] = 0.
                            aux_dict_stats[distkey] = (aux_dict_stats[distkey] * i +
                                                       l2_feats * bs) / (i + bs)

                    if "cos_feats" in what:
                        for domain in self.domain_to_mean_feats.keys():
                            domain_feats = self.domain_to_mean_feats[domain].reshape(1, -1).tile(
                                (bs, 1)
                            )
                            l2_feats = nn.CosineSimilarity(dim=1)(feats, domain_feats).mean()
                            distkey = "cos_" + domain
                            if distkey not in aux_dict_stats:
                                aux_dict_stats[distkey] = 0.
                            aux_dict_stats[distkey] = (aux_dict_stats[distkey] * i +
                                                       l2_feats * bs) / (i + bs)

                    # todo cos and l2varfeats

                i += float(bs)
                y = y.to(device)

                if "classes" in what:
                    if "batch_classes" not in aux_dict_stats:
                        aux_dict_stats["batch_classes"] = []
                    aux_dict_stats["batch_classes"].append(y)

                for key in prediction.keys():
                    if key in ["feats"]:
                        continue
                    logits = prediction[key]
                    if key not in dict_stats:
                        dict_stats[key] = {
                            "logits": [],
                            # "probs": [],
                            "preds": [],
                            "correct": [],
                            "tcp": []
                            # "confs": [],
                        }
                    preds = logits.argmax(1)
                    probs = torch.softmax(logits, dim=1)
                    dict_stats[key]["logits"].append(logits.cpu())
                    # dict_stats[key]["probs"].append(probs.cpu())
                    dict_stats[key]["preds"].append(preds.cpu())
                    dict_stats[key]["correct"].append(preds.eq(y).float().cpu())

                    dict_stats[key]["tcp"].append(
                        probs[range(len(torch.flatten(y))),
                              torch.flatten(y)].flatten().cpu()
                    )
                    # dict_stats[key]["confs"].append(probs.max(dim=1)[0].cpu())
                if os.environ.get("DEBUG"):
                    pdb.set_trace()
                    break

                if count_batch < float(os.environ.get("DIVFEATS", "10")):
                    dict_feats = self.predict_feat(x)
                    for key in dict_feats.keys():
                        if "feats" not in dict_stats[key]:
                            dict_stats[key]["feats"] = []
                        dict_stats[key]["feats"].append(dict_feats[key])

        for key0 in dict_stats:
            for key1 in dict_stats[key0]:
                dict_stats[key0][key1] = torch.cat(dict_stats[key0][key1])
            dict_stats[key0]["acc"] = sum(dict_stats[key0]["correct"].numpy()
                                         ) / len(dict_stats[key0]["correct"].numpy())

        return dict_stats, aux_dict_stats

    def get_dict_diversity(self, dict_stats, targets, device, divregex="net"):
        dict_diversity = collections.defaultdict(list)
        # num_classifiers = int(min(len(self.classifiers), float(os.environ.get("MAXM", math.inf))))
        num_members = int(min(len(self.networks), float(os.environ.get("MAXM", math.inf))))
        # regexes = [("waens", "wa_ens"), ("waprod", "wa_prod")]
        # regexes = [("netcla0", "net0_cla0"), ("netcla1", "net1_cla1"), ("netcla2", "net2_cla2")]
        # regexes = [
        #     (f"cla{i}{j}", f"cla{i}_cla{j}")
        #     for i in range(num_classifiers)
        #     for j in range(i + 1, num_classifiers)
        # ]
        if os.environ.get("SUBWA", "0") != "0":
            regexes = [
                ("net01", f"net0_net1"),
                ("net12", f"net1_net2"),
                ("net23", f"net2_net3"),
            ]
        elif divregex == "net":
            regexes = [
                ("netm", f"net{i}_net{j}")
                for i in range(num_members)
                for j in range(i + 1, num_members)
            ]
        elif divregex == "nethalf":
            regexes = [
                ("netm", f"net{i}_net{j}")
                for i in range(num_members // 2)
                for j in range(num_members // 2 + 1, num_members)
            ]
            regexes += [
                ("net0", f"net{i}_net{j}")
                for i in range(num_members // 2)
                for j in range(i + 1, num_members // 2)
            ]
            regexes += [
                ("net1", f"net{i}_net{j}")
                for i in range(num_members // 2, num_members)
                for j in range(i + 1, num_members)
            ]
        else:
            raise ValueError()

        for regexname, regex in regexes:
            key0, key1 = regex.split("_")

            if key0 == "wa" and "wa" not in dict_stats:
                key0 = ""
            if key0 not in dict_stats or key1 not in dict_stats:
                continue

            preds0 = dict_stats[key0]["preds"].numpy()
            preds1 = dict_stats[key1]["preds"].numpy()
            dict_diversity[f"divr_{regexname}"].append(
                diversity_metrics.ratio_errors(targets, preds0, preds1)
            )
            dict_diversity[f"divq_{regexname}"].append(
                diversity_metrics.Q_statistic(targets, preds0, preds1)
            )
            dict_diversity[f"divd_{regexname}"].append(
                diversity_metrics.double_fault(targets, preds0, preds1)
            )

            if os.environ.get("DIVFEATS", "10") != "0" and "feats" in dict_stats[
                key0] and "feats" in dict_stats[key1]:
                feats0 = dict_stats[key0]["feats"]
                feats1 = dict_stats[key1]["feats"]
                dict_diversity[f"divf_{regexname}"
                              ] = 1. - diversity_metrics.CudaCKA(device).linear_CKA(feats0,
                                                                                    feats1).item()

        dict_results = {key: np.mean(value) for key, value in dict_diversity.items()}

        if num_members > 0:
            # cf https://arxiv.org/abs/2110.13786
            def div_pac(row):
                max_row = np.max(row)
                normalized_row = [r / (math.sqrt(2) * max_row + 1e-7) for r in row]
                return np.var(normalized_row)

            tcps = [dict_stats[f"net{i}"]["tcp"].numpy() for i in range(num_members)]
            tcps = np.stack(tcps, axis=1)
            dict_results["divp_netm"] = np.mean(np.apply_along_axis(div_pac, 1, tcps))

        return dict_results

    def get_mean_cov_feats(self, x0, true_mean=None):
        # x0 = torch.mean(feats_0, dim=0)
        mean_x0 = x0.mean(0, keepdim=True)
        if true_mean is None:
            cent_x0 = x0 - mean_x0
            cova_x0 = (cent_x0.t() @ cent_x0) / (len(x0) - 1)
        else:
            cent_x0 = x0 - true_mean
            cova_x0 = (cent_x0.t() @ cent_x0) / len(x0)
        return mean_x0, cova_x0


from torch import nn, optim


class TrainableDiWA(DiWA):
    # hparams: celoss, entloss, bdiloss, coralloss, lrl, lrc, nsteps (1000)

    def _create_network(self):
        self.featurizer = None
        self.classifier = None
        self.featurizers = []
        self.featurizers_weights = []
        self.networks = []

    def train(self, *args):
        algorithms.ERM.train(self, *args)
        for featurizer in self.featurizers:
            featurizer.train(*args)

    def to(self, device):
        algorithms.ERM.to(self, device)
        for featurizer in self.featurizers:
            featurizer.to(device)

    def set_not_trainable(self):
        for net in [self.featurizer] + self.featurizers:
            for param in net.parameters():
                param.requires_grad = False

    def _init_lambdas(self):
        self.num_aux = len(self.featurizers)
        self.lambdas = torch.tensor(
            [0.] + [float(self.featurizers_weights[i]) for i in range(self.num_aux)],
            requires_grad=True
        )

    def _init_train(self):
        self._init_lambdas()
        lrl = self.hparams.get("lrl", 0.)
        lrc = self.hparams.get("lrc", 0.)
        if lrl != 0:
            self.optimizer_lambdas = optim.Adam([self.lambdas], lr=lrl)
            print('Load self.optimizer_lambdas')
        else:
            self.optimizer_lambdas = None

        if lrc != 0:
            self.optimizer_classifier = optim.Adam(self.classifier.parameters(), lr=lrc)
        else:
            self.optimizer_classifier = None

    def get_optimizer_at_step(self, step):
        if step % 2:
            return self.optimizer_classifier or self.optimizer_lambdas
        return self.optimizer_lambdas or self.optimizer_classifier

    def compute_loss(self, logits, y):
        dict_loss = {}
        dict_loss["ce"] = nn.CrossEntropyLoss()(logits, y)
        if self.hparams.get("entloss", 0.):
            dict_loss["ent"] = misc.get_entropy_loss(logits)
        if self.hparams.get("bdiloss", 0.):
            dict_loss["bdi"] = misc.get_batchdiversity_loss(logits)

        return dict_loss

    def compute_loss_t(self, feats_0, feats_1):
        mean_x0, cova_x0 = self.get_mean_cov_feats(feats_0)
        mean_x1, cova_x1 = self.get_mean_cov_feats(feats_1)
        dict_loss_t = {}
        dict_loss_t["coral"] = (mean_x0 - mean_x1).pow(2).mean()
        dict_loss_t["coralv"] = (cova_x0 - cova_x1).pow(2).mean()
        return dict_loss_t

    def predict(self, x, **kwargs):
        dict_predictions = {}

        if kwargs.get("lambdas") is not None:
            lambda_interpolation = [1.] + kwargs.get("lambdas")
        else:
            lambda_interpolation = torch.exp(self.lambdas)

        wa_weights = misc.get_wa_weights(
            lambda_interpolation=lambda_interpolation,
            featurizers=[self.featurizer] + self.featurizers
        )
        features_wa = torch.nn.utils.stateless.functional_call(self.featurizer, wa_weights, x)

        # w for wa, t for task
        dict_predictions["feats"] = features_wa
        dict_predictions["logits"] = self.classifier(features_wa)
        return dict_predictions

    def train_step(self, x, y, optimizer, xt=None, yt=None):
        optimizer.zero_grad()
        dict_predictions = self.predict(x)
        dict_loss = self.compute_loss(dict_predictions["logits"], y)

        if xt is not None:
            dict_predictions_t = self.predict(xt)
            dict_loss_t = self.compute_loss_t(
                dict_predictions["feats"], dict_predictions_t["feats"]
            )
            dict_loss.update(dict_loss_t)

        objective = (
            float(self.hparams.get("celoss", 0.)) * dict_loss["ce"] +
            float(self.hparams.get("entloss", 0.)) * dict_loss.get("ent", 0.) +
            float(self.hparams.get("bdiloss", 0.)) * dict_loss.get("bdi", 0.) +
            float(self.hparams.get("coralloss", 0.)) * dict_loss.get("coral", 0.)
        )
        # objective = torch.stack(list(loss.values()), dim=0).sum(dim=0)
        objective.backward(retain_graph=False)
        optimizer.step()
        return {key: value.item() for key, value in dict_loss.items()}

    def test_time_training(self, tta_loader, device, dict_data_loaders, train_loader=None):
        self.to(device)
        self.eval()
        self._init_train()
        self.set_not_trainable()

        iter_tta_loader = iter(tta_loader)
        if train_loader is not None:
            iter_train_loader = iter(train_loader)
        last_results_keys = []

        for step in range(0, self.hparams.get("nsteps", 1000) + 1):
            results = {'step': step}
            if step != 0:
                x, y = next(iter_tta_loader)
                x, y = x.to(device), y.to(device)
                if train_loader is not None:
                    xt, yt = next(iter_train_loader)
                    xt, yt = xt.to(device), yt.to(device)
                else:
                    xt, yt = None, None

                optimizer = self.get_optimizer_at_step(step)
                l = self.train_step(x, y, optimizer, xt, yt)
                results.update(l)

            for i in range(self.num_aux):
                results[f"lambda_{i}"] = self.lambdas[i].detach().float().cpu().numpy()

            if step % 10 == 0:
                for name, loader in dict_data_loaders.items():
                    if name in []:
                        print(f"Skip inference at {name}")
                        continue
                    else:
                        print(f"Inference at {name}")
                        _results_name, _ = misc.results_ensembling(
                            self, loader, device, do_div=False, do_ent=True
                        )
                        for key, value in _results_name.items():
                            new_key = name + "_" + key if name != "test" else key
                            results[new_key] = value
                        self.eval()
            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=20)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys], colwidth=20)
