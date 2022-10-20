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


class DiWA(algorithms.ERM):

    def __init__(self, input_shape, num_classes, num_domains):
        """
        """
        algorithms.Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams={})
        self._init_counts()
        self._create_network()

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
        self.featurizers = []
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
            param_1.data = (param_1.data * weight1 + param_0.data * weight0) / (
                weight0 + weight1)

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

    def add_featurizer(self, network):
        self.featurizers.append(network)

    def add_classifier(self, classifier, weight=1):
        self.classifiers.append(classifier)
        self.classifiers_weights.append(weight)

    def predict(self, x):
        if self.network_ma is not None:
            dict_predictions = {"": self.network_ma(x)}
        elif self.classifier is None or os.environ.get("NETWORKINFERENCE", "0") == "1":
            dict_predictions = {"": self.network(x)}
        else:
            dict_predictions = {}

        if self.network_product is not None:
            dict_predictions["prod"] = self.network_product(x)
            if "" in dict_predictions:
                dict_predictions["ensprod"] = torch.mean(
                    torch.stack([dict_predictions[""], dict_predictions["prod"]], dim=0), 0
                )

        if len(self.networks) != 0:
            logits_ens = []
            for i, network in enumerate(self.networks):
                _logits_i = network(x)
                logits_ens.append(_logits_i)
                if i < float(os.environ.get("MAXM", math.inf)):
                    dict_predictions["net" + str(i)] = _logits_i
            dict_predictions["ens"] = torch.mean(torch.stack(logits_ens, dim=0), 0)

        if self.featurizer is not None:
            logits_enscla = []
            features = self.featurizer(x)
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
                            "probs": [],
                            "preds": [],
                            "correct": [],
                            # "tcp": []
                            # "confs": [],
                        }
                    preds = logits.argmax(1)
                    probs = torch.softmax(logits, dim=1)
                    # dict_stats[key]["logits"].append(logits.cpu())
                    dict_stats[key]["probs"].append(probs.cpu())
                    dict_stats[key]["preds"].append(preds.cpu())
                    dict_stats[key]["correct"].append(preds.eq(y).float().cpu())

                    # dict_stats[key]["tcp"].append(
                    #     probs[range(len(torch.flatten(y))),
                    #           torch.flatten(y)].flatten().cpu()
                    # )
                    # dict_stats[key]["confs"].append(probs.max(dim=1)[0].cpu())
        for key0 in dict_stats:
            for key1 in dict_stats[key0]:
                try:
                    dict_stats[key0][key1] = torch.cat(dict_stats[key0][key1])
                except:
                    import pdb
                    pdb.set_trace()
        return dict_stats, batch_classes

    def get_dict_entropy(self, dict_stats, device):
        dict_results = {}

        def compute_entropy_predictions(x):
            #print(x)
            entropy = x * torch.log(x + 1e-10)  #bs * num_classes
            return -1. * entropy.sum() / entropy.size(0)

        for key, value in dict_stats.items():
            if "probs" not in value:
                print(value.keys())
                continue
            probs = value["probs"]
            entropy = compute_entropy_predictions(probs)
            dict_results["ent_" + key] = np.mean(
                entropy.float().cpu().numpy()
            )  # mean just to get rid of array

        return dict_results

    def get_dict_diversity(self, dict_stats, targets, device):
        dict_diversity = collections.defaultdict(list)
        # num_classifiers = int(min(len(self.classifiers), float(os.environ.get("MAXM", math.inf))))
        num_members = int(min(len(self.networks), float(os.environ.get("MAXM", math.inf))))
        regexes = [("waens", "wa_ens"), ("waprod", "wa_prod")]
        # regexes = [("netcla0", "net0_cla0"), ("netcla1", "net1_cla1"), ("netcla2", "net2_cla2")]
        # regexes = [
        #     (f"cla{i}{j}", f"cla{i}_cla{j}")
        #     for i in range(num_classifiers)
        #     for j in range(i + 1, num_classifiers)
        # ]
        regexes += [
            ("netm", f"net{i}_net{j}")
            for i in range(num_members)
            for j in range(i + 1, num_members)
        ]
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


from torch import nn, optim

class TrainableDiWA(DiWA):

    def _create_network(self):
        self.featurizer = None
        self.classifier = None
        self.featurizers = []
        self.networks = []

    def train(self, *args):
        algorithms.ERM.train(self, *args)
        for featurizer in self.featurizers:
            featurizer.train(*args)

    def to(self, device):
        algorithms.ERM.to(self, device)
        for featurizer in self.featurizers:
            featurizer.to(device)

    def init_train(self):
        self.num_aux = len(self.featurizers)
        self.classifier_task = copy.deepcopy(self.classifier)
        self.lambdas = torch.tensor([-5. for _ in range(self.num_aux)], requires_grad=True)
        self.optimizer_lambdas = optim.Adam(self.lambdas, lr=1e-4)
        self.optimizer_classifier = optim.Adam(self.classifier.parameters(), lr=1e-4)
        self.loss_fn = nn.CrossEntropyLoss()
        # torefine later

    def set_not_trainable(self):
        for net in [self.featurizer] + self.featurizers:
            for param in net.parameters():
                param.requires_grad = False

    def get_wa_weights(self):
        weights = {}
        list_gen_named_params = [featurizer.named_parameters() for featurizer in self.featurizers]
        for name_0, param_0 in self.featurizer.named_parameters():
            named_params = [next(gen_named_params) for gen_named_params in list_gen_named_params]
            new_data = param_0.data
            sum_lambdas = 1.
            for i in range(self.num_aux):
                name_i, param_i = named_params[i]
                assert name_0 == name_i
                exp_lambda_i = torch.exp(self.lambdas[i])
                new_data = new_data + exp_lambda_i * param_i
                sum_lambdas += exp_lambda_i
            weights[name_0] = new_data/sum_lambdas
        return weights

    def predict(self, x):
        dict_predictions = {}
        wa_weights = self.get_wa_weights()
        features_wa = torch.nn.utils.stateless.functional_call(
                self.featurizer, wa_weights, x)
        features_task = self.featurizer(x)

        # w for wa, t for task
        dict_predictions["ww"] = self.classifier(features_wa)
        dict_predictions["wt"] = self.classifier_task(features_wa)
        dict_predictions["tw"] = self.classifier(features_task)
        dict_predictions["tt"] = self.classifier_task(features_task)
        return dict_predictions

    def train_unlabeled(self, loader_train, device, data_evals, n_steps=100):
        self.to(device)
        self.eval()
        self.init_train()
        self.set_not_trainable()

        def train_step(x, y, optimizer="lambda"):
            optimizer.zero_grad()
            wa_weights = self.get_wa_weights()
            feats = torch.nn.utils.stateless.functional_call(self.featurizer, wa_weights, x)
            preds = self.classifier(feats)
            loss = self.loss_fn(preds, y)
            loss.backward(retain_graph=False)
            optimizer.step()
            return {"loss": loss.item()}

        # if os.environ.get('DEBUG', "0") != "0":
        #     pdb.set_trace()

        iter_loader_train = iter(loader_train)

        last_results_keys = []
        for step in range(0, n_steps):
            x, y = next(iter_loader_train)
            x = x.to(device)
            y = y.to(device)
            optimizer = self.optimizer_classifier if step % 2 else self.optimizer_lambdas
            l = train_step(x, y, optimizer)
            results = {'step': step}
            results.update(l)
            for i in enumerate(self.num_aux):
                results[f"lambda_{i}"] = self.lambdas[i].item()
            if step % 10 == 0:
                for name, loader in data_evals:
                    print(f"Inference at {name}")
                    _results_name = misc.accuracy(self, loader, device)
                    for key, value in _results_name.items():
                        new_key = name + "_" + key if name != "test" else key
                        results[new_key] = value
                    self.eval()
                results_keys = sorted(results.keys())
                if results_keys != last_results_keys:
                    misc.print_row(results_keys, colwidth=20)
                    last_results_keys = results_keys

                misc.print_row([results[key] for key in results_keys], colwidth=20)

        misc.print_row([results[key] for key in results_keys], colwidth=20)
# MODEL_SELECTION=train WHICHMODEL=stepbest INCLUDEVAL_UPTO=4 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.diwa --dataset OfficeHome --test_env 0  --output_dir /data/rame/experiments/domainbed/home0_ma_lp_0824 --trial_seed 0 --data_dir /data/rame/data/domainbed --checkpoints /data/rame/data/domainbed/inits/model_home0_ermll_saveall_si_0822.pkl 0 featurizer --what addfeats --topk 1 --weight_selection train
