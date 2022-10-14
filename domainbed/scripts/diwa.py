import argparse
import os
import json
import random
import numpy as np
import torch
import time
import torch.utils.data
from domainbed import datasets, algorithms_inference
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FastDataLoader
from domainbed.lib import misc


def _get_args():
    parser = argparse.ArgumentParser(description='Domain generalization')

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument('--test_env', type=int)
    parser.add_argument('--train_envs', nargs="+", type=int, default=[])
    parser.add_argument('--output_dir', nargs="+", type=str, default=None)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument(
        '--trial_seed',
        type=int,
        help='Trial number (used for seeding split_dataset and random_hparams).',
        default=0
    )
    parser.add_argument("--checkpoints", nargs='+', default=[])
    parser.add_argument('--weighting', type=str, default="norm")

    # select which checkpoints
    parser.add_argument('--weight_selection', type=str, default="uniform")  # or "restricted"
    parser.add_argument('--path_for_init', type=str, default=None)
    parser.add_argument('--topk', type=int, default=0)
    parser.add_argument('--what', nargs='+', default=[])

    inf_args = parser.parse_args()
    if len(inf_args.checkpoints) % 3 == 0:
        inf_args.checkpoints = [
            {
                "name": inf_args.checkpoints[3 * i],
                "weight": evalweight(inf_args.checkpoints[3 * i + 1]),
                "type": inf_args.checkpoints[3 * i + 2],
            } for i in range(len(inf_args.checkpoints) // 3)
        ]
    else:
        print("Your checkpoints should be of 'name weight type'")
        inf_args.checkpoints = [
            {
                "name": inf_args.checkpoints[2 * i],
                "weight": evalweight(inf_args.checkpoints[2 * i + 1]),
                "type": "network"
            } for i in range(len(inf_args.checkpoints) // 2)
        ]
    inf_args.checkpoints = [ckpt for ckpt in inf_args.checkpoints if float(ckpt["weight"]) != 0][::-1]
    misc.print_args(inf_args)
    return inf_args


def evalweight(string):
    if "/" in string:
        return evalweight(string.split("/")[0]) / evalweight(string.split("/")[1])
    elif string != "unifr":
        return eval(string)
    else:
        r = random.uniform(0, 1)
        return r/(1-r)


def create_splits(domain, inf_args, dataset, _filter, holdout_fraction):
    splits = []
    for env_i, env in enumerate(dataset):
        if domain.startswith("test") and env_i != inf_args.test_env:
            continue
        elif domain == "train" and env_i == inf_args.test_env:
            continue
        elif domain.startswith("env") and env_i != int(domain.split("_")[1]):
            continue

        if _filter == "full":
            splits.append(env)
        else:
            out_, in_ = misc.split_dataset(
                env, int(len(env) * holdout_fraction), misc.seed_hash(inf_args.trial_seed, env_i)
            )
            if _filter == "in":
                splits.append(in_)
            elif _filter == "out":
                splits.append(out_)
            elif _filter == "insmall":
                insmall_, inlarge_ = misc.split_dataset(
                    in_, int(len(in_) * 0.2), misc.seed_hash(inf_args.trial_seed, env_i)
                )
                splits.append(insmall_)
            elif _filter == "outsmall":
                outsmall_, outlarge_ = misc.split_dataset(
                    out_, int(len(out_) * 0.2), misc.seed_hash(inf_args.trial_seed, env_i)
                )
                splits.append(outsmall_)
            else:
                raise ValueError(_filter)

    return splits


def get_checkpoint_from_folder(output_folder):
    name = None
    if os.environ.get("WHICHMODEL", "best") in ['best', "stepbest"]:
        if "model_best.pkl" in os.listdir(output_folder):
            name = "model_best.pkl"
        elif "best" in os.listdir(output_folder):
            output_folder = os.path.join(output_folder, "best")
            name = "model_with_weights.pkl"
        elif "model.pkl" in os.listdir(output_folder):
            name = "model.pkl"
    elif os.environ.get("WHICHMODEL") in ['bestoracle', 'stepbestoracle']:
        name = "model_bestoracle.pkl"

    if name is None:
        if os.environ.get("WHICHMODEL", "last") in ['last', "steplast"]:
            name = "model_with_weights.pkl"
            if name not in os.listdir(output_folder):
                name = "model.pkl"
        elif os.environ.get("WHICHMODEL") not in ['best', "stepbest"]:
            name = "model_" + os.environ.get("WHICHMODEL") + ".pkl"

    if name in os.listdir(output_folder):
        return os.path.join(output_folder, name)
    return None


def get_dict_checkpoint_to_score(output_dir, inf_args, train_envs=None, device="cuda"):
    _output_folders = [os.path.join(output_dir, path) for path in os.listdir(output_dir)]
    output_folders = [
        output_folder for output_folder in _output_folders if os.path.isdir(output_folder) and
        (os.environ.get("DONEOPTIONAL", "0") != "0" or "done" in os.listdir(output_folder)) and
        get_checkpoint_from_folder(output_folder)
    ]
    if len(output_folders) == 0:
        raise ValueError(f"No done folders found for: {inf_args}")

    dict_checkpoint_to_score = {}
    for folder in output_folders:
        checkpoint = get_checkpoint_from_folder(folder)
        if device == "cpu":
            save_dict = torch.load(checkpoint, map_location=torch.device('cpu'))
        else:
            save_dict = torch.load(checkpoint)
        train_args = save_dict["args"]

        if train_args["dataset"] != inf_args.dataset:
            continue
        if misc.is_none(os.environ.get("INDOMAIN")):
            if inf_args.test_env not in train_args["test_envs"] + [-1]:
                continue
            if train_envs and any(train_env in train_args["test_envs"] for train_env in train_envs):
                continue
        else:
            if inf_args.test_env in train_args["test_envs"]:
                continue
            assert len(train_envs) == 0

        if train_args["trial_seed"] != inf_args.trial_seed and inf_args.trial_seed != -1:
            continue

        if "results" not in save_dict:
            score = -1
        else:
            score = misc.get_score(
                json.loads(save_dict["results"]), [inf_args.test_env],
                metric_key=os.environ.get("KEYACC", "out_acc"),
                model_selection=os.environ.get("MODEL_SELECTION", "train")
            )
        dict_checkpoint_to_score[checkpoint] = score

    if len(dict_checkpoint_to_score) == 0:
        raise ValueError(f"No folders found for: {inf_args}")
    return dict_checkpoint_to_score


def load_and_update_networks(wa_algorithm, good_checkpoints, dataset, action="mean", device="gpu"):
    model_hparams = {"nonlinear_classifier": False, "resnet18": False, "resnet_dropout": 0}

    for ckpt in good_checkpoints:
        checkpoint = ckpt["name"]
        checkpoint_weight = ckpt["weight"]
        checkpoint_type  = ckpt["type"]
        if device == "cpu":
            save_dict = torch.load(checkpoint, map_location=torch.device('cpu'))
        else:
            save_dict = torch.load(checkpoint)

        if "model_hparams" in save_dict:
            model_hparams = save_dict["model_hparams"]

        # load individual weights
        algorithm = algorithms_inference.ERM(
            dataset.input_shape, dataset.num_classes,
            len(dataset) - 1, model_hparams
        )
        algorithm.eval()
        try:
            if "model_dict" in save_dict:
                algorithm.load_state_dict(save_dict["model_dict"], strict=False)
            elif checkpoint_type in ["network", "networknotclassifier", "classifier", "featurizer"]:
                print(f"Load network {checkpoint} {checkpoint_weight} {checkpoint_type}")
                if "network_dict" in save_dict:
                    algorithm.network.load_state_dict(save_dict["network_dict"])
                else:
                    algorithm.network.load_state_dict(save_dict)
            else:
                assert checkpoint_type in ["featurizeronly"]
                print(f"Load featurizer {checkpoint} {checkpoint_weight} {checkpoint_type}")
                algorithm.featurizer.load_state_dict(save_dict)

        except Exception as e:
            print(f"Failed when trying to load {checkpoint} {checkpoint_weight} {checkpoint_type}")
            time.sleep(1)
            raise e

        if checkpoint_type in ["network", "networknotclassifier"]:
            if "mean" in action:
                wa_algorithm.update_mean_network(
                    algorithm.network, weight=checkpoint_weight
                )
            if "product" in action:
                wa_algorithm.update_product_network(
                    algorithm.network, weight=checkpoint_weight
                )
            if "ma" in action:
                wa_algorithm.update_mean_network_ma(algorithm.network_ma, weight=checkpoint_weight)
            if "netm" in action:
                wa_algorithm.add_network(algorithm.network)
            if "var" in action:
                wa_algorithm.update_var_network(algorithm.network)

        if checkpoint_type in ["network", "networknotclassifier", "featurizer", "featurizeronly"]:
            if "feats" in action:
                wa_algorithm.update_mean_featurizer(algorithm.featurizer, weight=checkpoint_weight)
            if "featsproduct" in action:
                wa_algorithm.update_product_featurizer(algorithm.featurizer, weight=checkpoint_weight)

        if checkpoint_type in ["network", "classifier"]:
            if "cla" in action:
                assert "feats" in action
                wa_algorithm.update_mean_classifier(algorithm.classifier, weight=checkpoint_weight)
            if "claproduct" in action:
                assert "featsproduct" in action
                wa_algorithm.update_product_classifier(algorithm.classifier, weight=checkpoint_weight)
            if "clas" in action:
                assert "feats" in action
                wa_algorithm.add_classifier(algorithm.classifier, weight=checkpoint_weight)

        del algorithm


def get_wa_results(good_checkpoints, dataset, inf_args, data_names, data_splits, device):
    wa_algorithm = algorithms_inference.DiWA(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - 1,
    )
    print("selected_checkpoints: ", good_checkpoints)
    load_and_update_networks(
        wa_algorithm, good_checkpoints, dataset, action=["mean"] + inf_args.what, device=device
    )
    if "var" in inf_args.what:
        load_and_update_networks(wa_algorithm, good_checkpoints, dataset, action=["var"], device=device)
        wa_algorithm.create_stochastic_networks()

    wa_algorithm.to(device)

    wa_algorithm.eval()
    if inf_args.path_for_init:
        wa_algorithm.save_path_for_future_init(inf_args.path_for_init)

    data_loaders = [
        FastDataLoader(dataset=split, batch_size=64, num_workers=dataset.N_WORKERS)
        for split in data_splits
    ]

    data_evals = zip(data_names, data_loaders)
    dict_results = {}

    for name, loader in data_evals:
        print(f"Inference at {name}")
        _results_name = misc.results_ensembling(wa_algorithm, loader, device)
        for key, value in _results_name.items():
            new_key = name + "_" + key if name != "test" else key
            dict_results[new_key] = value

    # some hacky queries to enrich dict_results
    dict_results["length"] = len(good_checkpoints)
    if "VARM" in os.environ:
        dict_results["varm"] = float(os.environ["VARM"])
    if "MAXM" in os.environ:
        dict_results["maxm"] = int(os.environ["MAXM"])
    if "WHICHMODEL" in os.environ:
        dict_results["step"] = str(os.environ["WHICHMODEL"])
        if dict_results["step"].startswith("step"):
            dict_results["step"] = dict_results["step"][4:]
        try:
            dict_results["step"] = int(dict_results["step"])
        except:
            pass
    if inf_args.checkpoints:
        dict_results["robust"] = "-".join([
            str(ckpt["weight"]) + "_" + str(ckpt["type"])
            for ckpt in inf_args.checkpoints])

    return dict_results


def weighting_checkpoint(checkpoint, weighting, dict_checkpoint_to_score, len_checkpoint):
    if weighting in [None, "uniform", "None"]:
        return 1.
    if weighting in "norm":
        return 1/len_checkpoint
    if misc.is_float(weighting):
        return float(weighting)
    if "/" in weighting:
        return eval(weighting)
    if weighting in ["linear"]:
        return dict_checkpoint_to_score[checkpoint]
    if weighting in ["quadratic"]:
        return dict_checkpoint_to_score[checkpoint]**2
    raise ValueError(weighting)


def print_results(dict_results):
    results_keys = sorted(list(dict_results.keys()))
    print("printres: ", {_key: dict_results[_key] for _key in results_keys})
    misc.print_row(results_keys, colwidth=20)
    misc.print_row([dict_results[key] for key in results_keys], colwidth=20)


def merge_checkpoints(inf_args, list_dict_checkpoint_to_score_i):

    dict_checkpoint_to_score = {}
    notsorted_checkpoints = []
    for dict_checkpoint_to_score_i in list_dict_checkpoint_to_score_i:
        sorted_checkpoints_i = sorted(
            dict_checkpoint_to_score_i.keys(),
            key=lambda x: dict_checkpoint_to_score_i[x],
            reverse=True
        )
        if inf_args.topk != 0:
            if inf_args.topk > 0:
                # select best according to metrics
                rand_nums = range(0, inf_args.topk)
            else:
                # select k randomly
                rand_nums = sorted(random.sample(range(len(sorted_checkpoints_i)), -inf_args.topk))

            sorted_checkpoints_i = [sorted_checkpoints_i[i] for i in rand_nums]
        for checkpoint in sorted_checkpoints_i:
            print("Found: ", checkpoint, " with score: ", dict_checkpoint_to_score_i[checkpoint])
        dict_checkpoint_to_score.update(dict_checkpoint_to_score_i)
        notsorted_checkpoints.extend(sorted_checkpoints_i)

    if inf_args.weight_selection != "restricted":
        return dict_checkpoint_to_score, notsorted_checkpoints

    sorted_checkpoints = sorted(
        notsorted_checkpoints, key=lambda x: dict_checkpoint_to_score[x], reverse=True
    )
    return dict_checkpoint_to_score, sorted_checkpoints

def create_data_splits(inf_args, dataset):
    # load data: test and optionally train_out for restricted weight selection
    data_splits, data_names = [], []

    if misc.is_not_none(os.environ.get("INDOMAIN")):
        dict_domain_to_filter = {"test": "out", "testin": "in"}
    elif inf_args.test_env != -1:
        dict_domain_to_filter = {"test": "full"}
    else:
        dict_domain_to_filter = {}

    # if os.environ.get("INCLUDE_TRAIN", "0") != "0":
    #     assert inf_args.trial_seed != -1
    #     dict_domain_to_filter["train"] = "out"

    if os.environ.get("INCLUDE_UPTO", "0") != "0":
        for env_i in range(0, int(os.environ.get("INCLUDE_UPTO", "0"))):
            dict_domain_to_filter["env_" + str(env_i) + "_out"] = "out"
            dict_domain_to_filter["env_" + str(env_i) + "_in"] = "in"
    if os.environ.get("INCLUDETSV_UPTO", "0") != "0":
        for env_i in range(0, int(os.environ.get("INCLUDETSV_UPTO", "0"))):
            dict_domain_to_filter["env_" + str(env_i) + "_out"] = "out"
            dict_domain_to_filter["env_" + str(env_i) + "_in"] = "insmall"
    if os.environ.get("INCLUDETRAIN_UPTO", "0") != "0":
        for env_i in range(0, int(os.environ.get("INCLUDETRAIN_UPTO", "0"))):
            dict_domain_to_filter["env_" + str(env_i) + "_in"] = "in"
    if os.environ.get("INCLUDEVAL_UPTO", "0") != "0":
        for env_i in range(0, int(os.environ.get("INCLUDEVAL_UPTO", "0"))):
            dict_domain_to_filter["env_" + str(env_i) + "_out"] = "out"

    for domain in dict_domain_to_filter:
        holdout_fraction = float(os.environ.get("HOLDOUT", 0.2))
        _data_splits = create_splits(
            domain,
            inf_args,
            dataset,
            dict_domain_to_filter[domain],
            holdout_fraction=holdout_fraction
        )
        if domain == "train":
            data_splits.append(misc.MergeDataset(_data_splits))
        else:
            data_splits.append(_data_splits[0])
        data_names.append(domain)
    return data_splits, data_names


def main():
    inf_args = _get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(inf_args.seed)
    np.random.seed(inf_args.seed)
    torch.manual_seed(inf_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Begin DiWA for: {inf_args} with device: {device}")

    if inf_args.dataset in vars(datasets):
        dataset_class = vars(datasets)[inf_args.dataset]
        dataset = dataset_class(
            inf_args.data_dir, [inf_args.test_env],
            hparams={"data_augmentation": os.environ.get("DATAAUG", "0") == "1"}
        )
    else:
        raise NotImplementedError

    # load individual folders and their corresponding scores on train_out
    list_dict_checkpoint_to_score_i = []
    if inf_args.output_dir[0] == "no":
        assert len(inf_args.output_dir) == 1
        assert len(inf_args.checkpoints) != 0
        print("Only from checkpoints with explicit name")

    elif not os.environ.get("PERD"):
        for output_dir in inf_args.output_dir:
            list_dict_checkpoint_to_score_i.append(
                get_dict_checkpoint_to_score(output_dir, inf_args, train_envs=inf_args.train_envs, device=device)
            )
    else:
        raise ValueError("PERD not implemented")
        # for output_dir in inf_args.output_dir[1:]:
        #     list_dict_checkpoint_to_score_i.append(
        #         get_dict_checkpoint_to_score(output_dir, inf_args, train_envs=inf_args.train_envs, device=device)
        #     )
        # list_i = [1, 2, 3] if os.environ.get("PERD") == "1" else [int(p) for p in os.environ.get("PERD").split(",")]
        # for i in list_i:
        #     list_dict_checkpoint_to_score_i.append(
        #         get_dict_checkpoint_to_score(inf_args.output_dir[0], inf_args, train_envs=[i], device=device)
        #     )

    dict_checkpoint_to_score, sorted_checkpoints = merge_checkpoints(
        inf_args, list_dict_checkpoint_to_score_i
    )
    data_splits, data_names = create_data_splits(inf_args, dataset)

    # compute score after weight averaging
    if inf_args.weight_selection == "restricted":
        # Restricted weight selection
        assert len(inf_args.checkpoints) == 0

        ## sort individual members by decreasing accuracy on train_out
        selected_indexes = []
        best_result = -float("inf")
        dict_best_results = {}
        ## incrementally add them to the WA
        for i in range(0, len(sorted_checkpoints)):
            selected_indexes.append(i)
            selected_checkpoints = [sorted_checkpoints[index] for index in selected_indexes]
            selected_checkpoints = [
                {
                    "name":
                    checkpoint,
                    "weight":
                    weighting_checkpoint(
                        checkpoint, inf_args.weighting,
                        dict_checkpoint_to_score,
                        len(selected_checkpoints)
                    ),
                    "type":
                    "network"
                } for checkpoint in selected_checkpoints
            ]

            ood_results = get_wa_results(
                selected_checkpoints, dataset, inf_args, data_names, data_splits, device
            )
            ood_results["i"] = i

            ## accept only if WA's accuracy is improved
            if ood_results["train_acc"] >= best_result:
                dict_best_results = ood_results
                ood_results["accept"] = 1
                best_result = ood_results["train_acc"]
                print(f"Accepting index {i}")
            else:
                ood_results["accept"] = 0
                selected_indexes.pop(-1)
                print(f"Skipping index {i}")
            print_results(ood_results)

            if inf_args.path_for_init:
                raise ValueError("Do not proceed when saving init")

        ## print final scores
        dict_best_results["final"] = 1
        print_results(dict_best_results)

    elif inf_args.weight_selection == "uniform":
        selected_checkpoints = [
                {
                    "name": checkpoint,
                    "weight": weighting_checkpoint(
                        checkpoint, inf_args.weighting, dict_checkpoint_to_score,
                        len(sorted_checkpoints)),
                    "type": "network"
                } for checkpoint in sorted_checkpoints
            ]
        if inf_args.checkpoints:
            # normalizer = len(selected_checkpoints)/20 if len(selected_checkpoints) else 1.
            print(f"Extending inf_args.checkpoints: {inf_args.checkpoints}")
            selected_checkpoints.extend(inf_args.checkpoints)

        dict_results = get_wa_results(
            selected_checkpoints, dataset, inf_args, data_names, data_splits, device
        )
        print_results(dict_results)

    else:
        raise ValueError(inf_args.weight_selection)


if __name__ == "__main__":
    main()
