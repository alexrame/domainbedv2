import argparse
import os
import json
import random
import numpy as np
import torch
import torch.utils.data
from domainbed import datasets, algorithms_inference
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FastDataLoader
from domainbed.lib import misc


def _get_args():
    parser = argparse.ArgumentParser(description='Domain generalization')

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--test_env', type=int)

    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument(
        '--trial_seed',
        type=int,
        help='Trial number (used for seeding split_dataset and random_hparams).'
    )
    parser.add_argument("--checkpoints", nargs=2, action="append", default=[])

    # select which checkpoints
    parser.add_argument('--weight_selection', type=str, default="uniform") # or "restricted"
    parser.add_argument('--path_for_init', type=str, default=None)
    parser.add_argument('--topk', type=int, default=0)
    parser.add_argument(
        '--what',
        nargs='+',
        default=[])

    inf_args = parser.parse_args()
    if inf_args.checkpoints:
        inf_args.checkpoints = [(str(key), float(val)) for (key, val) in inf_args.checkpoints]

    misc.print_args(inf_args)
    return inf_args


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
            else:
                raise ValueError(_filter)

    return splits


def get_best_model(output_folder):
    if os.environ.get("WHICHMODEL", "best") == 'last':
        name = "model.pkl"
    elif os.environ.get("WHICHMODEL", "best") == 'best':
        if "model_best.pkl" in os.listdir(output_folder):
            name = "model_best.pkl"
        else:
            name = "best/model_with_weights.pkl"
    else:
        name = "model_" + os.environ.get("WHICHMODEL") + ".pkl"

    if name in os.listdir(output_folder):
        return os.path.join(output_folder, name)
    return None

def get_dict_checkpoint_to_score(inf_args):
    _output_folders = [
        os.path.join(output_dir, path)
        for output_dir in inf_args.output_dir.split(",")
        for path in os.listdir(output_dir)
    ]
    output_folders = [
        output_folder for output_folder in _output_folders
        if os.path.isdir(output_folder)
        and (os.environ.get("DONEOPTIONAL") or "done" in os.listdir(output_folder))
        and get_best_model(output_folder)
    ]
    if len(output_folders) == 0:
        raise ValueError(f"No done folders found for: {inf_args}")

    dict_checkpoint_to_score = {}
    for folder in output_folders:
        model_path = get_best_model(folder)
        save_dict = torch.load(model_path)
        train_args = save_dict["args"]

        if train_args["dataset"] != inf_args.dataset:
            continue
        if misc.is_none(os.environ.get("INDOMAIN")):
            if inf_args.test_env not in train_args["test_envs"]:
                continue
        else:
            if inf_args.test_env in train_args["test_envs"]:
                continue

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
        dict_checkpoint_to_score[model_path] = score

    if len(dict_checkpoint_to_score) == 0:
        raise ValueError(f"No folders found for: {inf_args}")
    return dict_checkpoint_to_score


def load_and_update_networks(wa_algorithm, good_checkpoints, dataset, action="mean"):
    for checkpoint, checkpoint_weight in good_checkpoints.items():
        save_dict = torch.load(checkpoint)
        train_args = save_dict["args"]

        # load individual weights
        algorithm = algorithms_inference.ERM(
            dataset.input_shape, dataset.num_classes,
            len(dataset) - 1,
            save_dict["model_hparams"]
        )
        algorithm.load_state_dict(save_dict["model_dict"], strict=False)
        if "mean" in action:
            wa_algorithm.update_mean_network(algorithm.network, weight=checkpoint_weight)
        if "netm" in action:
            wa_algorithm.add_network(algorithm.network)
        if "var" in action:
            wa_algorithm.update_var_network(algorithm.network)
        del algorithm
    return train_args


def get_wa_results(
    good_checkpoints, dataset, inf_args, data_names, data_splits, device
):
    wa_algorithm = algorithms_inference.DiWA(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - 1,
    )
    train_args = load_and_update_networks(wa_algorithm, good_checkpoints, dataset, action=["mean"] + inf_args.what)
    if "var" in inf_args.what:
        _ = load_and_update_networks(wa_algorithm, good_checkpoints, dataset, action=["var"])
        wa_algorithm.create_stochastic_networks()

    wa_algorithm.to(device)
    wa_algorithm.eval()
    if inf_args.path_for_init:
        wa_algorithm.save_path_for_future_init(inf_args.path_for_init)

    random.seed(train_args["seed"])
    np.random.seed(train_args["seed"])
    torch.manual_seed(train_args["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_loaders = [
        FastDataLoader(
            dataset=split,
            batch_size=64,
            num_workers=dataset.N_WORKERS
        ) for split in data_splits
    ]

    data_evals = zip(data_names, data_loaders)
    dict_results = {}

    for name, loader in data_evals:
        print(f"Inference at {name}")
        _results_name = misc.results_ensembling(wa_algorithm, loader, device)
        for key, value in _results_name.items():
            new_key = name + "_" + key if name != "test" else key
            dict_results[new_key] = value

    dict_results["length"] = len(good_checkpoints)
    if "VARM" in os.environ:
        dict_results["varm"] = float(os.environ["VARM"])
    if "MAXM" in os.environ:
        dict_results["maxm"] = int(os.environ["MAXM"])
    return dict_results



def print_results(dict_results):
    results_keys = sorted(list(dict_results.keys()))
    misc.print_row(results_keys, colwidth=20)
    misc.print_row([dict_results[key] for key in results_keys], colwidth=20)


def main():
    inf_args = _get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    dict_checkpoint_to_score = get_dict_checkpoint_to_score(inf_args)
    sorted_checkpoints = sorted(dict_checkpoint_to_score.keys(), key=lambda x: dict_checkpoint_to_score[x], reverse=True)
    if inf_args.topk != 0:
        if inf_args.topk > 0:
            rand_nums = sorted(random.sample(range(len(sorted_checkpoints)), inf_args.topk))
        else:
            # select best according to metrics
            rand_nums = range(0, - inf_args.topk)
        sorted_checkpoints = [sorted_checkpoints[i] for i in rand_nums]
    for checkpoint in sorted_checkpoints:
        print("Found: ", checkpoint, " with score: ", dict_checkpoint_to_score[checkpoint])

    # load data: test and optionally train_out for restricted weight selection
    data_splits, data_names = [], []

    if misc.is_not_none(os.environ.get("INDOMAIN")):
        dict_domain_to_filter = {"test": "out", "testin": "in"}
    else:
        dict_domain_to_filter = {"test": "full"}

    if inf_args.weight_selection == "restricted" or misc.is_not_none(os.environ.get("INCLUDE_TRAIN")):
        assert inf_args.trial_seed != -1
        dict_domain_to_filter["train"] = "out"

    if os.environ.get("INCLUDE_UPTO", "0") != "0":
        for env_i in range(0, int(os.environ.get("INCLUDE_UPTO", "0"))):
            dict_domain_to_filter["env_" + str(env_i) + "_out"] = "out"
            dict_domain_to_filter["env_" + str(env_i) + "_in"] = "in"

    for domain in dict_domain_to_filter:
        holdout_fraction = float(os.environ.get("HOLDOUT", 0.2)) if domain.startswith("test") else 0.2
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
            selected_checkpoints = [(sorted_checkpoints[index], 1.) for index in selected_indexes]

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
        selected_checkpoints = [(checkpoint, 1.) for checkpoint in sorted_checkpoints]
        if inf_args.checkpoints:
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
