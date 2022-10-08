import argparse
import os
import json
import torch
import torch.utils.data
from domainbed.lib import misc
import shutil
# CUDA_VISIBLE_DEVICES=-1 python3 -m domainbed.scripts.get_oracle_weights --dataset OfficeHome --test_env 0 --output_dir /gpfsscratch/rech/edr/utr15kn/experiments/domainbed/home0_erm123wn_idn1erm0921r0_lp_0916 --trial_seed 0


def _get_args():
    parser = argparse.ArgumentParser(description='Domain generalization')

    parser.add_argument('--test_env', type=int)
    parser.add_argument('--trial_seed', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    inf_args = parser.parse_args()

    misc.print_args(inf_args)
    return inf_args


def get_checkpoint_from_folder(output_folder, return_oracle=False):
    name = "model_bestoracle.pkl"
    l = [os.path.join(output_folder, file) for file in os.listdir(output_folder)
         if file.startswith("model") and file.endswith(".pkl") and file != name]
    if return_oracle:
        return os.path.join(output_folder, name)
    else:
        return l


def main():
    inf_args = _get_args()
    output_dir = inf_args.output_dir
    _output_folders = [os.path.join(output_dir, path) for path in os.listdir(output_dir)]
    output_folders = [
        output_folder for output_folder in _output_folders if os.path.isdir(output_folder) and
        (os.environ.get("DONEOPTIONAL") or "done" in os.listdir(output_folder)) and
        get_checkpoint_from_folder(output_folder)
    ]
    if len(output_folders) == 0:
        raise ValueError(f"No done folders found for: {inf_args}")

    for folder in output_folders:
        checkpoints = get_checkpoint_from_folder(folder)
        best_score = 0
        best_checkpoint = None
        for checkpoint in checkpoints:
            save_dict = torch.load(checkpoint, map_location=torch.device('cpu'))
            train_args = save_dict["args"]

            if train_args["dataset"] != inf_args.dataset:
                continue
            if inf_args.test_env not in train_args["test_envs"] + [-1]:
                continue
            if train_args["trial_seed"] != inf_args.trial_seed and inf_args.trial_seed != -1:
                continue

            if "results" not in save_dict:
                score = -1
            else:
                score = misc.get_score(
                    json.loads(save_dict["results"]), [inf_args.test_env],
                    metric_key="out_acc",
                    model_selection="oracle"
                )
                if score > best_score:
                    best_score = score
                    best_checkpoint = checkpoint
        if best_checkpoint is None:
            print("Failure for checkpoints:", checkpoints)
        else:
            oracle = get_checkpoint_from_folder(folder, return_oracle=True)
            print("Copying:", best_checkpoint)
            shutil.copyfile(best_checkpoint, oracle)


if __name__ == "__main__":
    main()
