import os
import numpy as np
import re
import pandas as pd
import mlflow
import datetime
from domainbed.lib import misc


def set_mlflow_experiment(experiment_name):
    TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    assert mlflow.get_tracking_uri() == TRACKING_URI


def set_experiment_name(args):
    MLFLOWEXPES_VERSION = os.environ.get("MLFLOWEXPES_VERSION", "v0")
    test_env = args["test_envs"][0]
    return args["dataset"] + str(test_env) + MLFLOWEXPES_VERSION
    # cli:
    #   & mlflow experiments create -n ColoredMNISTte2v8
    # python:
    #   return None
    #   mlflow.create_experiment(
    #     cfg.MLFLOW.EXPERIMENT_NAME)


def get_run_name(args, hparams):
    name = "_".join([args[key] for key in [
        "dataset",
        "algorithm",
    ]])
    name += 'te' + "".join([str(te) for te in args["test_envs"]])

    keys = sorted(hparams.keys())
    keys = list(dict.fromkeys(keys))

    def params_to_str(param):
        if isinstance(param, (int, np.int, np.int32, np.int64)):
            param = str(int(param))
        elif param is None:
            return "none"
        else:
            try:
                param = re.sub(".0$", "", f"{param:.5}")
                param = re.sub("^0.", "e", param)
            except:
                param = ""
        return param.replace(".", "e")

    name += "_" + "_".join([key[:3].replace("_", "") + params_to_str(hparams[key]) for key in keys])
    name += "_" + datetime.datetime.now().strftime("%d%H%M%S")
    return name


def split_args_between_tags_and_params(args):
    dict_tags = {}
    params = {}
    for key, value in args.items():
        if isinstance(value, str):
            dict_tags[key] = value
        elif key in ["trial_seed", "seed", "holdout_fraction", "sweep_id"]:
            dict_tags[key] = value
        elif key in "test_envs":
            dict_tags[key] = "_".join([str(v) for v in value])
        elif key in [
            "hparams", "hp", "hparams_seed", "checkpoint_freq", "output_dir", "skip_model_save",
            "save_model_every_checkpoint"
        ]:
            pass
        elif key in ["steps", "uda_holdout_fraction"]:
            params[key] = value

    dict_tags["machine"] = misc.get_machine_name()
    return dict_tags, params


def clean_metrics(metrics, output_format="float"):
    new_dict = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            continue
            # v = v["string"]
        if isinstance(v, str):
            if v.endswith("%"):
                v = v[:-1]
        if output_format == "float":
            try:
                v = float(v)
            except:
                continue
        new_dict[k] = v
    return new_dict


def _get_tensorboard_paths(output_folder, topn=None):
    list_tb_files = sorted([f for f in os.listdir(output_folder) if "events.out.tfevents." in f])
    list_tb_files = [os.path.join(output_folder, str(tb_file)) for tb_file in list_tb_files][::-1]
    if not topn:
        return list_tb_files
    if topn == 1:
        return list_tb_files[0]
    return list_tb_files[:int(topn)]


def _get_logs_path(output_folder):
    logs_path = os.path.join(output_folder, "results.jsonl")
    return logs_path


def add_artifact_to_run(output_folder):
    print(f"Adding artifact from output_folder: {output_folder}")
    try:
        list_tensorboard_paths = _get_tensorboard_paths(output_folder, topn=None)
        for tensorboard_path in list_tensorboard_paths:
            mlflow.log_artifact(tensorboard_path)
    except:
        print("Could not because run was not finished ?", exc_info=True)

    try:
        logs_path = _get_logs_path(output_folder)
        mlflow.log_artifact(logs_path)
    except:
        print("Could not because runned on other machine", exc_info=True)



def main_mlflow(run_name, metrics, args, output_dir, hparams, move_artifacts=True):

    set_mlflow_experiment(experiment_name=set_experiment_name(args))
    dict_tags, new_params = split_args_between_tags_and_params(args)
    new_params.update({k: v for k, v in hparams.items()})
    metrics = clean_metrics(metrics)

    with mlflow.start_run(run_name=run_name) as run:
        print(f"Starting run: {run.info} for run_name: {run_name}")
        # Log our parameters into mlflow
        for key, value in new_params.items():
            mlflow.log_param(key, value)

        mlflow.set_tags(dict_tags)
        mlflow.log_metrics(metrics)

        # Upload the TensorBoard event logs as a run artifact

        if output_dir is not None and move_artifacts:
            add_artifact_to_run(output_folder=output_dir)
    print(f"Finishing run: {run.info} for run_name: {run_name}")
    return run
