# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections

import json
import os

import tqdm

from domainbed.lib.query import Q

def load_list_records(list_path):
    records = []
    for path in list_path:
        for i, subdir in tqdm.tqdm(list(enumerate(os.listdir(path))),
                                ncols=80,
                                leave=False):
            results_path = os.path.join(path, subdir, "results.jsonl")
            if not os.path.exists(os.path.join(path, subdir, "done")) and os.environ.get("DONEOPTIONAL", "0") == "0":
                # print(f"{os.path.join(path, subdir)} without done")
                continue
                # pass
            try:
                with open(results_path, "r") as f:
                    for line in f:
                        records.append(json.loads(line[:-1]))
            except IOError:
                pass

    return Q(records)

def load_records(path):
    records = []
    for i, subdir in tqdm.tqdm(list(enumerate(os.listdir(path))),
                               ncols=80,
                               leave=False):
        results_path = os.path.join(path, subdir, "results.jsonl")
        try:
            with open(results_path, "r") as f:
                for line in f:
                    records.append(json.loads(line[:-1]))
        except IOError:
            pass

    return Q(records)

def get_grouped_records(records):
    """Group records by (trial_seed, dataset, algorithm, test_env). Because
    records can have multiple test envs, a given record may appear in more than
    one group."""
    result = collections.defaultdict(lambda: [])
    for r in records:
        r["args"]["hpstep"] = str(r["args"]["hparams_seed"]) + "_" + str(r["step"])
        for test_env in r["args"]["test_envs"]:
            output_dir_clean = "_".join(r["args"]["output_dir"].split("/")[-2].split("_")[0:])
            group = (r["args"]["trial_seed"],
                r["args"]["dataset"],
                r["args"]["algorithm"],
                test_env,
                output_dir_clean)
            result[group].append(r)
    # import pdb; pdb.set_trace()
    return Q(
        [
            {
                "trial_seed": t,
                "dataset": d,
                "algorithm": a,
                "test_env": e,
                "algorithmid": o,
                "records": Q(r)
            } for (t, d, a, e, o), r in result.items()
        ]
    )
