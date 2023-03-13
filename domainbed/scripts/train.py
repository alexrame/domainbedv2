# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc, experiments_handler
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="default+name")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', type=str, default="0")
    ## DiWA ##
    parser.add_argument('--what_is_trainable', type=str, default="0")
    parser.add_argument('--path_for_init', type=str, default="")
    parser.add_argument('--path_for_save', type=str, default="")
    args = parser.parse_args()


    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))

    if args.hparams:
        hparams.update(json.loads(args.hparams))

    if args.output_dir == "default+name":
        run_name = experiments_handler.get_run_name(args.__dict__, hparams)
        if "DATA" in os.environ:
            args.output_dir = os.path.join(
                os.environ["DATA"], f"experiments/domainbed/singleruns/{args.dataset}", run_name
            )
        else:
            args.output_dir = os.path.join(f"logs/singleruns/{args.dataset}", run_name)
    # if os.environ.get("FROMSWEEP", "1") != "0" and os.path.exists(os.path.join(args.output_dir, 'out.txt')) and args.dataset != "TerraIncognita":
    #     time.sleep(1)
    #     print(f"Output {args.output_dir} directory already exists, exiting")
    #     sys.exit(0)

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    writer = SummaryWriter(log_dir=args.output_dir)

    print('HParams:')
    hparams["hparams_seed"] = args.hparams_seed
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discarded at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            raise ValueError("Class balanced not supported for domainbed")
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    # eval_loaders = [FastDataLoader(
    #     dataset=env,
    #     batch_size=64,
    #     num_workers=dataset.N_WORKERS)
    #     for env, _ in (out_splits)]
    # eval_weights = [None for _, weights in (out_splits)]
    # # eval_loader_names = ['env{}_in'.format(i) for i in range(len(in_splits))]
    # eval_loader_names = ['env{}_out'.format(i) for i in range(len(out_splits))]
    # # eval_loader_names += ['env{}_uda'.format(i) for i in range(len(uda_splits))]

    algorithm = algorithms.get_algorithm_class(args.algorithm)(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(args.test_envs),
        hparams,
        what_is_trainable=args.what_is_trainable,
        path_for_init=args.path_for_init
    )

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    n_steps = args.steps if args.steps is not None else dataset.N_STEPS
    if args.save_model_every_checkpoint!= "0" and args.save_model_every_checkpoint.isdigit():
        checkpoint_freq = args.checkpoint_freq or int(args.save_model_every_checkpoint)
    else:
        checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def save_checkpoint(filename, results=None, light=False):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
        }
        if not light:
            save_dict["model_dict"] = algorithm.state_dict()
        else:
            save_dict["network_dict"] = algorithm.get_network_state_dict()
            if hasattr(algorithm, "network_ma"):
                save_dict["network_ma_dict"] = algorithm.network_ma.state_dict()

        ## DiWA ##
        if results is not None:
            save_dict["results"] = results
        save_path = os.path.join(args.output_dir, filename)
        torch.save(save_dict, save_path)

    best_score = 0
    dict_metric_to_best_score = {"out_acc": 0, "out_acc_ma": 0}
    last_results_keys = None
    results = {}

    for step in range(0, n_steps + 1):
        step_start_time = time.time()
        if step > 0:
            minibatches_device = [(x.to(device), y.to(device))
                for x,y in next(train_minibatches_iterator)]
            if args.task == "domain_adaptation":
                uda_device = [x.to(device)
                    for x,_ in next(uda_minibatches_iterator)]
            else:
                uda_device = None

            step_vals = algorithm.update(minibatches_device, uda_device)
            checkpoint_vals['step_time'].append(time.time() - step_start_time)

            for key, val in step_vals.items():
                checkpoint_vals[key].append(val)

        do_inference_at_this_step = (step % checkpoint_freq == 0) or step == n_steps or (
            step < checkpoint_freq and step % int(os.environ.get("START_CHKPT_FREQ", checkpoint_freq)) == 0
        )

        if do_inference_at_this_step:
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)
                writer.add_scalar("Metrics/" + key, results[key], step)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                if "_out" in name or "env" + str(args.test_envs[0]) in name or True:
                    _results_name = misc.accuracy(algorithm, loader, weights, device)
                    for key, value in _results_name.items():
                        results[name + '_' + key] = value
                        writer.add_scalar(name + '_' + key, value, step)

            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)
            # results["holdout_fraction"] = args.holdout_fraction

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=20)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=20)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            if hasattr(algorithm, "lambdas"):
                results["hparams"]["featurizers_lambdas_step"] = " ".join(
                    [
                        "{:.4f}".format(float(_lambda.detach().float().cpu().numpy()))
                        for _lambda in algorithm.lambdas
                    ])

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True, default=misc.np_encoder) + "\n")

            for metric in dict_metric_to_best_score.keys():
                current_score = misc.get_score(results, args.test_envs, metric_key=metric, model_selection="train")
                if current_score > dict_metric_to_best_score[metric]:
                    dict_metric_to_best_score[metric] = current_score
                    if metric != "out_acc":
                        path = "model_best" + metric.split("_")[-1] + ".pkl"
                    else:
                        path = "model_best.pkl"
                    print(f"Saving new best train for: {metric} at step: {step} at path: {path}")
                    save_checkpoint(
                        path,
                        results=json.dumps(results, sort_keys=True, default=misc.np_encoder),
                    )

            checkpoint_vals = collections.defaultdict(lambda: [])

            save_ckpt = (args.save_model_every_checkpoint == "2" and step in [200, 1000, 2000, 4000, 4500])
            save_ckpt |= str(step) == args.save_model_every_checkpoint and step >= 2
            save_ckpt |= args.save_model_every_checkpoint == "1000" and step % 1000 == 0
            save_ckpt |= args.save_model_every_checkpoint == "100" and step % 100 == 0
            save_ckpt |= args.save_model_every_checkpoint == "500" and step % 500 == 0
            save_ckpt |= args.save_model_every_checkpoint == "all" and step in [200, 400, 600, 800, 1000, 1200, 1400, 1600, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 4900]
            if save_ckpt:
                save_checkpoint(
                    f'model_step{step}.pkl', results=json.dumps(results, sort_keys=True, default=misc.np_encoder),
                    light=True)

    save_checkpoint(
        'model.pkl',
        results=json.dumps(results, sort_keys=True, default=misc.np_encoder),
    )

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')

    ## DiWA ##
    if args.path_for_save:
        assert misc.is_not_none(args.what_is_trainable) or n_steps == -1
        print("Save for future: ", args.path_for_save)
        algorithm.save_path_for_future_init(args.path_for_save)
    # else:
    #     if os.environ.get("MLFLOWEXPES_VERSION", "v0") == "nomlflow":
    #         pass
    #     else:
    #         experiments_handler.main_mlflow(
    #             experiments_handler.get_run_name(args.__dict__, hparams=hparams),
    #             results,
    #             args=args.__dict__,
    #             output_dir=args.output_dir,
    #             hparams=hparams,
    #         )
