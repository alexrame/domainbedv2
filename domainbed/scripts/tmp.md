31c31
<     parser.add_argument('--weight_selection', type=str, default="uniform") # or "restricted"
---
>     parser.add_argument('--weight_selection', type=str, default="uniform")  # or "restricted"
34,37c34
<     parser.add_argument(
<         '--what',
<         nargs='+',
<         default=[])
---
>     parser.add_argument('--what', nargs='+', default=[])
90a88
> 
92,95c90
<     _output_folders = [
<         os.path.join(output_dir, path)
<         for path in os.listdir(output_dir)
<     ]
---
>     _output_folders = [os.path.join(output_dir, path) for path in os.listdir(output_dir)]
97,100c92,94
<         output_folder for output_folder in _output_folders
<         if os.path.isdir(output_folder)
<         and (os.environ.get("DONEOPTIONAL") or "done" in os.listdir(output_folder))
<         and get_checkpoint_from_folder(output_folder)
---
>         output_folder for output_folder in _output_folders if os.path.isdir(output_folder) and
>         (os.environ.get("DONEOPTIONAL") or "done" in os.listdir(output_folder)) and
>         get_checkpoint_from_folder(output_folder)
116c110,112
<             if train_envs and any(train_env in train_args["test_envs"] for train_env in train_envs):
---
>             if train_envs and any(
>                 train_env in train_args["test_envs"] for train_env in train_envs
>             ):
153,154c149
<             len(dataset) - 1,
<             model_hparams
---
>             len(dataset) - 1, model_hparams
171,173c166
< def get_wa_results(
<     good_checkpoints, dataset, inf_args, data_names, data_splits, device
< ):
---
> def get_wa_results(good_checkpoints, dataset, inf_args, data_names, data_splits, device):
179c172,174
<     train_args = load_and_update_networks(wa_algorithm, good_checkpoints, dataset, action=["mean"] + inf_args.what)
---
>     train_args = load_and_update_networks(
>         wa_algorithm, good_checkpoints, dataset, action=["mean"] + inf_args.what
>     )
196,200c191,192
<         FastDataLoader(
<             dataset=split,
<             batch_size=64,
<             num_workers=dataset.N_WORKERS
<         ) for split in data_splits
---
>         FastDataLoader(dataset=split, batch_size=64, num_workers=dataset.N_WORKERS)
>         for split in data_splits
222c214
<         dict_results["robust"] = float(inf_args.checkpoints[0][-1])/20
---
>         dict_results["robust"] = float(inf_args.checkpoints[0][-1]) / 20
227d218
< 
234c225
<         return dict_checkpoint_to_score[checkpoint] ** 2
---
>         return dict_checkpoint_to_score[checkpoint]**2
262,298c253,273
<     sorted_checkpoints = []
<     if not os.environ.get("PERD"):
<         for i, output_dir in enumerate(inf_args.output_dir):
<             dict_checkpoint_to_score_i = get_dict_checkpoint_to_score(
<                 output_dir, inf_args, train_envs=inf_args.train_envs
<             )
<             sorted_checkpoints_i = sorted(dict_checkpoint_to_score_i.keys(), key=lambda x: dict_checkpoint_to_score_i[x], reverse=True)
<             if inf_args.topk != 0:
<                 if inf_args.topk > 0:
<                     # select best according to metrics
<                     rand_nums = range(0, inf_args.topk)
<                 else:
<                     # select k randomly
<                     rand_nums = sorted(random.sample(range(len(sorted_checkpoints_i)), -inf_args.topk))
< 
<                 sorted_checkpoints_i = [sorted_checkpoints_i[i] for i in rand_nums]
<             for checkpoint in sorted_checkpoints_i:
<                 print("Found: ", checkpoint, " with score: ", dict_checkpoint_to_score_i[checkpoint])
<             dict_checkpoint_to_score.update(dict_checkpoint_to_score_i)
<             sorted_checkpoints.extend(sorted_checkpoints_i)
<     else:
<         for i in [1, 2, 3]:
<             dict_checkpoint_to_score_i = get_dict_checkpoint_to_score(inf_args.output_dir[0], inf_args, train_envs=[i])
<             sorted_checkpoints_i = sorted(dict_checkpoint_to_score_i.keys(), key=lambda x: dict_checkpoint_to_score_i[x], reverse=True)
<             if inf_args.topk != 0:
<                 if inf_args.topk > 0:
<                     # select best according to metrics
<                     rand_nums = range(0, inf_args.topk)
<                 else:
<                     # select k randomly
<                     rand_nums = sorted(random.sample(range(len(sorted_checkpoints_i)), -inf_args.topk))
< 
<                 sorted_checkpoints_i = [sorted_checkpoints_i[i] for i in rand_nums]
<             for checkpoint in sorted_checkpoints_i:
<                 print("Found: ", checkpoint, " with score: ", dict_checkpoint_to_score_i[checkpoint])
<             dict_checkpoint_to_score.update(dict_checkpoint_to_score_i)
<             sorted_checkpoints.extend(sorted_checkpoints_i)
---
> 
> 
>     for i in [1, 2, 3]:
>         dict_checkpoint_to_score_i = get_dict_checkpoint_to_score(inf_args.output_dir[0], inf_args, train_envs=[i])
>         sorted_checkpoints = sorted(
>             dict_checkpoint_to_score_i.keys(),
>             key=lambda x: dict_checkpoint_to_score_i[x],
>             reverse=True
>         )
>         if inf_args.topk != 0:
>             if inf_args.topk > 0:
>                 # select best according to metrics
>                 rand_nums = range(0, inf_args.topk)
>             else:
>                 # select k randomly
>                 rand_nums = sorted(random.sample(range(len(sorted_checkpoints)), -inf_args.topk))
> 
>             sorted_checkpoints = [sorted_checkpoints[i] for i in rand_nums]
>         for checkpoint in sorted_checkpoints:
>             print("Found: ", checkpoint, " with score: ", dict_checkpoint_to_score_i[checkpoint])
>         dict_checkpoint_to_score.update(dict_checkpoint_to_score_i)
308c283,285
<     if inf_args.weight_selection == "restricted" or misc.is_not_none(os.environ.get("INCLUDE_TRAIN")):
---
>     if inf_args.weight_selection == "restricted" or misc.is_not_none(
>         os.environ.get("INCLUDE_TRAIN")
>     ):
318c295,296
<         holdout_fraction = float(os.environ.get("HOLDOUT", 0.2)) if domain.startswith("test") else 0.2
---
>         holdout_fraction = float(os.environ.get("HOLDOUT", 0.2)
>                                 ) if domain.startswith("test") else 0.2
344,347c322
<             selected_checkpoints = [
<                 sorted_checkpoints[index]
<                 for index in selected_indexes
<             ]
---
>             selected_checkpoints = [sorted_checkpoints[index] for index in selected_indexes]
388,389c363,364
<                 (str(key), float(val) * len(selected_checkpoints)/20)
<                 for (key, val) in inf_args.checkpoints if float(val) != 0
---
>                 (str(key), float(val) * len(selected_checkpoints) / 20)
>                 for (key, val) in inf_args.checkpoints
403,404d377
< 
< 
