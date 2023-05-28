import copy
from collections import OrderedDict
import torch
import glob
import os
import itertools
from peft import PeftModel
from transformers import LlamaForCausalLM
from peft.utils.save_and_load import get_peft_model_state_dict

import args_utils


class Predictor:
    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 1}

    def __init__(self, sentiment_pipes, tokenizer, output_max_length, device):
        self.sentiment_pipes = sentiment_pipes
        self.tokenizer = tokenizer
        self.output_max_length = output_max_length
        self.device = device

    def get_rewards(self, texts):
        raise ValueError()

    @staticmethod
    def transform_reward(reward):
        d_reward = []
        for rew in reward:
            d = {}
            assert len(rew) == 1
            for r in rew[0]:
                d[r["label"]] = r["score"]
            d_reward.append(d)
        return d_reward

    def average_rewards(self, rewards):
        avg_reward = None
        for reward in rewards:
            if avg_reward is None:
                avg_reward = copy.deepcopy(reward)
            else:
                for a_dict_reward, r_dict_reward in zip(avg_reward, reward):
                    for label in a_dict_reward:
                        a_dict_reward[label] = a_dict_reward[label] + r_dict_reward[label]
        assert avg_reward is not None
        for i in range(len(avg_reward)):
            a_dict_reward = avg_reward[i]
            for label in a_dict_reward:
                a_dict_reward[label] = a_dict_reward[label] / len(rewards)
            a_dict_reward["n"] = args_utils.Naming.get_name_model(
                self.sentiment_pipes[i].model.name_or_path
            )
        return avg_reward

    def get_prediction_rewards(self, model, query_tensors):

        texts = []
        # with torch.cuda.amp.autocast():
        for i in range(len(query_tensors)):
            query_tensor = torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(self.device)
            output = model.generate(
                input_ids=query_tensor,
                max_new_tokens=self.output_max_length,
                pad_token_id=self.tokenizer.eos_token_id
            ).squeeze()
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            texts.append(text)

        rewards = self.get_rewards(texts)
        avg_reward = self.average_rewards(rewards)
        avg_reward.append({"length": len(query_tensors)})
        return texts, rewards, avg_reward

    def predict(self, dict_models_to_merge, query_tensors, verbose=False):
        list_rewards = []
        for model_name, model in dict_models_to_merge.items():
            texts, rewards, avg_reward = self.get_prediction_rewards(model, query_tensors)
            print("=== For model:", model_name)
            for text, reward in zip(texts, rewards):
                print("=== text:", text.replace("\n", "[NEWLINE] "), reward)
                if not verbose:
                    break
            list_rewards.append(avg_reward)
        return list_rewards


class ResultsComputer:

    def __init__(self, predictor, base_model, query_tensors, verbose):
        self.predictor = predictor
        self.base_model = base_model
        self.query_tensors = query_tensors
        self.verbose = verbose

    def singlenolora(self):
        list_rewards_wa = self.predictor.predict(
            {"single": self.base_model}, self.query_tensors, verbose=self.verbose
        )
        return list_rewards_wa[0]

    def single(self, peft_name):
        print("Single")
        list_rewards_wa = self.create_and_call_wa(
            [peft_name], coefficients=[1.], name=peft_name.split("/")[-1]
        )
        return list_rewards_wa[0]

    def average(self, peft_names):
        print("Average")
        len_peft_names = len(peft_names)
        coefficients = [1 / len_peft_names for _ in range(len_peft_names)]
        list_rewards_wa = self.create_and_call_wa(
            peft_names, coefficients, name="_".join(name[0] for name in peft_names)
        )
        return list_rewards_wa[0]

    def combination(self, peft_names):
        print("combination")
        dict_coeff_to_reward = {}
        len_peft_names = len(peft_names)
        for r in [len_peft_names] + list(range(1, len_peft_names)):
            combinations = list(itertools.combinations(range(len_peft_names), r=r))
            coefficients = [1 / r for _ in range(r)]
            for combination in combinations:
                selected_peft_names = [peft_name for i, peft_name in enumerate(peft_names) if i in combination]
                combin_name = "_".join([name.split("/")[-2].split("-")[5] + name.split("/")[-1][5:] for name in selected_peft_names])
                print(combination, combin_name, selected_peft_names)
                list_rewards_wa = self.create_and_call_wa(
                    selected_peft_names, coefficients, name=combin_name
                )
                dict_coeff_to_reward[combin_name] = list_rewards_wa[0]

        return dict_coeff_to_reward

    def interpolation(self, peft_names, every):
        print("Interpolation")
        dict_coeff_to_reward = {}
        list_coeffs = [0, 1] + [x / 10 for x in range(1, 10, 1)]
        # list_coeffs = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        for i, coeff in enumerate(list_coeffs):
            do_coeff = True
            if every != 0:
                if every > 0 and i % every != 0 and coeff not in [0, 1]:
                    do_coeff = False
                if every < 0 and (i % (-every) == 0 or coeff in [0, 1]):
                    do_coeff = False
            if do_coeff:
                list_rewards_wa = self.create_and_call_wa(peft_names, [1 - coeff, coeff])
                dict_coeff_to_reward[coeff] = list_rewards_wa[0]

        return dict_coeff_to_reward

    def arithmetic(self, peft_names, every):
        print("Arithmetic")
        dict_coeff_to_reward = {}
        for i, coeff in enumerate([0, 1] + [x / 20 for x in range(1, 20, 1)]):
            if every != 0 and i % every != 0 and coeff not in [0, 1]:
                continue
            else:
                coefficients = [0.5 - coeff, coeff, 0.5]
                list_rewards_wa = self.create_and_call_wa(peft_names, coefficients)
                dict_coeff_to_reward[tuple(coefficients)] = list_rewards_wa[0]

        return dict_coeff_to_reward

    def create_and_call_wa(self, peft_names, coefficients, name=None):
        # 4.1 load wa
        wa = WeightAverager.build_wa(
            base_model=self.base_model, peft_names=peft_names, coefficients=coefficients
        )
        torch.cuda.empty_cache()

        # 4.2 predict with wa
        if name is None:
            coeff = 1 - coefficients[0]
            name = "wa coeff coefficient[1] " + str(coeff)
        list_rewards_wa = self.predictor.predict(
            {name: wa}, self.query_tensors, verbose=self.verbose
        )
        print("==", name, list_rewards_wa[0], "\n")

        # 4.3 del wa
        del wa
        torch.cuda.empty_cache()
        wa = None
        return list_rewards_wa


LOAD_ONLY_LORA = True


class WeightAverager:

    @staticmethod
    def average_weights(base_model, peft_names, coefficients):
        weights_averaged = OrderedDict()
        i = 0
        for peft_name, coefficient in zip(peft_names, coefficients):
            if coefficient == 0.:
                continue
            if peft_name is None:
                print("Skipping none peft_name")
                continue
            current_model = Loader.load_peft_model(base_model, peft_name)
            assert LOAD_ONLY_LORA
            current_weights = get_peft_model_state_dict(current_model, state_dict=None)
            for key in list(current_weights.keys()):
                if i == 0:
                    weights_averaged[key] = coefficient * current_weights[key]
                else:
                    weights_averaged[key] += coefficient * current_weights[key]
                del current_weights[key]
            del current_model
            torch.cuda.empty_cache()
            i += 1
        return weights_averaged

    @staticmethod
    def build_wa(base_model, peft_names, coefficients):
        weights_averaged = WeightAverager.average_weights(
            base_model=base_model, peft_names=peft_names, coefficients=coefficients
        )

        torch.cuda.empty_cache()
        wa = Loader.load_peft_model(base_model, peft_names[0])
        wa.load_state_dict(weights_averaged, strict=not LOAD_ONLY_LORA)
        return wa


class Loader:

    @staticmethod
    def load_base_model(base_model_name):
        base_model = LlamaForCausalLM.from_pretrained(
            base_model_name,
            load_in_8bit=args_utils.LOAD_IN_8BIT,
            device_map="auto",
            local_files_only=args_utils.LOCAL_FILES_ONLY
        )
        return base_model

    @staticmethod
    def load_peft_model(base_model, peft_name):
        peft_model = PeftModel.from_pretrained(
            base_model, peft_name, local_files_only=args_utils.LOCAL_FILES_ONLY
        )
        peft_model.eval()
        return peft_model


def get_dict_peft_names(folder):
    list_folder = sorted(glob.glob(folder))
    print("list_folder:", list_folder)
    dict_peft_names = {int(os.path.split(path)[-1].split("epoch")[1]): path for path in list_folder}
    if os.environ.get("MINEPOCH", "0") != "0":
        dict_peft_names = {
            key: value
            for key, value in dict_peft_names.items()
            if key > int(os.environ.get("MINEPOCH"))
        }
    if os.environ.get("MAXEPOCH", "0") != "0":
        dict_peft_names = {
            key: value
            for key, value in dict_peft_names.items()
            if key < int(os.environ.get("MAXEPOCH"))
        }
    dict_peft_names = OrderedDict(
        {key: dict_peft_names[key] for key in sorted(dict_peft_names.keys(), reverse=True)}
    )
    return dict_peft_names


def get_results_rewards(resultscomputer, peft_names, every):
    if len(peft_names) == 1:
        if peft_names[0] == "nolora":
            dict_coeff_to_reward = {0: resultscomputer.singlenolora()}
        elif "*" in peft_names[0]:
            dict_peft_names = get_dict_peft_names(peft_names[0])
            print(dict_peft_names)
            max_step = max(dict_peft_names.keys())
            dict_coeff_to_reward = {}
            for step, peft_name in dict_peft_names.items():
                dict_coeff_to_reward[step / max_step] = resultscomputer.single(peft_name)
                dict_coeff_to_reward[step / max_step].append({"step": step, "peft_name": peft_name})
        else:
            dict_coeff_to_reward = {0: resultscomputer.single(peft_names[0])}
    elif len(peft_names) == 2:
        dict_coeff_to_reward = resultscomputer.interpolation(peft_names, every)
    elif len(peft_names) == 3:
        dict_coeff_to_reward = resultscomputer.arithmetic(peft_names, every)
    else:
        dict_coeff_to_reward = resultscomputer.combination(peft_names=peft_names)
    # elif len(peft_names) > 3:
    #     peft_names = [name for name in peft_names if name not in ["no", "alexrame/no"]]
    #     print("Filtered peft_names", peft_names)
    #     dict_coeff_to_reward = resultscomputer.average(peft_names=peft_names)
    # else:
    #     raise ValueError()

    # 4. print results
    for coeff in sorted(dict_coeff_to_reward.keys()):
        print("d[", coeff, "] =", dict_coeff_to_reward[coeff])
