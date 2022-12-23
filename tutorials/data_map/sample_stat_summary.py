#coding=utf-8

import sys
import json
import numpy as np

file_name = "output_data.json" 
output_file_names = ["correct_times", "correct_ratio", "avg_probs", "label_var", 
        "max_label_probs", "min_label_probs", "forgetting_times", "learnt_times", "first_forget", "first_learn", "pred_label", "pred_dist"]
num_samples = 100 

_input_path = "./outputs/" 
_output = open(_input_path + file_name + ".result", "w")

def list_concat(score_dict, input_file, input_path = "./data/", sample_size=1, pred_idx=-2, label_idx=-1, score_idx=-3, get_max_probs=False):
    """
    add info of each epoch (or given steps) into a dict of lists
    """
    _input = open(input_path + input_file, "r")
    for i, line in enumerate(_input):
        info = json.loads(line.strip().replace("\'", "\""))
        sid = int(info["id"])
        label = int(info["label"]) # [1:-1] to avoid "[]"
        if "noisy_label" in info: 
            s_label = int(info["noisy_label"]) 
        else:
            s_label = 0
        # score = float(info["probs"]) 
        label_probs = float(info["label_probs"]) # the score under GL 
        pred_correctness = info["correct"]
        if get_max_probs:
            all_probs = [eval(j) for j in info["probs"]]
            max_probs = np.max(all_probs)
        else:
            max_probs = 1.0

        score_info = [] # list of scores under different class
        for score in info["probs"]:    # the number of classes here
            score_float = float(score)
            score_info.append(score_float)

        if not score_dict["id"][sid]:
            score_dict["id"][sid] = sid
            score_dict["label"][sid] = label
            score_dict["s_label"][sid] = s_label
        score_dict["label_probs"][sid].append(label_probs)
        score_dict["max_probs"][sid].append(max_probs)
        score_dict["pred_info"][sid].append(pred_correctness)
        score_dict["pred_label"][sid].append(str(np.argmax(score_info)))

        # add forget info
        list_length = len(score_dict["pred_info"][sid])
        if list_length > 1:
            if score_dict["pred_info"][sid][list_length - 1] == score_dict["pred_info"][sid][list_length - 2]:
                score_dict["forget_info"][sid].append("None")
            elif score_dict["pred_info"][sid][list_length - 1] == "true":
                score_dict["forget_info"][sid].append("Learn")                
            else:
                score_dict["forget_info"][sid].append("Forget")
        else:
            score_dict["forget_info"][sid].append("None")
        #if sid == 1:
        #    print(score_dict["forget_info"][sid])

        # if i >= sample_size:
        #     break

    _input.close()

def check_correct_ratio(correct_lists):
    """
    ratio that a model predict classes correctly in different epochs
    """
    if len(correct_lists) == 0 or len(correct_lists[0]) == 0:
        return [0], [0]
    ratio_list = []
    pos_list = []
    for c_list in correct_lists: 
        pos_cnt = 0
        for info in c_list:
            if info == "true":
                pos_cnt += 1
        ratio_list.append(float(pos_cnt)/len(c_list) if len(c_list)!=0 else 0)
        pos_list.append(pos_cnt)
    return pos_list, ratio_list

def check_forget_time(forget_lists):
    if len(forget_lists) == 0 or len(forget_lists[0]) == 0:
        return [0], [0], 0, 0
    forgetting_list = []
    learnt_list = []
    first_forgetting_time = []
    first_learnt_time = []
    for f_list in forget_lists:
        forgetting_cnt = 0
        learnt_cnt = 0
        first_f_time = 0
        first_l_time = 0
        for i, info in enumerate(f_list):
            if info == "Forget":
                forgetting_cnt += 1
                if first_f_time == 0:
                    first_f_time = i
            elif info == "Learn":
                learnt_cnt += 1
                if first_l_time == 0:
                    first_l_time = i
        forgetting_list.append(forgetting_cnt)
        learnt_list.append(learnt_cnt)
        first_forgetting_time.append(first_f_time)
        first_learnt_time.append(first_l_time)

    return forgetting_list, learnt_list, first_forgetting_time, first_learnt_time

def check_pred_distribution(pred_lists):
    pred_list = []
    for scores in pred_lists:
        score_dist_dict = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0}
        for score in scores:
            score_dist_dict[score] += 1
        pred_list.append(score_dist_dict)
    return pred_list


info_dict = {"id": [[] for i in range(num_samples)], "label": [[] for i in range(num_samples)], 
        "s_label": [[] for i in range(num_samples)], "label_probs": [[] for i in range(num_samples)], 
        "max_probs": [[] for i in range(num_samples)], "pred_info": [[] for i in range(num_samples)],
        "forget_info": [[] for i in range(num_samples)], "pred_label": [[] for i in range(num_samples)]}
list_concat(info_dict, file_name, _input_path, sample_size=num_samples)

print(len(info_dict["label_probs"]), len(info_dict["label_probs"][0]))

info_dict["correct_times"], info_dict["correct_ratio"] = check_correct_ratio(info_dict["pred_info"])
info_dict["label_var"] = np.var(info_dict["label_probs"], axis=1)
info_dict["max_var"] = np.var(info_dict["max_probs"], axis=1)
info_dict["avg_probs"] = np.mean(info_dict["label_probs"], axis=1)
info_dict["max_label_probs"] = np.max(info_dict["label_probs"], axis=1)
info_dict["min_label_probs"] = np.min(info_dict["label_probs"], axis=1)
info_dict["forgetting_times"], info_dict["learnt_times"], info_dict["first_forget"], info_dict["first_learn"] = check_forget_time(info_dict["forget_info"])
info_dict["pred_dist"] = check_pred_distribution(info_dict["pred_label"])
output_file_names = ["id", "label", "s_label"] + output_file_names

_output.write("\t".join(output_file_names) + "\n")
for i in range(num_samples):
    info_list = []
    for name in output_file_names:
        info_list.append(str(info_dict[name][i]))
    _output.write("\t".join(info_list) + "\n")

_output.close()