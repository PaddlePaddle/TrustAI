#coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
import argparse

input_path = "./outputs/"
input_file = "output_data.json.result"
input_file = "train_lcqmc_orig_238k_10ep.json.result.rpm_grdt.sparsity.low_bias"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--attr1",
    default='avg_probs',
    type=str,
    help="The attribute used for plotting the y axis of the data map.")
parser.add_argument(
    "--attr2",
    default='label_var',
    type=str,
    help="The attribute used for plotting the x axis of the data map.")
parser.add_argument(
    "--criterion",
    default='',
    type=str,
    help="The criterion used for selecting data points for plotting, used together with threshold.")
parser.add_argument(
    "--threshold",
    default=0,
    type=float,
    help="The threshold used for selecting data points for plotting, used together with criterion.")
parser.add_argument(
    "--use_f_times",
    default=-1,
    type=int,
    help="The forgotten time threshold for data map. -1 indicates this attribute is not used in plotting")
parser.add_argument(
    "--use_l_times",
    default=-1,
    type=int,
    help="The learnt time threshold for data map. -1 indicates this attribute is not used in plotting")
args = parser.parse_args()

dirty_label = False # use s_label to indicate dirty labeled samples, i.e., s_label == 2. s_label == 1 refers to hard samples. 
use_f_times = args.use_f_times
use_l_times = args.use_l_times
attr1 = args.attr1
attr2 = args.attr2
criterion = args.criterion
threshold = args.threshold

confidence_list = []
variance_list = []

def build_dict(header_list, init_dict={}, standard_dict={}, error_dict={}):
    for feature in header_list:
        init_dict[feature] = [[] for i in range(2)]
    specical_dict_f = {"forgotten_times-+": [[] for i in range(2)], "forgotten_times-": [[] for i in range(2)], "forgotten_times=0": [[] for i in range(2)], 
                    "forgotten_times=1": [[] for i in range(2)], "forgotten_times=2": [[] for i in range(2)], "forgotten_times>=3": [[] for i in range(2)]}
    specical_dict_l = {"learnt_times+-": [[] for i in range(2)], "learnt_times-": [[] for i in range(2)], "learnt_times=0": [[] for i in range(2)], 
                    "learnt_times=1": [[] for i in range(2)], "learnt_times=2": [[] for i in range(2)], "learnt_times>=3": [[] for i in range(2)]}
                    
    standard_dict = {"correct_ratio_0.0": [[] for i in range(2)], "correct_ratio_0.2": [[] for i in range(2)], "correct_ratio_0.4": [[] for i in range(2)], \
        "correct_ratio_0.6": [[] for i in range(2)], "correct_ratio_0.8": [[] for i in range(2)], "correct_ratio_1.0": [[] for i in range(2)]}
    error_dict = {"errors": [[] for i in range(2)], "corrects": [[] for i in range(2)]}
    return init_dict, specical_dict_f, specical_dict_l, standard_dict, error_dict


standard_marker_dict = {"correct_ratio_0.0": ["r", "X", 3], "correct_ratio_0.2": ["m", "v", 3], "correct_ratio_0.4": ["orange", "^", 3], 
                    "correct_ratio_0.6": ["g", "<", 3], "correct_ratio_0.8": ["c", ">", 3], "correct_ratio_1.0": ["b", "D", 3]}

special_marker_dict = {"forgotten_times-+": ["black", "X", 3], "forgotten_times-": ["c", "v", 3], "forgotten_times=0": ["b", ">", 3],  
                        "forgotten_times=1": ["r", "<", 3], "forgotten_times=2": ["g", "D", 3], "forgotten_times>=3": ["orange", "^", 3], 
                        "learnt_times+-": ["black", "X", 3], "learnt_times-": ["c", "v", 3], "learnt_times=0": ["b", ">", 3], 
                        "learnt_times=1": ["r", "<", 3], "learnt_times=2": ["g", "D", 3], "learnt_times>=3": ["orange", "^", 3]}

error_marker_dict = {"correct_ratio_0.0": ["deeppink", "X", 3], "correct_ratio_0.2": ["darkred", "v", 3], "correct_ratio_0.4": ["darkorange", "^", 3], 
                    "correct_ratio_0.6": ["darkgreen", "<", 3], "correct_ratio_0.8": ["deepskyblue", ">", 3], "correct_ratio_1.0": ["darkblue", "D", 3]}

def fill_in_standard_dict(standard_dict, cri_cnt, feature1, feature2, correct_ratio):
    if correct_ratio < 0.2:
        cri_cnt[1] += 1
        standard_dict["correct_ratio_0.0"][0].append([feature1])
        standard_dict["correct_ratio_0.0"][1].append([feature2])
    elif correct_ratio < 0.4:
        cri_cnt[2] += 1
        standard_dict["correct_ratio_0.2"][0].append([feature1])
        standard_dict["correct_ratio_0.2"][1].append([feature2])
    elif correct_ratio < 0.6:
        cri_cnt[3] += 1
        standard_dict["correct_ratio_0.4"][0].append([feature1])
        standard_dict["correct_ratio_0.4"][1].append([feature2])
    elif correct_ratio < 0.8:
        cri_cnt[4] += 1
        standard_dict["correct_ratio_0.6"][0].append([feature1])
        standard_dict["correct_ratio_0.6"][1].append([feature2])
    elif correct_ratio < 1.0:
        cri_cnt[5] += 1
        standard_dict["correct_ratio_0.8"][0].append([feature1])
        standard_dict["correct_ratio_0.8"][1].append([feature2])
    else:
        cri_cnt[6] += 1
        standard_dict["correct_ratio_1.0"][0].append([feature1])
        standard_dict["correct_ratio_1.0"][1].append([feature2])
    return standard_dict, cri_cnt

def fill_in_specical_dict_f(specical_dict_f, cri_cnt, feature1, feature2, forgotten_times, correct_ratio):
    if forgotten_times == 0:
        if correct_ratio == 0:
            specical_dict_f["forgotten_times-"][0].append([feature1])
            specical_dict_f["forgotten_times-"][1].append([feature2])
            cri_cnt[1] += 1
        elif correct_ratio < 1:
            specical_dict_f["forgotten_times-+"][0].append([feature1])
            specical_dict_f["forgotten_times-+"][1].append([feature2])
            cri_cnt[2] += 1
        else:
            specical_dict_f["forgotten_times=0"][0].append([feature1])
            specical_dict_f["forgotten_times=0"][1].append([feature2])
            cri_cnt[3] += 1
    elif forgotten_times == 1:
        specical_dict_f["forgotten_times=1"][0].append([feature1])
        specical_dict_f["forgotten_times=1"][1].append([feature2])
        cri_cnt[4] += 1
    elif forgotten_times == 2:
        specical_dict_f["forgotten_times=2"][0].append([feature1])
        specical_dict_f["forgotten_times=2"][1].append([feature2])
        cri_cnt[5] += 1
    else:
        specical_dict_f["forgotten_times>=3"][0].append([feature1])
        specical_dict_f["forgotten_times>=3"][1].append([feature2])
        cri_cnt[6] += 1
    return specical_dict_f, cri_cnt

def fill_in_specical_dict_l(specical_dict_l, cri_cnt, feature1, feature2, learnt_times, correct_ratio):
    if learnt_times == 0:
        if correct_ratio == 0:
            specical_dict_l["learnt_times-"][0].append([feature1])
            specical_dict_l["learnt_times-"][1].append([feature2])
            cri_cnt[1] += 1
        elif correct_ratio < 1:
            specical_dict_l["learnt_times+-"][0].append([feature1])
            specical_dict_l["learnt_times+-"][1].append([feature2])
            cri_cnt[2] += 1
        else:
            specical_dict_l["learnt_times=0"][0].append([feature1])
            specical_dict_l["learnt_times=0"][1].append([feature2])
            cri_cnt[3] += 1
    elif learnt_times == 1:
        specical_dict_l["learnt_times=1"][0].append([feature1])
        specical_dict_l["learnt_times=1"][1].append([feature2])
        cri_cnt[4] += 1
    elif learnt_times == 2:
        specical_dict_l["learnt_times=2"][0].append([feature1])
        specical_dict_l["learnt_times=2"][1].append([feature2])
        cri_cnt[5] += 1
    else:
        specical_dict_l["learnt_times>=3"][0].append([feature1])
        specical_dict_l["learnt_times>=3"][1].append([feature2])
        cri_cnt[6] += 1
    return specical_dict_l, cri_cnt

def run_plot(in_path, in_file, dim1, dim2, criterion=False, threshold=0, dirty_label=False, use_f_times=-1, use_l_times=-1):
    criteria_list = ["correct_times", "correct_ratio", "avg_probs", "label_var", "max_label_probs", "min_label_probs", 
                    "forgetting_times", "learnt_times", "first_forget"]
    if criterion and criterion not in criteria_list:
        print("The criterion is not in the signal list!")
        return
    elif dim1 not in criteria_list:
        print("The dim1 is not in the signal list!")
        return
    elif dim2 not in criteria_list:
        print("The dim2 is not in the signal list!")
        return
    if use_f_times != -1 and use_l_times != -1:
        print("Only one of forgotten_times and learnt_times can be used!")
        return
    if use_f_times != -1:
        forgotten_times_threshold = use_f_times # related to use_f_times 3
    if use_l_times != -1:
        learnt_times_threshold = use_l_times # related to use_l_times 3
    if criterion:
        out_file = criterion + str(threshold) + "_" + in_file.split(".")[0] + "." + dim1 + "-" + dim2
    else:
        out_file = in_file.split(".")[0] + "." + dim1 + "-" + dim2
    _input = open(in_path + in_file, "r")    
    index_dict = {}
    cri_cnt = [0, 0, 0, 0, 0, 0, 0]
    time_cnt = [0, 0, 0, 0, 0, 0, 0]
    for idx, line in enumerate(_input):
        line = line.strip().split("\t")
        if idx == 0:             
            if "id" == line[0]:
                for item in line:
                    index_dict[item] = line.index(item)
                header_dict, specical_dict_f, specical_dict_l, standard_dict, error_dict = build_dict(line)
                continue
            else:
                print("no id in header!")
                break
        
        forgotten_times = int(line[index_dict["forgetting_times"]])
        learnt_times = int(line[index_dict["learnt_times"]])
        correct_times = int(line[index_dict["correct_times"]])
        correct_ratio = float(line[index_dict["correct_ratio"]])
        feature1 = float(line[index_dict[dim1]])
        feature2 = float(line[index_dict[dim2]])
        if criterion:
            cri_feature = float(line[index_dict[criterion]])
        if dirty_label and "s_label" in index_dict: 
            error_label = int(line[index_dict["s_label"]])    
            if error_label == 2:
                error_dict["errors"][0].append([feature1])
                error_dict["errors"][1].append([feature2])
            else:
                error_dict["corrects"][0].append([feature1])
                error_dict["corrects"][1].append([feature2])
            continue

        if use_f_times != -1:            
            if forgotten_times >= forgotten_times_threshold:
                time_cnt[0] += 1
                specical_dict_f, time_cnt = fill_in_specical_dict_f(specical_dict_f, 
                    time_cnt, feature1, feature2, forgotten_times, correct_ratio)
                
        elif use_l_times != -1:
            if learnt_times >= learnt_times_threshold:
                time_cnt[0] += 1
                specical_dict_l, time_cnt = fill_in_specical_dict_l(specical_dict_l, 
                    time_cnt, feature1, feature2, learnt_times, correct_ratio)
                
        else:
            if criterion and cri_feature < threshold:
                continue
            cri_cnt[0] += 1
            standard_dict, cri_cnt = fill_in_standard_dict(standard_dict, cri_cnt, feature1, feature2, correct_ratio)
    _input.close()
    special_marker_list = [["*", 3], ["*", 4], ["*", 5], ["o", 3], ["o", 4], ["o", 5]]
    if use_f_times != -1:
        print("sample number under forgotten_times is: ", time_cnt)
    elif use_l_times != -1:
        print("sample number under learnt_times is: ", time_cnt)
    else:
        print("sample number under criterion is: ", cri_cnt)
    
    plt.figure(dpi=600) 
    if dirty_label:
        plt.scatter(error_dict["errors"][1], error_dict["errors"][0], marker = "X", c = "r")
        plt.scatter(error_dict["corrects"][1], error_dict["corrects"][0], marker = "D", c = "b")
    else:
        if use_l_times != -1:
            for key, value in specical_dict_l.items():
                if key == "learnt_times+-":
                    plt.scatter(value[1], value[0], marker = special_marker_dict[key][1], alpha = 1.0,
                            c = special_marker_dict[key][0], s = special_marker_dict[key][2], label = key)
                else:
                    plt.scatter(value[1], value[0], marker = special_marker_dict[key][1], alpha = 0.3,
                            c = special_marker_dict[key][0], s = special_marker_dict[key][2], label = key)
        elif use_f_times != -1:
            for key, value in specical_dict_f.items():
                if key == "forgotten_times-+":
                    plt.scatter(value[1], value[0], marker = special_marker_dict[key][1], alpha = 1.0,
                            c = special_marker_dict[key][0], s = special_marker_dict[key][2], label = key)
                else:
                    plt.scatter(value[1], value[0], marker = special_marker_dict[key][1], alpha = 0.3,
                            c = special_marker_dict[key][0], s = special_marker_dict[key][2], label = key)
        else:
            for key, value in standard_dict.items():
                plt.scatter(value[1], value[0], marker = standard_marker_dict[key][1], alpha = 0.3,
                            c = standard_marker_dict[key][0], s = standard_marker_dict[key][2], label = key)

    standard_dict.clear()
    plt.legend()
    plt.ylabel(dim1)
    plt.xlabel(dim2)
    if criterion:
        plt.title(dim1 + " - " + dim2 + " plot greater than " + criterion + " " + str(threshold))
    elif use_f_times != -1:
        plt.title(dim1 + " - " + dim2 + " plot forgotten_times greater than " + str(forgotten_times_threshold))
    elif use_l_times != -1:
        plt.title(dim1 + " - " + dim2 + " plot learnt_times greater than " + str(learnt_times_threshold))
    else:
        plt.title(dim1 + " - " + dim2 + " plot")
    if use_f_times != -1:
        plt.savefig('./outputs/ftimes' + str(forgotten_times_threshold) + '.' + out_file + '.png')
    elif use_l_times != -1:
        plt.savefig('./outputs/ltimes' + str(learnt_times_threshold) + '.' + out_file + '.png')
    elif dirty_label:
        plt.savefig('./outputs/dirty.' + out_file + '.png')
    else:
        plt.savefig('./outputs/' + out_file + '.png')
    plt.clf()

if __name__ == "__main__":

    run_plot(in_path=input_path, in_file=input_file, dim1=attr1, dim2=attr2, 
        criterion=criterion, threshold=threshold, use_f_times=use_f_times, use_l_times=use_l_times)
    run_plot(in_path=input_path, in_file=input_file, dim1=attr1, dim2=attr2, criterion="forgetting_times", threshold=1)
    run_plot(in_path=input_path, in_file=input_file, dim1=attr1, dim2=attr2, use_f_times=0)
    run_plot(in_path=input_path, in_file=input_file, dim1=attr1, dim2=attr2, use_l_times=0)
