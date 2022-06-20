#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This script includes code to calculate MAP score"""

import json
import re
import os
import math
import numpy as np
import argparse


class Evaluator():
    """Evaluate prediction data.
    """

    def __init__(self):
        """Init functions

        Args:
            golden (dict): The golden dataset. Format should be dict of dict.
                e.g. {id: actual data under this id in format of dict}
                One example:
                {
                    5:{
                        "sent_id": 5,
                        "sent_text": "Today's weather is good!",
                        "sent_label": 0,
                        "sent_token": ["Today", "'s", "weather", "is", "good", "!"],
                        "rationale_tokens": [[["good"]]],
                        "rationale_ids": [[[4]]]
                        "sample_type": "ori",
                        "rel_ids": [105]
                    }
                }
                The field of "id", "sample_type" and "rel_ids" are necessary.
                "sample_type" should show whether this data is original or disturbed.
                Value of this field should be chosen in "ori" or "disturb".
                "rel_ids" should only appear when the data is original data. It points
                out the IDs of the related disturbed data. The data type should be a 
                list.
            pred (dict): The prediction dataset. Format should be dict of dict.
                e.g. {id: actual data under this id in format of dict}
                One example:
                {
                    5:{
                        id: 5,
                        pred_label: 0,
                        rationale: [[4, 2]],
                        rationale_tokens: [[good", "weather"]],
                        non_rationale: [[0, 3, 1, 5]],
                        non_rationale_tokens: [["Today", "is", "'s", "!"]],
                        rationale_pred_proba: [0.999, 0.001],
                        non_rationale_pred_proba: [0.342, 0.658],
                        pred_proba: [0.998, 0.002]
                    }
                }
                The field of "id" and "rationale_tokens" are necessary.
                "rationale_tokens" represents the tokens extracted as rationale.
                The format of this field should be list of list. For tasks with multiple
                sentences, for example, the textual similarity task, rationales of both
                sentences need to be covered. 
                One example for those cases can be:
                "rationale_tokens": [["is", "good", "today", "weather", "the"],["is", "bad", "today", "weather", "the"]]
        """
        #self.golden = golden
        #self.pred = pred
        #self.num_of_pair = self._count_disturb_data()

    def _count_disturb_data(self):
        """This function counts the number of data pairs in golden dataset
        """
        assert self.golden is not None

        num_of_pair = 0
        for idx in self.golden:
            if self.golden[idx]['sample_type'] == 'disturb':
                num_of_pair += 1
        return num_of_pair

    def _calc_map_by_bin(self, dis_attriRank_list, ori_attriRank_list):
        """This function calculates MAP using the equation in our paper,
        which follows equation one in consistency section of README

        Args:
            dis_attriRank_list (list): rationle tokens of current disturbed sentence.
            ori_attriRank_list (list): rationle tokens of current original sentence.

        Returns:
            [float]: MAP score of current sentence pair
        """

        total_precs = 0.0
        length_dis = len(dis_attriRank_list)
        if length_dis == 0:
            print("Disturbed sentence has a length of zero.")
            return 0

        for i in range(length_dis):
            hits = 0.0
            i += 1
            dis_t = dis_attriRank_list[:i]
            for idx, token in enumerate(dis_t):
                if token in ori_attriRank_list[:i]:
                    hits += 1
            hits = hits / i
            total_precs += hits
        return total_precs / length_dis

    def _calc_map(self, pred, sequence_idx, num_of_pair):
        """This function calculates MAP score for sentences of a certain token type in the whole dataset.

        Args:
            pred (dict): Please refer to the description of the "pred" argument in init function.
            sequence_idx (int): Sequence index represents the position of the sentence in one instance.
                If it is a task of single sentence e.g. sentiment analysis, the token type for all the
                sentences is 0. If it is a multiple sentence task e.g. textual similarity task, the 
                token type of the first sentence in the instance will be 0 and that of the second 
                sentence will be 1.
            num_of_pair (int): Number of pairs of sentence need to be evaluated in the golden dataset.

        Returns:
            [float]: Average MAP score of all sentence pairs of a certain token type
        """
        map_score = 0.0
        for idx in pred:
            if self.golden[idx]['sample_type'] == 'ori':
                ori_idx = idx
                ori = pred[ori_idx]
                # One original instance can be related to several disturbed instance
                for dis_idx in self.golden[ori_idx]['rel_ids']:
                    if dis_idx in pred:
                        dis = pred[dis_idx]
                        ori_attriRank_list = list(ori['rationale_tokens'][sequence_idx])
                        dis_attriRank_list = list(dis['rationale_tokens'][sequence_idx])

                        sum_precs = self._calc_map_by_bin(dis_attriRank_list, ori_attriRank_list)
                        map_score += sum_precs

        return map_score / num_of_pair

    def _check_length(self, pred=None):
        """Ensure the sentence number in one instance is the same

        Args:
            pred (list): Please refer to the description of "pred" argument in init function

        Returns:
            [int]: The sentence number in every instance
        """
        if pred is None:
            pred = self.pred
        assert pred is not None, "Prediction dataset is empty"

        length = []
        for idx in pred:
            length.append(len(pred[idx]['rationale_tokens']))

        assert len(set(length)) == 1, "the sentence number in one instance is not the same"

        return length[0]

    def cal_map(self, golden, pred):
        """This function calculates the MAP score of the pred dataset.
        MAP score represents consistency of the model that been interpreted.

        Args:
            pred (list): Please refer to the description of "pred" argument in init function

        Returns:
            [float]: Average MAP score
        """

        self.golden = golden
        self.pred = pred
        self.num_of_pair = self._count_disturb_data()

        assert self.pred is not None, "Prediction dataset is empty"
        assert self.golden is not None, "Golden dataset is empty"

        if self.num_of_pair == 0:
            print("The golden dataset does not have any disturbed data.")
            return 0

        num_of_sentence = self._check_length(pred)
        assert num_of_sentence != 0, "The number of sentence in each instance is zero."

        map_score = 0
        for i in range(num_of_sentence):
            t_map_tmp = self._calc_map(pred, i, self.num_of_pair)
            map_score += t_map_tmp
        map_score /= num_of_sentence
        return map_score

    def _f1(self, precision, recall):
        """Realizes the calculation of F1 score from precision and recall

        Args:
            precision (float): precision
            recall (float): recall

        Returns:
            float: F1 score
        """
        if precision == 0 or recall == 0:
            return 0
        return 2 * precision * recall / (precision + recall)

    def _calc_f1(self, golden_evid, pred_evid):
        """Realizes the calculation of F1 score of a single piece of data

        Args:
            golden_evid (list): golden rationale IDs
            pred_evid (list): predicted rationale IDs

        Returns:
            [float]: F1 score of current piece of data
        """
        tp = set(pred_evid) & set(golden_evid)
        prec = len(tp) / len(pred_evid) if len(pred_evid) else 0
        rec = len(tp) / len(golden_evid) if len(golden_evid) else 0
        f1 = self._f1(prec, rec)
        return f1

    def _combine(self, cur_max_f1, union_set, golden_evid, pred_evid):
        """potentially combine two sets

        Args:
            cur_max_f1  float:      current max F1 score
            union_set   set():      set that have already combined
            golden_evid list():     golden rationale
            pred_evid   list():     predicted rationale

        Returne:
            cur_max_f1: the max F1 we can observe at this point
            union_set: the combined set
        """
        if len(union_set & set(golden_evid)) < len(golden_evid) and self._calc_f1(golden_evid, pred_evid) > 0:
            new_union_set = union_set | set(golden_evid)
            new_f1 = self._calc_f1(new_union_set, pred_evid)
            if new_f1 > cur_max_f1:  # If the F1 score of the combined set is not larger than cur_max_f1, we do not update
                cur_max_f1 = new_f1
                union_set = new_union_set

        return cur_max_f1, union_set

    def list_elem_to_int(self, target_list):
        """Transfer all the string number elements in a list (or nested list) to integer

        Args:
            target_list (list): The target list that need to be transfered

        Returns:
            [list]: The transfered list
        """
        result = []
        if isinstance(target_list[0], list):
            for t in target_list:
                result.append(self.list_elem_to_int(t))
        else:
            result = [int(x) for x in target_list]
        return result

    def pick_max_golden_evid(self, pred=None):
        """Find a golden rationale set from golden dataset that have max F1 score with pred_raw

        Args:
            pred (list): Please refer to the description of "pred" argument in init function

        Returns:
            [list]: A dictionary that contains the golden rationale should be used for each instance
        """
        if pred is None:
            pred = self.pred
        assert pred is not None, "Prediction dataset is empty"
        assert self.golden is not None, "Golden dataset is empty"

        golden_dict = {}
        err_rationale = []

        for idx in pred.keys():
            if idx not in self.golden:
                continue
            num_of_sentence = self._check_length(pred)
            best_rationale = []
            golden_evids_total = self.list_elem_to_int(self.golden[idx]['rationale_ids'])
            for seq_idx in range(num_of_sentence):
                cur_best = []
                golden_rationale_ids = golden_evids_total[seq_idx]
                pred_evid = pred[idx]['rationale'][seq_idx]
                max_f1 = 0

                if len(golden_rationale_ids) == 1:
                    cur_best = golden_rationale_ids[0]
                else:
                    # find single golden_rationale_ids that maximize F1 score
                    for golden_rationale_id in golden_rationale_ids:
                        f1 = self._calc_f1(golden_rationale_id, pred_evid)
                        if f1 > max_f1:
                            max_f1 = f1
                            cur_best = golden_rationale_id

                    # find combined golden_rationale_ids sets that maximize F1 score
                    for startid in range(len(golden_rationale_ids) - 1):
                        union_set = set()
                        cur_max_f1 = 0
                        for id in range(startid, len(golden_rationale_ids)):
                            golden_rationale_id = golden_rationale_ids[id]
                            cur_max_f1, union_set = self._combine(cur_max_f1, union_set, golden_rationale_id, pred_evid)

                        if cur_max_f1 > max_f1:
                            max_f1 = cur_max_f1
                            cur_best = list(union_set)

                    if max_f1 == 0:
                        cur_best = []
                        err_rationale.append(idx)
                best_rationale.append(cur_best)
            golden_dict[idx] = best_rationale

        return golden_dict

    def cal_f1(self, golden, pred):
        """Calculate F1 score of the predicted dataset, which represent the plausibiliaty of the result

        Args:
            pred (list): Please refer to the description of "pred" argument in init function

        Return:
            [float]: The F1 score of the predicted dataset
        """
        self.golden = golden
        self.pred = pred
        self.num_of_pair = self._count_disturb_data()

        assert self.pred is not None, "Prediction dataset is empty"
        assert self.golden is not None, "Golden dataset is empty"

        f1 = 0.0
        golden_dict = self.pick_max_golden_evid(pred)
        golden_len = len(golden_dict)

        for idx in pred.keys():
            if idx not in golden_dict:
                continue
            num_of_sentence = self._check_length(pred)
            for seq_idx in range(num_of_sentence):
                golden_evid = golden_dict[idx][seq_idx]
                pred_evid = pred[idx]['rationale'][seq_idx]

                tp = set(golden_evid) & set(pred_evid)
                prec = len(tp) / len(pred_evid) if len(pred_evid) else 0
                rec = len(tp) / len(golden_evid) if len(golden_evid) else 0
                f1 += self._f1(prec, rec)

        macro_f1 = f1 / (golden_len * num_of_sentence) if golden_len else 0
        return macro_f1

    def cal_suf_com(self, golden, pred):
        """Calculate sufficency and comprehensiveness of the predicted dataset, which represent the faithfulness of the result

        Args:
            pred (list): Please refer to the description of "pred" argument in init function

        Return:
            [float, float]: The sufficency and comprehensiveness of the predicted dataset
        """
        self.golden = golden
        self.pred = pred
        self.num_of_pair = self._count_disturb_data()

        assert self.pred is not None, "Prediction dataset is empty"
        assert self.golden is not None, "Golden dataset is empty"

        suf_score = 0
        com_score = 0
        example_num = 0
        for idx in pred:
            senti_id = pred[idx]['id']
            label_id = int(pred[idx]['pred_label'])
            suf_score += pred[idx]['pred_proba'][label_id] - pred[idx]['rationale_pred_proba'][label_id]
            com_score += pred[idx]['pred_proba'][label_id] - pred[idx]['non_rationale_pred_proba'][label_id]
            example_num += 1

        suf_score_avg = suf_score / example_num if example_num else 0
        com_score_avg = com_score / example_num if example_num else 0
        return suf_score_avg, com_score_avg

    def calc_iou_f1(self, golden, pred):
        """Calculate IOU F1 score of the predicted dataset, which represent the faithfulness of the result

        Args:
            pred (list): Please refer to the description of "pred" argument in init function

        Return:
            [float]: The IOU F1 score of the predicted dataset
        """
        self.golden = golden
        self.pred = pred
        self.num_of_pair = self._count_disturb_data()

        assert self.pred is not None, "Prediction dataset is empty"
        assert self.golden is not None, "Golden dataset is empty"

        golden_dict = self.pick_max_golden_evid(pred)
        golden_len = len(golden_dict)
        match_num = 0.0

        for idx in pred.keys():
            if idx not in golden_dict:
                continue
            num_of_sentence = self._check_length(pred)
            for seq_idx in range(num_of_sentence):
                golden_evid = golden_dict[idx][seq_idx]
                pred_evid = pred[idx]['rationale'][seq_idx]

                inter_set = set(golden_evid) & set(pred_evid)
                union_set = set(golden_evid) | set(pred_evid)
                f1 = float(len(inter_set)) / len(union_set) if len(union_set) else 0
                if int(f1 * 10) >= 5:
                    match_num += 1

        macro_f1 = (float(match_num) / (golden_len * num_of_sentence)) if golden_len else 0
        return macro_f1
