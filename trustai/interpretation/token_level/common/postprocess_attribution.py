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
"""postprocess attribution"""

import copy
import warnings


def get_word_offset(context, words):
    """get_word_offset"""
    pointer = 0  # point at the context
    offset_map = []
    for i in range(len(words)):
        seg_start_idx = context.find(words[i], pointer)
        seg_end_idx = seg_start_idx + len(words[i])
        offset_map.append([seg_start_idx, seg_end_idx])
        pointer = seg_end_idx
    return offset_map


def get_word_attributions(words, word_offset_map, subword_offset_map, attributions):
    """get_word_attributions"""
    result = []

    pointer1 = 0  # point at the context
    pointer2 = 0  # point at the sorted_token array

    for i in range(len(word_offset_map)):
        # merge spcial offset position in subword_offset_map
        seg_start_idx, seg_end_idx = word_offset_map[i]
        cur_set = []
        while pointer2 < len(subword_offset_map):
            while pointer2 < len(subword_offset_map) and subword_offset_map[pointer2][1] <= seg_start_idx:
                pointer2 += 1
            if subword_offset_map[pointer2][0] >= seg_end_idx:
                break
            cur_set.append(pointer2)
            pointer2 += 1
        result.append([cur_set, i, words[i]])
        pointer2 -= 1
        pointer1 = seg_end_idx
    word_attributions = merge_attributions(result, attributions)
    return word_attributions


def get_rationales_and_non_ratioanles(words, word_attributions, special_tokens=[], rationale_num=5):
    """"get_rationales_and_non_ratioanles"""
    assert len(words) == len(word_attributions)

    sorted_rationale_ids = list(sorted(range(len(words)), key=lambda i: word_attributions[i], reverse=True))
    rationale_tokens = []
    rationale_ids = []
    non_rationale_tokens = []
    non_rationale_ids = []
    for idx in sorted_rationale_ids:
        if words[idx] in special_tokens:
            continue
        if len(rationale_ids) < rationale_num:
            rationale_ids.append(idx)
            rationale_tokens.append(words[idx])
        else:
            non_rationale_ids.append(idx)
            non_rationale_tokens.append(words[idx])
    rationale_ids, rationale_tokens = zip(*list(sorted(zip(rationale_ids, rationale_tokens), key=lambda ele: ele[0])))
    if len(non_rationale_ids) == 0:
        non_rationale_ids = []
        non_rationale_tokens = []
    else:
        non_rationale_ids, non_rationale_tokens = zip(
            *list(sorted(zip(non_rationale_ids, non_rationale_tokens), key=lambda ele: ele[0])))
    return {
        "rationale_ids": rationale_ids,
        "rationale_tokens": rationale_tokens,
        "non_rationale_ids": non_rationale_ids,
        "non_rationale_tokens": non_rationale_tokens
    }


def merge_subword_special_idx(words, word_offset_map, subword_offset_map, special_tokens):
    """merge_subword_special_idx"""
    spcial_token_ids = []
    for idx, word in enumerate(words):
        if word in special_tokens:
            spcial_token_ids.append(idx)
    special_token_offset = []
    special_token_offset = [word_offset_map[idx] for idx in spcial_token_ids]
    subword_start_ids, subword_end_ids = list(zip(*subword_offset_map))
    merge_idx = []
    for token_start, token_end in special_token_offset:
        try:
            sub_start_id = subword_start_ids.index(token_start)
            sub_end_id = subword_end_ids.index(token_end)
            merge_idx.append([sub_start_id, sub_end_id])
        except:
            warnings.warn("Error offset mapping! Please check your offset map.")
    new_subword_offset_map = copy.deepcopy(subword_offset_map)
    for merge_start, merge_end in merge_idx[::-1]:
        spceial_toekn_start_id = new_subword_offset_map[merge_start][0]
        spceial_toekn_end_id = new_subword_offset_map[merge_end][1]
        del new_subword_offset_map[merge_start:merge_end + 1]
        new_subword_offset_map.insert(merge_start, [spceial_toekn_start_id, spceial_toekn_end_id])
    return new_subword_offset_map


def merge_attributions(match_list, attributions):
    """merge_attributions"""
    over_all = []
    miss = 0
    for i in match_list:
        over_all.extend(i[0])

    attribution_dic = {}
    for i in range(len(attributions)):
        split_time = over_all.count(i)
        if split_time:
            attribution_dic[i] = attributions[i] / split_time
        else:
            attribution_dic[i] = 0.0
    if miss != 0:
        print(miss)

    attributions = []
    for i in range(len(match_list)):
        cur_attribution = 0.0
        for j in match_list[i][0]:
            if j == -1:
                continue
            cur_attribution += attribution_dic[j]
        attributions.append(cur_attribution)
    return attributions