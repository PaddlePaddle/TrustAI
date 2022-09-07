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
import re


def remove_None(str_list: list) -> list:
    return [t for t in str_list if t is not None]


def remove_blank_str(str_list: list) -> list:
    return [t for t in str_list if t != ""]


def process_single_sign(str_list: list) -> list:
    x = []
    for i in str_list:
        if i in ["|", "!", "。", "！", "？", "\\x0d", ";", "；", "?", "!"]:
            continue
        x.append(i)
    return x


def strip_str(s: str) -> str:
    return s.strip().strip("\n").strip("\\x0d")


def print_red(s: str, flag=True):
    if flag:
        print('\033[31m' + s + '\033[0m')


def split_sentence(s: str, remove_illegal_sign=True) -> list:
    # s.replace(" ","。")
    c_list = re.split('(。|！|？|\\x0d|;|；|\?|!|\||\.{2,}|[\u4E00-\u9FA5]\.{1,} *|\. )', s)
    c_list.append("")
    c_list = ["".join(i) for i in zip(c_list[0::2], c_list[1::2])]
    if remove_illegal_sign:
        c_list = remove_None(c_list)
        c_list = process_single_sign(c_list)
        c_list = remove_blank_str(c_list)
        return [strip_str(c) for c in c_list]
    else:
        return c_list


def batchify(dataset, batch_size):
    batch = []
    for i, data in enumerate(dataset):
        if (i % batch_size == 0):
            batch.append([data])
        else:
            batch[int(i / batch_size)].append(data)
    return batch


def padding_sentence(data, padding_value=0):
    max_x_len = 0
    max_y_len = 0
    for x in data:
        if len(x) > max_x_len:
            max_x_len = len(x)
        for y in x:
            if len(y) > max_y_len:
                max_y_len = len(y)
    for x in data:
        x += [[0] * max_y_len] * (max_x_len - len(x))
        for y in x:
            y += [0] * (max_y_len - len(y))
    return data


def padding_batch(data, padding_value=0):
    for x in data:
        max_y_len = 0
        for y in x:
            if len(y) > max_y_len:
                max_y_len = len(y)
        for y in x:
            y += [0] * (max_y_len - len(y))
    return data
