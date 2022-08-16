# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import json
import functools
import random
import time
import os
import argparse

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle.metric import Accuracy
from paddle.io import DataLoader, BatchSampler, DistributedBatchSampler
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer, LinearDecayWithWarmup
from paddlenlp.utils.log import logger
from trustai.interpretation import RepresenterPointModel

from utils import evaluate, preprocess_function

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",
                    default="./data",
                    type=str,
                    help="The dataset directory should include train.tsv, dev.tsv and test.tsv files.")
parser.add_argument("--train_file", type=str, default=None, help="train data filename")
parser.add_argument("--max_seq_length",
                    default=128,
                    type=int,
                    help="The maximum total input sequence length after tokenization. "
                    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument('--model_name',
                    default="ernie-3.0-base-zh",
                    help="Select model to train, defaults to ernie-3.0-base-zh.")
parser.add_argument('--device',
                    choices=['cpu', 'gpu', 'xpu', 'npu'],
                    default="gpu",
                    help="Select which device to train model, defaults to gpu.")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--init_from_ckpt", type=str, help="The path of checkpoint to be loaded.")
parser.add_argument("--seed", type=int, default=3, help="random seed for initialization")
parser.add_argument('--num_classes', type=int, default=2, help='Number of classification.')
parser.add_argument("--dirty_num", type=int, default=500, help="Number of dirty data. default:50")
parser.add_argument("--dirty_path", type=str, default="./data/dirty_train.tsv", help="Path to save dirty data.")
parser.add_argument("--rest_path", type=str, default="", help="The path of rest data.")
parser.add_argument("--dirty_threshold", type=float, default="0", help="The threshold to select dirty data.")

args = parser.parse_args()


def set_seed(seed):
    """
    Sets random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def read(data_path):
    """Reads data"""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            text_a, text_b, label = line.strip().split('\t')
            yield {"text_a": text_a, "text_b": text_b, "label": int(label)}


def get_dirty_data(weight_matrix, dirty_num, threshold=0):
    """
    Get index of dirty data from train data
    """
    scores = []
    for idx in range(weight_matrix.shape[0]):
        weight_sum = 0
        count = 0
        for weight in weight_matrix[idx].numpy():
            if weight > threshold:
                count += 1
                weight_sum += weight
        scores.append((count, weight_sum))
    sorted_scores = sorted(scores)[::-1]
    sorted_idxs = sorted(range(len(scores)), key=lambda idx: scores[idx])[::-1]

    ret_scores = sorted_scores[:dirty_num]
    ret_idxs = sorted_idxs[:dirty_num]

    return ret_idxs, ret_scores


class LocalDataCollatorWithPadding(DataCollatorWithPadding):
    """
    Convert the result of DataCollatorWithPadding from dict dictionary to a list
    """

    def __call__(self, features):
        batch = super().__call__(features)
        batch = list(batch.values())
        return batch


def run():
    """
    Get dirty data
    """
    set_seed(args.seed)
    paddle.set_device(args.device)

    train_path = os.path.join(args.dataset_dir, args.train_file)
    train_ds = load_dataset(read, data_path=train_path, lazy=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    trans_func = functools.partial(preprocess_function,
                                   tokenizer=tokenizer,
                                   max_seq_length=args.max_seq_length,
                                   is_test=True)
    train_ds = train_ds.map(trans_func)

    # batchify dataset
    collate_fn = LocalDataCollatorWithPadding(tokenizer)
    train_batch_sampler = BatchSampler(train_ds, batch_size=args.batch_size, shuffle=False)
    train_data_loader = DataLoader(dataset=train_ds, batch_sampler=train_batch_sampler, collate_fn=collate_fn)

    # define model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_classes=args.num_classes)
    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
    else:
        raise ValueError("The init_from_ckpt should exist.")

    #classifier_layer_name is the layer name of the last output layer
    rep_point = RepresenterPointModel(model, train_data_loader, classifier_layer_name="classifier")
    weight_matrix = rep_point.weight_matrix
    # get dirty data
    dirty_indexs, dirty_scores = get_dirty_data(weight_matrix, args.dirty_num, args.dirty_threshold)
    with open(args.dirty_path, 'w') as f:
        for idx, score in zip(dirty_indexs, dirty_scores):
            f.write(train_ds.data[idx]['text_a'] + '\t' + train_ds.data[idx]['text_b'] + '\t' +
                    str(train_ds.data[idx]['label']) + '\t' + str(score[1]) + '\n')

    with open(args.rest_path, 'w') as f:
        for idx in range(len(train_ds)):
            if idx in dirty_indexs:
                continue
            f.write(train_ds.data[idx]['text_a'] + '\t' + train_ds.data[idx]['text_b'] + '\t' +
                    str(train_ds.data[idx]['label']) + '\n')


if __name__ == "__main__":
    run()
