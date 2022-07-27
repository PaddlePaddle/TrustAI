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
from trustai.interpretation import FeatureSimilarityModel

from utils import evaluate, preprocess_function

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",
                    default="./data",
                    type=str,
                    help="The dataset directory should include train.tsv, dev.tsv and test.tsv files.")
parser.add_argument("--train_file", type=str, default=None, help="Train data filename")
parser.add_argument("--dev_file", type=str, default=None, help="Dev data filename")

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
parser.add_argument("--rationale_num", type=int, default=3, help="Number of rationales per example. default:3")
parser.add_argument("--sparse_num", type=int, default=50, help="Number of sparse data. default:50")
parser.add_argument("--sparse_path", type=str, default="./data/sparse_data.tsv", help="Path to save sparse data.")
# parser.add_argument("--sparse_threshold",
# type=float,
# default="0.7",
# help="The threshold to select sparse data. default:0.7")

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


class LocalDataCollatorWithPadding(DataCollatorWithPadding):
    """
    Convert the  result of DataCollatorWithPadding from dict dictionary to a list
    """
    def __call__(self, features):
        batch = super().__call__(features)
        batch = list(batch.values())
        return batch


def get_sparse_data(analysis_result, sparse_num):
    """
    get sparse data
    """
    idx_scores = {}
    preds = []
    for i in range(len(analysis_result)):
        # pos_indexes
        scores = analysis_result[i].pos_scores
        idx_scores[i] = sum(scores) / len(scores)
        preds.append(analysis_result[i].pred_label)

    idx_socre_list = list(sorted(idx_scores.items(), key=lambda x: x[1]))[:sparse_num]
    ret_idxs, ret_scores = list(zip(*idx_socre_list))
    return ret_idxs, ret_scores, preds


def run():
    """
    Get dirty data
    """
    set_seed(args.seed)
    paddle.set_device(args.device)

    train_path = os.path.join(args.dataset_dir, args.train_file)
    dev_path = os.path.join(args.dataset_dir, args.dev_file)
    train_ds = load_dataset(read, data_path=train_path, lazy=False)
    dev_ds = load_dataset(read, data_path=dev_path, lazy=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    trans_func = functools.partial(preprocess_function,
                                   tokenizer=tokenizer,
                                   max_seq_length=args.max_seq_length,
                                   is_test=True)
    train_ds = train_ds.map(trans_func)
    dev_ds = dev_ds.map(trans_func)

    # batchify dataset
    collate_fn = LocalDataCollatorWithPadding(tokenizer)
    train_batch_sampler = BatchSampler(train_ds, batch_size=args.batch_size, shuffle=False)
    dev_batch_sampler = BatchSampler(dev_ds, batch_size=args.batch_size, shuffle=False)
    train_data_loader = DataLoader(dataset=train_ds, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
    dev_data_loader = DataLoader(dataset=dev_ds, batch_sampler=dev_batch_sampler, collate_fn=collate_fn)

    # define model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_classes=args.num_classes)
    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
    else:
        raise ValueError("The init_from_ckpt should exist.")

    # classifier_layer_name is the layer name of the last output layer
    feature_sim = FeatureSimilarityModel(model, train_data_loader, classifier_layer_name="classifier")
    # To do feature similarity analysis
    analysis_result = []
    for batch in dev_data_loader:
        analysis_result += feature_sim(batch, sample_num=args.rationale_num)

    # select sparse data
    sparse_indexs, sparse_scores, preds = get_sparse_data(analysis_result, args.sparse_num)

    # write data to disk
    is_true = []
    with open(args.sparse_path, 'w') as f:
        for idx in sparse_indexs:
            data = dev_ds.data[idx]
            f.write(data['text_a'] + '\t' + data['text_b'] + '\t' + str(data['label']) + '\n')
            is_true.append(1 if str(preds[idx]) == str(data['label']) else 0)
        print("accuracy in sparse data:", sum(is_true) / len(is_true))
        print("average score in sparse data:", sum(sparse_scores) / len(sparse_scores))

if __name__ == "__main__":
    run()
