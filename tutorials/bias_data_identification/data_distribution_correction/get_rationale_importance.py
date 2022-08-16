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
import collections
import random
import time
import os
import argparse

import numpy as np
import paddle
from paddle.io import DataLoader, BatchSampler
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.utils.log import logger
from trustai.interpretation import get_word_offset
from trustai.interpretation.token_level.common import attention_predict_fn_on_paddlenlp
from trustai.interpretation.token_level import AttentionInterpreter

from tqdm import tqdm
import jieba

from utils import evaluate, preprocess_function

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",
                    default="./data",
                    type=str,
                    help="The dataset directory should include train.tsv, dev.tsv and test.tsv files.")
parser.add_argument("--input_file", type=str, default=None, help="input data filename")
parser.add_argument("--max_seq_length",
                    default=512,
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
parser.add_argument('--ig_steps', type=int, default=16, help='Number of steps for Integrated Gradients. default: 16')
parser.add_argument("--rationale_path",
                    type=str,
                    default="./data/rationale_importance.tsv",
                    help="Path to save rationale importance data.")

args = parser.parse_args()


def set_seed(seed):
    """
    Sets random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def read(data_path, max_seq_length):
    """Reads data"""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            label, text = line.strip().split('\t')
            yield {"text_a": text[:max_seq_length - 2], "label": int(label)}


class LocalDataCollatorWithPadding(DataCollatorWithPadding):
    """
    Convert the  result of DataCollatorWithPadding from dict dictionary to a list
    """

    def __call__(self, features):
        batch = super().__call__(features)
        batch = list(batch.values())
        return batch


def run():
    """
    Get rationale importance
    """
    set_seed(args.seed)
    paddle.set_device(args.device)

    # init lexical analyzer of chinese

    input_path = os.path.join(args.dataset_dir, args.input_file)
    input_ds = load_dataset(read, data_path=input_path, max_seq_length=args.max_seq_length, lazy=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    trans_func = functools.partial(preprocess_function,
                                   tokenizer=tokenizer,
                                   max_seq_length=args.max_seq_length,
                                   is_test=True)
    input_ds = input_ds.map(trans_func)

    # batchify dataset
    collate_fn = LocalDataCollatorWithPadding(tokenizer)
    input_batch_sampler = BatchSampler(input_ds, batch_size=args.batch_size, shuffle=False)
    input_data_loader = DataLoader(dataset=input_ds, batch_sampler=input_batch_sampler, collate_fn=collate_fn)

    # define model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_classes=args.num_classes)
    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
    else:
        raise ValueError("The init_from_ckpt should exist.")

    # get word_offset_maps and subword_offset_maps for align result
    contexts = []
    batch_words = []
    for example in input_ds.data:
        example['text'] = example['text_a']
        contexts.append("[CLS]" + " " + example['text_a'] + " " + "[SEP]")
        batch_words.append(["[CLS]"] + list(jieba.cut(example['text_a'])) + ["[SEP]"])
    word_offset_maps = []
    subword_offset_maps = []
    for i in range(len(contexts)):
        word_offset_maps.append(get_word_offset(contexts[i], batch_words[i]))
        subword_offset_maps.append(tokenizer.get_offset_mapping(contexts[i]))

    # get interpret result by intgrad
    print("The Interpreter method will take some minutes, please be patient.")
    att = AttentionInterpreter(model, predict_fn=attention_predict_fn_on_paddlenlp)

    analysis_result = []
    for batch in tqdm(input_data_loader):
        analysis_result += att(batch)
    align_res = att.alignment(analysis_result,
                              contexts,
                              batch_words,
                              word_offset_maps,
                              subword_offset_maps,
                              special_tokens=["[CLS]", '[SEP]'],
                              rationale_num=1)

    # sort rationale and return rationale and frequency pair
    rationale_dict = collections.defaultdict(int)
    for i in range(len(align_res)):
        for token in align_res[i].rationale_tokens:
            rationale_dict[token] += 1
    pair = sorted(list(rationale_dict.items()), key=lambda x: x[1], reverse=True)

    # write pair to disk
    with open(args.rationale_path, 'w') as f:
        for key, value in pair:
            if value <= 2:
                continue
            f.write(key + '\t' + str(value) + '\n')


if __name__ == "__main__":
    run()
