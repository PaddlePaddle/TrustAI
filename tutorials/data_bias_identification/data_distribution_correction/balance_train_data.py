import re
import json
import collections
import random
import time
import os
import argparse

import numpy as np
import paddle
import jieba
from paddle.io import DataLoader, BatchSampler
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.utils.log import logger
from trustai.interpretation import get_word_offset
from trustai.interpretation import IntGradInterpreter
from LAC import LAC
from tqdm import tqdm

from utils import evaluate, preprocess_function

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default=None, help="file path of input data.")
parser.add_argument("--output_path", type=str, default=None, help="file path of output data.")

parser.add_argument("--seed", type=int, default=3, help="random seed for initialization")
parser.add_argument("--rationale_path",
                    type=str,
                    default="./data/rationale_importance.txt",
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


def run():
    """
    Get rationale importance
    """
    set_seed(args.seed)

    # init lexical analyzer of chinese
    lac = LAC(mode='lac')

    # load ratioanle importance
    with open(args.rationale_path, 'r') as f:
        tokens = []
        for line in f:
            if line.strip():
                token, frequency = line.split('\t')
                frequency = int(frequency)
                if frequency > 2:
                    tokens.append(token)
    # load ChnSentiCorp train data
    with open(args.input_path, 'r') as f:
        examples = []
        for i, line in enumerate(tqdm(list(f))):
            label, text = line.strip().split('\t')
            examples.append((i, int(label), text, list(lac.run(text)[0])))

    # Statistics rationale index in positive and negative examples respectively
    pos_dict = collections.defaultdict(list)
    neg_dict = collections.defaultdict(list)
    rate_dict = {}
    for i, token in enumerate(tqdm(tokens[::-1])):
        for example in examples:
            if token in example[3]:
                if example[1] == 1:
                    pos_dict[token].append(example[0])
                else:
                    neg_dict[token].append(example[0])

    # filter rationale by postag and positive negative ratio
    for token in sorted(list(set(pos_dict.keys()) & set(neg_dict.keys()))):
        pos_list = pos_dict[token]
        neg_list = neg_dict[token]
        pos_ratio = len(pos_list) / (len(pos_list) + len(neg_list))
        postags = lac.run(token)[1]
        if (pos_ratio <= 0.25 or pos_ratio >= 0.75) and not (set(['r', 'd', 'w', 'm']) & set(postags)):
            rate_dict[token] = [pos_ratio if pos_ratio < 0.5 else 1 - pos_ratio, len(pos_list), len(neg_list), postags]

    print(list(rate_dict))
    # sampling the data that will be added to the training set
    add_list = []
    for token in rate_dict:
        pos_num = len(pos_dict[token])
        neg_num = len(neg_dict[token])
        if pos_num > neg_num:
            add_list.extend(random.choices(neg_dict[token], k=min(pos_num - neg_num, neg_num * 2)))
        else:
            add_list.extend(random.choices(pos_dict[token], k=min(neg_num - pos_num, pos_num * 2)))

    # write data to train data
    logger.info(f"add number: {len(add_list)}")
    with open(args.output_path, 'w') as f:
        for example in examples:
            f.write(str(example[1]) + '\t' + example[2] + '\n')
        for idx in add_list:
            example = examples[idx]
            f.write(str(example[1]) + '\t' + example[2] + '\n')


if __name__ == "__main__":
    run()
