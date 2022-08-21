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
import collections

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle.metric import Accuracy
from paddle.io import DataLoader, BatchSampler, DistributedBatchSampler
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer, LinearDecayWithWarmup
from paddlenlp.utils.log import logger
from paddlenlp.data import Stack, Tuple, Pad

from utils import evaluate, preprocess_function

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir",
                    default="./checkpoint",
                    type=str,
                    help="The output directory where the model checkpoints will be written.")
parser.add_argument("--dataset_dir",
                    default="./data",
                    type=str,
                    help="The dataset directory should include train.tsv, dev.tsv and test.tsv files.")
parser.add_argument("--train_file", type=str, default=None, help="train data filename")
parser.add_argument("--dev_file", type=str, default=None, help="dev data filename")
parser.add_argument("--test_files", type=str, nargs='*', default=None, help="test data filenames")
parser.add_argument("--max_seq_length",
                    default=256,
                    type=int,
                    help="The maximum total input sequence length after tokenization. "
                    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument('--model_name',
                    default="ernie-2.0-base-en",
                    help="Select model to train, defaults to ernie-2.0-base-en.")
parser.add_argument('--device',
                    choices=['cpu', 'gpu', 'xpu', 'npu'],
                    default="gpu",
                    help="Select which device to train model, defaults to gpu.")
parser.add_argument("--batch_size", default=2, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
parser.add_argument('--early_stop', type=bool, default=True, help='Epoch before early stop.')
parser.add_argument('--early_stop_nums', type=int, default=2, help='Number of epoch before early stop.')
parser.add_argument("--epochs", default=1000, type=int, help="Total number of training epochs to perform.")
parser.add_argument('--warmup', type=bool, default=True, help="whether use warmup strategy")
parser.add_argument("--warmup_steps", default=100, type=int, help="Linear warmup steps over the training process.")
parser.add_argument("--logging_steps", default=100, type=int, help="The interval steps to logging.")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
parser.add_argument("--seed", type=int, default=3, help="random seed for initialization")
parser.add_argument('--num_classes', type=int, default=2, help='Number of classification.')
parser.add_argument("--lam", type=float, default=0.01, help="Weight of MAW loss")
parser.add_argument("--use_maw", action="store_true", help="Whether to enhance the model by using ratioanles")

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
        # skip header
        next(f)
        for line in f:
            text, tokens, label, rationales = line.strip().split('\t')
            tokens = json.loads(tokens)
            rationales = json.loads(rationales)
            yield {"text_a": text, "label": int(label), "tokens": tokens, "rationales": rationales}


def cal_maw_loss(inputs,
                 paddle_model,
                 criterion,
                 query_name="self_attn.q_proj",
                 key_name="self_attn.k_proj",
                 layers=[11],
                 lam=0.01,
                 use_maw=True):
    """calculate mean attention weight loss"""
    input_ids, token_type_ids, labels, rationales, attention_mask = inputs

    query_feature = collections.defaultdict(list)
    key_feature = collections.defaultdict(list)

    def hook_for_query(layer, input, output):
        """hook_for_query"""
        query_feature[layer.alias].append(output)
        return output

    def hook_for_key(layer, input, output):
        """hook_for_key"""
        key_feature[layer.alias].append(output)
        return output

    hooks = []
    for name, v in paddle_model.named_sublayers():
        for layer in layers:
            if str(layer) + '.' + query_name in name:
                h = v.register_forward_post_hook(hook_for_query)
                v.alias = layer
                hooks.append(h)
            if str(layer) + '.' + key_name in name:
                h = v.register_forward_post_hook(hook_for_key)
                v.alias = layer
                hooks.append(h)

    logits = paddle_model(input_ids, token_type_ids)  # get logits, [bs, num_c]

    for h in hooks:
        h.remove()
    attentions = []

    num_heads = paddle_model._layers.ernie.config['num_attention_heads']
    hidden_size = paddle_model._layers.ernie.config['hidden_size']
    head_dim = hidden_size // num_heads
    for layer in layers:
        q = paddle.tensor.reshape(x=query_feature[layer][0], shape=[0, 0, num_heads, head_dim])
        q = paddle.tensor.transpose(x=q, perm=[0, 2, 1, 3])
        k = paddle.tensor.reshape(x=key_feature[layer][0], shape=[0, 0, num_heads, head_dim])
        k = paddle.tensor.transpose(x=k, perm=[0, 2, 1, 3])
        attention = paddle.matmul(x=q * (head_dim**-0.5), y=k, transpose_y=True)
        attentions.append(attention)

    maw_loss_list = []
    maw_loss = 0
    rationale_mask = paddle.cast(paddle.sum(rationales, axis=[1]) != 0, 'float32')
    cross_loss = criterion(logits, labels)

    if use_maw and paddle.sum(rationale_mask) > 0:
        rationales += (1 - attention_mask) * -1e6
        rationales = rationales.cast('float32')
        rationales = paddle.nn.functional.softmax(rationales)
        for i in range(attentions[-1].shape[1]):
            att_weights = attentions[-1][:, i, 0, :]
            att_weights += (1 - attention_mask) * -1e6

            loss_att = lam * paddle.sum(
                paddle.nn.functional.cross_entropy(att_weights, rationales, soft_label=True, reduction='none') *
                rationale_mask) / paddle.sum(rationale_mask)

            maw_loss_list.append(loss_att)
        maw_loss = paddle.add_n(maw_loss_list) / len(maw_loss_list)
    return logits, cross_loss, maw_loss


def train():
    """
    Training a hierarchical classification model
    """

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    set_seed(args.seed)
    paddle.set_device(args.device)

    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    train_path = os.path.join(args.dataset_dir, args.train_file)
    dev_path = os.path.join(args.dataset_dir, args.dev_file)
    train_ds = load_dataset(read, data_path=train_path, lazy=False)
    dev_ds = load_dataset(read, data_path=dev_path, lazy=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    trans_func = functools.partial(preprocess_function, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    train_ds = train_ds.map(trans_func)
    dev_ds = dev_ds.map(trans_func)

    # batchify dataset
    collate_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        Stack(dtype="int64"),  # label,
        Pad(axis=0, pad_val=0),  # rationales
        Pad(axis=0, pad_val=0),  # attention_mask
    ): [data for data in fn(samples)]
    if paddle.distributed.get_world_size() > 1:
        train_batch_sampler = DistributedBatchSampler(train_ds, batch_size=args.batch_size, shuffle=True)
    else:
        train_batch_sampler = BatchSampler(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_batch_sampler = BatchSampler(dev_ds, batch_size=args.batch_size, shuffle=False)
    train_data_loader = DataLoader(dataset=train_ds, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
    dev_data_loader = DataLoader(dataset=dev_ds, batch_sampler=dev_batch_sampler, collate_fn=collate_fn)

    # load test dataloader
    if args.test_files is not None:
        test_data_loaders = []
        for test_file in args.test_files:
            test_path = os.path.join(args.dataset_dir, test_file)
            test_ds = load_dataset(read, data_path=test_path, lazy=False).map(trans_func)
            test_batch_sampler = BatchSampler(test_ds, batch_size=args.batch_size, shuffle=False)
            test_data_loader = DataLoader(dataset=test_ds, batch_sampler=test_batch_sampler, collate_fn=collate_fn)
            test_data_loaders.append(test_data_loader)
    else:
        test_data_loaders = []

    # define model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_classes=args.num_classes)
    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
    model = paddle.DataParallel(model)

    num_training_steps = len(train_data_loader) * args.epochs
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps, args.warmup_steps)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    optimizer = paddle.optimizer.AdamW(learning_rate=lr_scheduler,
                                       parameters=model.parameters(),
                                       weight_decay=args.weight_decay,
                                       apply_decay_param_fun=lambda x: x in decay_params)

    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = Accuracy()

    global_step = 0
    best_dev_acc = 0
    early_stop_count = 0
    tic_train = time.time()
    test_file_num = len(test_data_loaders)
    test_accs = [0] * test_file_num
    best_test_accs = [0] * test_file_num

    for epoch in range(1, args.epochs + 1):

        if args.early_stop and early_stop_count >= args.early_stop_nums:
            logger.info("Early stop!")
            break

        for step, batch in enumerate(train_data_loader, start=1):

            logits, cross_loss, maw_loss = cal_maw_loss(batch, model, criterion, lam=args.lam, use_maw=args.use_maw)
            if args.use_maw:
                loss = cross_loss + maw_loss
            else:
                loss = cross_loss

            probs = F.softmax(logits, axis=1)
            correct = metric.compute(probs, batch[2])
            metric.update(correct)
            acc = metric.accumulate()

            loss.backward()
            optimizer.step()
            if args.warmup:
                lr_scheduler.step()
            optimizer.clear_grad()

            global_step += 1
            if global_step % args.logging_steps == 0 and rank == 0:
                logger.info(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, cross_loss: %.5f, maw_loss: %.5f, acc: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, cross_loss, maw_loss, acc, args.logging_steps /
                       (time.time() - tic_train)))
                tic_train = time.time()

        early_stop_count += 1
        dev_acc = evaluate(model, criterion, metric, dev_data_loader, "dev")
        if test_file_num != 0:
            for n, test_data_loader in enumerate(test_data_loaders):
                test_acc = evaluate(model, criterion, metric, test_data_loader, f"test_{n}")
                test_accs[n] = test_acc

        save_best_path = args.save_dir
        if not os.path.exists(save_best_path):
            os.makedirs(save_best_path)

        # save models
        if dev_acc > best_dev_acc:
            logger.info("Current best dev accuracy: %.5f" % (dev_acc))
            for n, test_acc in enumerate(test_accs):
                logger.info("Current best test_%d accuracy: %.5f" % (n, test_acc))
            early_stop_count = 0
            best_dev_acc = dev_acc
            best_test_accs = list(test_accs)
            model._layers.save_pretrained(save_best_path)
            tokenizer.save_pretrained(save_best_path)

    logger.info("Final best dev accuracy: %.5f" % (best_dev_acc))
    for n in range(test_file_num):
        logger.info("Final best test_%d accuracy: %.5f" % (n, best_test_accs[n]))
    logger.info("Save best accuracy text classification model in %s" % (args.save_dir))


if __name__ == "__main__":
    train()