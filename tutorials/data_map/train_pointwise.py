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

from functools import partial
import argparse
import os
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F

import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.transformers.roberta.tokenizer import RobertaTokenizer, RobertaBPETokenizer
from paddlenlp.transformers.bert.tokenizer import BertTokenizer
from paddlenlp.transformers import AutoTokenizer, AutoModelForSequenceClassification
from paddlenlp.datasets import DatasetBuilder
from data import create_dataloader
from data import convert_pointwise_example as convert_example
import sys

parser = argparse.ArgumentParser()
parser.add_argument(
    "--save_dir",
    default='./checkpoint',
    type=str,
    help="The output directory where the model checkpoints will be written.")
parser.add_argument(
    "--max_seq_length",
    default=256,
    type=int,
    help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded."
)
parser.add_argument(
    "--batch_size",
    default=32,
    type=int,
    help="Batch size per GPU/CPU for training.")
parser.add_argument(
    "--learning_rate",
    default=3e-5,
    type=float,
    help="The initial learning rate for Adam.")
parser.add_argument(
    "--weight_decay",
    default=0.0,
    type=float,
    help="Weight decay if we apply some.")
parser.add_argument(
    "--epochs",
    default=5,
    type=int,
    help="Total number of training epochs to perform.")
parser.add_argument(
    "--eval_step", default=800, type=int, help="Step interval for evaluation.")
parser.add_argument(
    '--save_step',
    default=500,
    type=int,
    help="Step interval for saving checkpoint.")

parser.add_argument(
    '--data_dir',
    type=str,
    default='./data',
    help='data directory includes train / develop data')
parser.add_argument(
    '--train_set',
    type=str,
    required=True,
    help='Path to training data.')
parser.add_argument(
    '--dev_set',
    type=str,
    required=True,
    help='Path to validation data.')

parser.add_argument(
    "--warmup_proportion",
    default=0.1,
    type=float,
    help="Linear warmup proption over the training process.")
parser.add_argument(
    "--init_from_ckpt",
    type=str,
    default=None,
    help="The path of checkpoint to be loaded.")
parser.add_argument(
    "--seed", type=int, default=1000, help="Random seed for initialization.")
parser.add_argument(
    '--device',
    choices=['cpu', 'gpu'],
    default="gpu",
    help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()

base_model = 'bert_base'


class sim_data(DatasetBuilder):
    
    def _read(self, filename):
        with open(filename, "r", encoding="utf8") as f:            
            for line in f.readlines():
                line = line.strip()
                line = line.split('\t')                
                if line[0] == 'text_a':
                    continue
                elif len(line) == 4:
                    yield{#'id': line[0],
                    'text_t': line[0],
                    'text_q': line[1],
                    'label': line[2],
                    's_label':line[3]}
                elif len(line) == 3:
                    yield{#'id': line[0],
                    'text_t': line[0],
                    'text_q': line[1],
                    'label': line[2],
                    's_label': "0"}
                else:
                    continue
                
def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader, phase="dev"):
    """
    Given a dataset, it evals model and computes the metric.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    model.eval()
    metric.reset()
    losses = []
    dev_preds, preds_label, all_loss = [], [], []
    dev_labels, dev_s_labels= [], []
    hard_loss, hard_num, normal_loss, normal_num, noisy_loss, noisy_num = 0, 0, 0, 0, 0, 0
    wrong_hard, wrong_noisy, wrong_clean = 0, 0, 0
    for batch in data_loader:
        input_ids, token_type_ids, labels, s_labels, sep_ids = batch
        loss = 0.0
        probs = model(input_ids=input_ids, token_type_ids=token_type_ids)
        dev_labels.extend(labels.numpy().tolist())
        dev_s_labels.extend(s_labels.numpy().tolist())
        
        dev_preds.extend(F.softmax(probs, axis=1).numpy().tolist())
        preds_label.extend(np.argmax(F.softmax(probs, axis=1).numpy(), axis=1).tolist())

        for j in range(len(s_labels)):
            all_loss.append(criterion(probs[j], labels[j]).numpy().tolist())
            if s_labels[j].item() == 1: # hard
                hard_loss += criterion(probs[j], labels[j])
                hard_num += 1
            elif s_labels[j].item() == 2:
                noisy_loss += criterion(probs[j], labels[j])
                noisy_num += 1
            else:
                normal_loss += criterion(probs[j], labels[j])
                normal_num += 1
        
        loss = criterion(probs, labels)
        losses.append(loss.numpy())
        correct = metric.compute(probs, labels)
        metric.update(correct)
        accu = metric.accumulate()
    
    print("eval {} loss: {:.5}, accu: {:.5}".format(phase, np.mean(losses), accu))
    print('noisy num: %d, hard num: %d, clean num: %d' % (noisy_num, hard_num, normal_num))
    print('wrong noisy: %d, wrong hard: %d, wrong clean: %d' % (wrong_noisy, wrong_hard, wrong_clean))
    all_data = []
    for i in range(len(dev_labels)):
        data = {}
        data['id'] = i
        data['label'] = dev_labels[i][0]
        data['pred_label'] = preds_label[i]
        data['noisy_label'] = dev_s_labels[i][0]
        
        if data['label'] != data['pred_label']:
            data['correct'] = 'false'
        else:
            data['correct'] = 'true'
        data['loss'] = all_loss[i][0]
        data['probs'] = dev_preds[i]
        data['label_probs'] = dev_preds[i][int(data['label'])]
        all_data.append(data)

    with open('./outputs/output_data.json', 'a+') as f:
        for i in range(len(all_data)):
            f.write(str(all_data[i]) + '\n') 
    model.train()
    metric.reset()


def do_train():
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)

    train_file = os.path.join(args.data_dir, args.train_set)
    dev_file = os.path.join(args.data_dir, args.dev_set)
    train_ds = sim_data().read(train_file) 
    dev_ds = sim_data().read(dev_file)

    tokenizer = AutoTokenizer.from_pretrained('ernie-3.0-base-zh')
    pretrained_model = AutoModelForSequenceClassification.from_pretrained('ernie-3.0-base-zh', num_classes=2)

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        language='ch')

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # text_pair_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # text_pair_segment
        Stack(dtype="int64"),  # label
        Stack(dtype="int64"),  # s_label
        Stack(dtype="int64") # sep_ids
    ): [data for data in fn(samples)]

    train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    dev_data_loader = create_dataloader(
        dev_ds,
        mode='dev',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    model = pretrained_model

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)

    model = paddle.DataParallel(model)

    num_training_steps = len(train_data_loader) * args.epochs

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                        args.warmup_proportion)

    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    global_step = 0
    tic_train = time.time()
    for epoch in range(1, args.epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            loss = 0.0
            input_ids, token_type_ids, labels, s_labels, sep_ids = batch
            probs= model(input_ids=input_ids, token_type_ids=token_type_ids)

            loss = criterion(probs, labels)
            correct = metric.compute(probs, labels)
            metric.update(correct)
            acc = metric.accumulate()

            global_step += 1
            if global_step % 100 == 0 and rank == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, acc,
                    100 / (time.time() - tic_train)),
                    flush=True)
                tic_train = time.time()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if global_step % args.eval_step == 0 and rank == 0:
                evaluate(model, criterion, metric, dev_data_loader)

            # save model
            # if global_step % args.save_step == 0 and rank == 0:
            #     save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
            #     if not os.path.exists(save_dir):
            #         os.makedirs(save_dir)
            #     save_param_path = os.path.join(save_dir, 'model_state.pdparams')
            #     paddle.save(model.state_dict(), save_param_path)
            #     tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    do_train()
