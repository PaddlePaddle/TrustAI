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

import json
import os

import numpy as np
import paddle
from paddlenlp.data import Dict, Pad
from utils import logger, tools


# Read data from file path
def get_data(filepath):
    with open(filepath, encoding="utf-8") as f:
        durobust = json.load(f)
    data = []
    for article in durobust["data"]:
        title = article.get("title", "")
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                answer_starts = [answer["answer_start"] for answer in qa.get("answers", '')]
                answers = [answer["text"] for answer in qa.get("answers", '')]
                # Features currently used are "context", "question", and "answers".
                # Others are extracted here for the ease of future expansions.
                data.append({
                    "title": title,
                    "context": context,
                    "question": qa["question"],
                    "id": qa["id"],
                    "answers": {
                        "answer_start": answer_starts,
                        "text": answers,
                    },
                })
    return data


# Process data
def get_dataset(args, tokenizer, split="train", return_raw_data=False):
    dataset = get_data(os.path.join(args.data_dir, split + ".json"))
    set_k_sentences_ground_true = args.set_k_sentences_ground_true
    final_dataset = []
    BSENT_ID = tokenizer._convert_token_to_id_with_added_voc("[BSENT]")
    ESENT_ID = tokenizer._convert_token_to_id_with_added_voc("[ESENT]")
    SEP_ID = tokenizer.sep_token_id
    CLS_ID = tokenizer.cls_token_id

    # Preprocess data
    for d in dataset:
        # Split into sentences and tokenize these sentences
        sentences = tools.split_sentence(d["context"])
        tokenized = tokenizer(sentences)
        input_ids = tokenized["input_ids"]

        temp_context_input_ids = [CLS_ID]
        end_list = [0]
        temp_sentence_label = []

        # Add special tokens, re-organize input_ids and label the sentences which contain answers
        for i, input_id in enumerate(input_ids):
            temp_context_input_ids.append(BSENT_ID)
            temp_context_input_ids += input_id[1:-1]
            temp_context_input_ids.append(ESENT_ID)
            end_list.append(end_list[-1] + len(input_id))
            flag = False
            for answer in d['answers']["text"]:
                if answer in sentences[i]:
                    flag = True
            if flag:
                temp_sentence_label.append(1)
            else:
                temp_sentence_label.append(0)

        # Use loose strategy to set k sentences true nearby ground true sentences to compute loss to improve recall.
        # set_k_sentences_ground_true = 1: [0,0,1,0,0] -> [0,1,1,1,0]
        sentence_label = [0] * len(temp_sentence_label)
        for i_sent, sentence in enumerate(temp_sentence_label):
            if sentence == 1:
                for j_sent in range(max(0, i_sent - set_k_sentences_ground_true),
                                    min(len(sentences), i_sent + set_k_sentences_ground_true + 1)):
                    sentence_label[j_sent] = 1

        # Set the special token position list, viz [BSENT] and [ESENT] positions in input_ids
        # `[CLS] [BSENT] a b c [ESENT] [SEP]` -> start_list=[1], end=[5]
        start_list = [x + 1 for x in end_list][:-1]
        end_list.remove(0)

        # splicing context input_ids and question input_ids
        temp_question = tokenizer(d["question"])
        temp_input_ids = temp_context_input_ids + [SEP_ID] + temp_question["input_ids"][1:]

        # ignore too long data
        if (len(temp_input_ids) <= args.max_seq_length):
            processed_data = {
                "input_ids":
                np.array(temp_input_ids),
                "token_type_ids":
                np.array([0] * (len(temp_context_input_ids) + 1) + [1] * (len(temp_question["input_ids"]) - 1)),
                "attention_mask":
                np.array([1] * len(temp_input_ids)),
                "start_list":
                start_list,
                "end_list":
                end_list,
                "ground_label_list":
                temp_sentence_label,
                "label_list":
                np.array(sentence_label),
                "question_input_ids":
                np.array(temp_question["input_ids"]),
                "question_token_type_ids":
                np.array(temp_question["token_type_ids"]),
                "question_attention_mask":
                np.array([1] * len(temp_question["input_ids"]))
            }
            if return_raw_data:
                processed_data["raw_data"] = {"sentences": sentences, "id": d["id"]}
            final_dataset.append(processed_data)

    return final_dataset


# Iterator Dataset: a easy wrapper class
class DuRobustDataset(paddle.io.Dataset):

    def __init__(self, args, tokenizer, split="train", return_raw_data=False):
        self.dataset = get_dataset(args, tokenizer, split, return_raw_data)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


# Stack a list of data for dataloader to padding
class StackList(object):

    def __call__(self, data):
        return [data]


# Asynchronous loading Dataloader


def get_dataloader(args, data_dir_path, batch_size, tokenizer, split="train", return_raw_data=False, num_workers=4):
    log = logger.Logger(args)
    log.info("Loading data From " + str(os.path.join(data_dir_path, split + ".json")) + "...")
    dataset = DuRobustDataset(args, tokenizer=tokenizer, split=split, return_raw_data=return_raw_data)
    log.info("Loading Finished.")
    log.info("Preprocessing....")
    shuffle = True if split == 'train' else False

    if split == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)

    data_function_map = {
        "input_ids": Pad(axis=0, pad_val=0),
        "token_type_ids": Pad(axis=0, pad_val=0),
        "attention_mask": Pad(axis=0, pad_val=0),
        "label_list": Pad(axis=0, pad_val=-100),
        "question_input_ids": Pad(axis=0, pad_val=0),
        "question_token_type_ids": Pad(axis=0, pad_val=0),
        "question_attention_mask": Pad(axis=0, pad_val=0),
        "start_list": StackList(),
        "end_list": StackList(),
        "ground_label_list": StackList(),
    }
    if return_raw_data:
        data_function_map["raw_data"] = StackList()
    batchify_fn = lambda samples, fn=Dict(data_function_map): [data for data in fn(samples)]
    # use paddle.io.DataLoader to create Asynchronous DataLoader
    data_loader = paddle.io.DataLoader(dataset=dataset,
                                       batch_sampler=batch_sampler,
                                       collate_fn=batchify_fn,
                                       num_workers=num_workers)

    log.info("Preprocessing finished.")
    return data_loader
