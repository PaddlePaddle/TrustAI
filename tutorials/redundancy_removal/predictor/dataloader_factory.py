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

import paddle
from datasets import load_dataset
from paddlenlp.data import Dict, Pad, Stack


def prepare_validation_features(args, examples, tokenizer):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    # NOTE: Almost the same functionality as HuggingFace's prepare_train_features function. The main difference is
    # that HugggingFace uses ArrowTable as basic data structure, while we use list of dictionary instead.
    contexts = examples['context']
    questions = examples['question']

    tokenized_examples = tokenizer(questions,
                                   contexts,
                                   stride=args.doc_stride,
                                   max_seq_len=args.max_seq_length,
                                   return_attention_mask=True)

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples['token_type_ids'][i]
        context_index = 1

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [(o if sequence_ids[k] == context_index else None)
                                                   for k, o in enumerate(tokenized_examples["offset_mapping"][i])]

    return tokenized_examples


def prepare_train_features(args, examples, tokenizer):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    # NOTE: Almost the same functionality as HuggingFace's prepare_train_features function. The main difference is
    # that HugggingFace uses ArrowTable as basic data structure, while we use list of dictionary instead.
    contexts = examples['context']
    questions = examples['question']

    tokenized_examples = tokenizer(questions, contexts, stride=args.doc_stride, max_seq_len=args.max_seq_length)

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples['token_type_ids'][i]

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples['answers'][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


# Asynchronous loading Dataloader
def get_dataloader(args, tokenizer, split="train", num_workers=4, return_raw_data=False):
    if split == "dev":
        split = "validation"
    data_examples = load_dataset('./utils/dureader_robust.py',
                                 name="plain_text",
                                 split=split,
                                 cache_dir="./cache",
                                 data_dir=args.data_dir,
                                 do_train=args.do_train,
                                 do_predict=args.do_predict)
    column_names = data_examples.column_names
    if split == "validation" or split == "test":
        data_ds = data_examples.map(lambda x: prepare_validation_features(args, x, tokenizer),
                                    load_from_cache_file=False,
                                    batched=True,
                                    remove_columns=column_names,
                                    num_proc=num_workers)
        data_batch_sampler = paddle.io.BatchSampler(data_ds, batch_size=args.batch_size, shuffle=False)
        data_batchify_fn = lambda samples, fn=Dict({
            "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
            "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id)
        }): fn(samples)
    elif split == "train":
        data_ds = data_examples.map(lambda x: prepare_train_features(args, x, tokenizer),
                                    batched=True,
                                    remove_columns=column_names,
                                    num_proc=num_workers)
        data_batch_sampler = paddle.io.DistributedBatchSampler(data_ds, batch_size=args.batch_size, shuffle=True)
        data_batchify_fn = lambda samples, fn=Dict({
            "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
            "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
            "start_positions": Stack(dtype="int64"),
            "end_positions": Stack(dtype="int64")
        }): fn(samples)

    data_loader = paddle.io.DataLoader(dataset=data_ds,
                                       batch_sampler=data_batch_sampler,
                                       collate_fn=data_batchify_fn,
                                       return_list=True)
    if return_raw_data:
        return data_loader, data_examples
    return data_loader
