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

import numpy as np
import itertools

import paddle
import paddle.nn.functional as F
from paddlenlp.utils.log import logger


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader, name=''):
    """
    Given a dataset, it evaluates model and computes the metric.
    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
    """

    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels, _, _ = batch
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)

    acc = metric.accumulate()
    logger.info("%s: eval loss: %.5f, acc: %.5f" % (name, np.mean(losses), acc))
    model.train()
    metric.reset()

    return acc


def preprocess_function(example, tokenizer, max_seq_length, is_test=False):
    """
    Builds model inputs from a sequence for sequence classification tasks
    by concatenating and adding special tokens.
        
    Args:
        example(obj:`list[str]`): input data, containing text and label if it have label.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_length(obj:`int`): The maximum total input sequence length after tokenization. 
            Sequences longer than this will be truncated, sequences shorter will be padded.
        label_nums(obj:`int`): The number of the labels.
    Returns:
        result(obj:`dict`): The preprocessed data including input_ids, token_type_ids, labels.
    """
    if is_test:
        result = tokenizer(text=example['text_a'], max_seq_len=max_seq_length, return_attention_mask=True)
        return result['input_ids'], result['token_type_ids']
    else:
        tokens = example['tokens']
        rationales = example['rationales']
        tokens = [tokenizer._tokenize(token) for token in tokens]
        assert len(tokens) == len(rationales)
        rationales = list(
            itertools.chain(*[[rationale] * len(sub_tokens) for sub_tokens, rationale in zip(tokens, rationales)]))
        tokens = list(itertools.chain(*tokens))
        result = tokenizer(text=tokens,
                           max_seq_len=max_seq_length,
                           is_split_into_words=True,
                           return_attention_mask=True)
        input_ids = result["input_ids"]
        token_type_ids = result["token_type_ids"]
        attention_mask = result["attention_mask"]
        seq_len = len(input_ids)
        rationales = [0] + rationales[:seq_len - 2] + [0]
        assert len(rationales) == seq_len
        label = np.array([example['label']], dtype="int64")
        return input_ids, token_type_ids, label, rationales, attention_mask
