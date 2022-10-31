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
"""Some useful functions."""
import paddle
import paddle.nn.functional as F
import numpy as np
from .data_class import ExampleResult


def get_sublayer(model, sublayer_name='classifier'):
    """
    Get the sublayer named sublayer_name in model.
    Args:
        model (obj:`paddle.nn.Layer`): Any paddle model.
        sublayer_name (obj:`str`, defaults to classifier): The sublayer name.
    Returns:
        layer(obj:`paddle.nn.Layer.common.sublayer_name`):The sublayer named sublayer_name in model.
    """
    for name, layer in model.named_children():
        if name == sublayer_name:
            return layer


def dot_similarity(inputs_a, inputs_b):
    """
    calaculate dot-product similarity between the two inputs.
    """
    return paddle.sum(inputs_a * inputs_b, axis=1)


def cos_similarity(inputs_a, inputs_b, step=500000):
    """
    calaculate cosine similarity between the two inputs.
    """
    # Processing to avoid paddle bug
    start, end = 0, step
    res = []
    while start < inputs_a.shape[0]:
        res.append(F.cosine_similarity(inputs_a[start:end], inputs_b.unsqueeze(0)))
        start = end
        end = end + step
    return paddle.concat(res, axis=0)


def euc_similarity(inputs_a, inputs_b):
    """
    calaculate  euclidean similarity between the two inputs.
    """
    return -paddle.linalg.norm(inputs_a - inputs_b.unsqueeze(0), axis=-1).squeeze(-1)


def get_top_and_bottom_n_examples(scores, pred_label, sample_num=3):
    """
    get n index of the highest and lowest score, return the structual result.
    """

    top_score, top_index = paddle.topk(scores, sample_num, axis=0, largest=True, sorted=True)

    bottom_score, bottom_index = paddle.topk(scores, sample_num, axis=0, largest=False, sorted=True)

    res = ExampleResult(pred_label=pred_label,
                        pos_indexes=top_index.numpy(),
                        neg_indexes=bottom_index.numpy(),
                        pos_scores=top_score.numpy(),
                        neg_scores=bottom_score.numpy())

    return res
