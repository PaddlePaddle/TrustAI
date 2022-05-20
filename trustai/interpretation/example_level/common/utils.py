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


def cos_similarity(inputs_a, inputs_b):
    """
    calaculate cosine similarity between the two inputs.
    """
    return F.cosine_similarity(inputs_a, inputs_b.unsqueeze(0))


def euc_similarity(inputs_a, inputs_b):
    """
    calaculate  euclidean similarity between the two inputs.
    """
    return -paddle.linalg.norm(inputs_a - inputs_b.unsqueeze(0), axis=-1).squeeze(-1)


def get_top_and_bottom_n_examples(scores, sample_num=3):
    """
    get n index of the highest and lowest score.
    """
    index = paddle.flip(paddle.argsort(scores), axis=0)
    top_index = index[:sample_num].tolist()
    bottom_index = index[-sample_num:].tolist()
    return top_index, bottom_index