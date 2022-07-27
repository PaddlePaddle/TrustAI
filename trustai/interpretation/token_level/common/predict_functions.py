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
"""predict_functions"""

import numpy as np
import paddle
from paddle import tensor
from paddle.fluid import layers


def attention_predict_fn_on_paddlenlp(inputs,
                                      paddle_model,
                                      query_name="self_attn.q_proj",
                                      key_name="self_attn.k_proj",
                                      layer=11):
    """attention_predict_fn_on_paddlenlp"""
    query_feature = []
    key_feature = []

    def hook_for_query(layer, input, output):
        """hook_for_query"""
        query_feature.append(output)
        return output

    def hook_for_key(layer, input, output):
        """hook_for_key"""
        key_feature.append(output)
        return output

    hooks = []
    for name, v in paddle_model.named_sublayers():
        if str(layer) + '.' + query_name in name:
            h = v.register_forward_post_hook(hook_for_query)
            hooks.append(h)
        if str(layer) + '.' + key_name in name:
            h = v.register_forward_post_hook(hook_for_key)
            hooks.append(h)
    if isinstance(inputs, (list, tuple)):
        logits = paddle_model(*inputs)  # get logits, [bs, num_c]
    else:
        logits = paddle_model(inputs)  # get logits, [bs, num_c]
    bs = logits.shape[0]
    for h in hooks:
        h.remove()

    num_heads = paddle_model.ernie.config['num_attention_heads']
    hidden_size = paddle_model.ernie.config['hidden_size']
    head_dim = hidden_size // num_heads
    q = tensor.reshape(x=query_feature[0], shape=[0, 0, num_heads, head_dim])
    q = tensor.transpose(x=q, perm=[0, 2, 1, 3])
    k = tensor.reshape(x=key_feature[0], shape=[0, 0, num_heads, head_dim])
    k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
    attention = layers.matmul(x=q, y=k, transpose_y=True, alpha=head_dim**-0.5)
    
    attention = attention.sum(1)[:, 0]

    probas = paddle.nn.functional.softmax(logits, axis=1)  # get probabilities.
    preds = paddle.argmax(probas, axis=1)  # get predictions.

    # logits or probas
    preds = preds.reshape((bs, ))

    return attention.numpy(), preds.numpy(), probas.numpy()


def general_predict_fn(inputs, paddle_model):
    """general predict function"""

    if isinstance(inputs, (list, tuple)):
        logits = paddle_model(*inputs)  # get logits, [bs, num_c]
    else:
        logits = paddle_model(inputs)  # get logits, [bs, num_c]
    probas = paddle.nn.functional.softmax(logits, axis=1)  # get probabilities.
    preds = paddle.argmax(probas, axis=1)  # get predictions.

    return preds.numpy(), probas.numpy()


def ig_predict_fn_on_paddlenlp_pretrain(inputs,
                                        label,
                                        left=None,
                                        right=None,
                                        steps=None,
                                        paddle_model=None,
                                        embedding_name=None):
    """ig_predict_fn_on_paddlenlp_pretrain"""

    target_feature_map = []
    for i in inputs:
        i.stop_gradient = True

    def hook(layer, input, output):
        if steps is not None:
            noise_array = np.arange(left, right) / steps
            output_shape = output.shape
            assert right - left == output_shape[0]
            noise_tensor = paddle.to_tensor(noise_array, dtype=output.dtype)
            noise_tensor = noise_tensor.unsqueeze(axis=list(range(1, len(output_shape))))
            output = noise_tensor * output
        target_feature_map.append(output)
        return output

    hooks = []
    for name, v in paddle_model.named_sublayers():
        if embedding_name in name:
            h = v.register_forward_post_hook(hook)
            hooks.append(h)
    assert isinstance(inputs, (tuple, list))
    *model_inputs, masked_positions = inputs
    logits = paddle_model(*model_inputs)  # get logits, [bs, num_c]

    bs = logits.shape[0]

    for h in hooks:
        h.remove()

    probas = paddle.nn.functional.softmax(logits, axis=2)  # get probabilities.
    preds = paddle.argmax(probas, axis=2)  # get predictions.
    mask_num = masked_positions.sum(axis=1).tolist()
    logits = paddle.gather_nd(logits, layers.where(masked_positions))
    logits = logits.split(num_or_sections=mask_num)
    probas = paddle.gather_nd(probas, layers.where(masked_positions))
    probas = probas.split(num_or_sections=mask_num)
    preds = paddle.gather_nd(preds, layers.where(masked_positions))
    preds = preds.split(num_or_sections=mask_num)

    if label is None:
        assert bs == 1
        label = [preds[0].numpy()]
    labels = label * bs

    loss = 0
    for proba, ori_label in zip(probas, labels):
        ori_label = paddle.to_tensor(ori_label).unsqueeze(axis=1)
        loss += proba.index_sample(ori_label).sum()
    loss = loss / masked_positions.sum()

    loss.backward()
    gradients = target_feature_map[0].grad  # get gradients of "embedding".
    loss.clear_gradient()

    gradients = gradients.numpy()
    probas = [proba.numpy() for proba in probas]
    return gradients, labels, target_feature_map[0].numpy(), probas


def attention_predict_fn_on_paddlenlp_pretrain(inputs,
                                               paddle_model,
                                               query_name="self_attn.q_proj",
                                               key_name="self_attn.k_proj",
                                               layer=11):
    """attention_predict_fn_on_paddlenlp"""
    query_feature = []
    key_feature = []

    def hook_for_query(layer, input, output):
        """hook_for_query"""
        query_feature.append(output)
        return output

    def hook_for_key(layer, input, output):
        """hook_for_key"""
        key_feature.append(output)
        return output

    hooks = []
    for name, v in paddle_model.named_sublayers():
        if str(layer) + '.' + query_name in name:
            h = v.register_forward_post_hook(hook_for_query)
            hooks.append(h)
        if str(layer) + '.' + key_name in name:
            h = v.register_forward_post_hook(hook_for_key)
            hooks.append(h)

    assert isinstance(inputs, (tuple, list)) and len(inputs) == 3
    *model_inputs, masked_positions = inputs
    logits = paddle_model(*model_inputs)  # get logits, [bs, num_c]

    bs = logits.shape[0]
    for h in hooks:
        h.remove()

    num_heads = paddle_model.ernie.config['num_attention_heads']
    hidden_size = paddle_model.ernie.config['hidden_size']
    head_dim = hidden_size // num_heads
    q = tensor.reshape(x=query_feature[0], shape=[0, 0, num_heads, head_dim])
    q = tensor.transpose(x=q, perm=[0, 2, 1, 3])
    k = tensor.reshape(x=key_feature[0], shape=[0, 0, num_heads, head_dim])
    k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
    attention = layers.matmul(x=q, y=k, transpose_y=True, alpha=head_dim**-0.5)

    attention = attention.sum(1)
    mask_sum_attention = (attention * masked_positions.unsqueeze(2)).sum(axis=1)
    mask_num = masked_positions.sum(axis=1, keepdim=True)
    attention = mask_sum_attention / mask_num
    probas = paddle.nn.functional.softmax(logits, axis=2)  # get probabilities.
    preds = paddle.argmax(probas, axis=2)  # get predictions.
    mask_num = mask_num.squeeze().tolist()
    probas = paddle.gather_nd(probas, layers.where(masked_positions))
    probas = probas.split(num_or_sections=mask_num)
    preds = paddle.gather_nd(preds, layers.where(masked_positions))
    preds = preds.split(num_or_sections=mask_num)

    return attention.numpy(), preds, probas


def general_predict_fn_on_paddlenlp_pretrain(inputs, paddle_model):
    """general_predict_fn_on_paddlenlp_pretrain"""

    *model_inputs, masked_positions = inputs
    if isinstance(model_inputs, (tuple, list)):
        logits = paddle_model(*model_inputs)  # get logits, [bs, num_c]
    else:
        logits = paddle_model(model_inputs)  # get logits, [bs, num_c]

    mask_num = masked_positions.sum(axis=1, keepdim=True)
    probas = paddle.nn.functional.softmax(logits, axis=2)  # get probabilities.
    preds = paddle.argmax(probas, axis=2)  # get predictions.
    mask_num = mask_num.squeeze().tolist()
    probas = paddle.gather_nd(probas, layers.where(masked_positions))
    probas = probas.split(num_or_sections=mask_num)
    preds = paddle.gather_nd(preds, layers.where(masked_positions))
    preds = preds.split(num_or_sections=mask_num)

    return preds, probas