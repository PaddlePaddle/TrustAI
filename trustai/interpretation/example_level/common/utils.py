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