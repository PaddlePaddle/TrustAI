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
"""data class"""

from dataclasses import dataclass
from typing import Any
from typing import List
from typing import Dict
from typing import Tuple


@dataclass
class TokenResult(object):
    attributions: List[float]
    pred_label: float
    pred_proba: List[float]


@dataclass
class AttentionResult(TokenResult):
    pass


@dataclass
class GradShapResult(TokenResult):
    pass


@dataclass
class IGResult(TokenResult):
    error_percent: float


@dataclass
class LimeResult(TokenResult):
    lime_score: float


@dataclass
class NormLIMEResult(object):
    # {id : (attribution, word_idx)}
    attributions: Dict[int, Tuple[float, int]]


@dataclass
class InterpretResult(object):
    words: List[str]
    word_attributions: List[float]
    pred_label: float
    pred_proba: List[float]
    rationale: List[int]
    non_rationale: List[int]
    rationale_tokens: List[str]
    non_rationale_tokens: List[str]
    rationale_pred_proba: float = None
    non_rationale_pred_proba: float = None
