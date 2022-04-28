"""data class"""
# !/usr/bin/env python3
from dataclasses import dataclass
from typing import Any
from typing import List
from typing import Dict
from typing import Tuple


@dataclass
class AttentionResult(object):
    attributions: List[float]
    pred_label: float
    pred_proba: List[float]


@dataclass
class GradShapResult(object):
    attributions: List[float]
    pred_label: float
    pred_proba: List[float]


@dataclass
class IGResult(object):
    attributions: List[float]
    pred_label: float
    pred_proba: List[float]
    error_percent: float


@dataclass
class LimeResult(object):
    attributions: List[float]
    pred_label: float
    pred_proba: List[float]
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
