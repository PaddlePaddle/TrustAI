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
"""visualization function"""

from IPython.core.display import display, HTML

import numpy as np

from .data_class import TokenResult
from .data_class import InterpretResult


class VisualizationTextRecord(object):
    """
    A record for text visulization.
    Part of the code is modified from https://github.com/pytorch/captum/blob/master/captum/attr/_utils/visualization.py
    """

    def __init__(self, interpret_res, true_label=None, words=None):
        if words is not None:
            self.words = words
        else:
            self.words = interpret_res.words
        self.pred_label = interpret_res.pred_label
        if isinstance(self.pred_label, np.ndarray):
            self.pred_proba = [
                round(proba[label], 2) for proba, label in zip(interpret_res.pred_proba, self.pred_label)
            ]
            self.pred_label = self.pred_label.tolist()
        else:
            self.pred_proba = interpret_res.pred_proba[self.pred_label]
        self.true_label = true_label if true_label is not None else ''

        # Normalization for attributions
        if isinstance(interpret_res, InterpretResult):
            word_attributions = interpret_res.word_attributions
        else:
            word_attributions = interpret_res.attributions
        _max = max(word_attributions)
        _min = min(word_attributions)
        self.word_attributions = [(word_imp - _min) / (_max - _min) for word_imp in word_attributions]

    def record_html(self):
        """change all informations to html"""
        return "".join([
            "<tr>",
            self._format_class(self.true_label),
            self._format_class(self.pred_label, self.pred_proba),
            self._format_word_attributions(),
            "<tr>",
        ])

    def _format_class(self, label, prob=None):
        if prob is None:
            return '<td align="center"><text style="padding-right:2em"><b>{label}</b></text></td>'.format(label=label)
        elif isinstance(prob, list):
            return '<td align="center"><text style="padding-right:2em"><b>{label} ({prob})</b></text></td>'\
        .format(label=str(label), prob=str(prob))
        else:
            return '<td align="center"><text style="padding-right:2em"><b>{label} ({prob:.2f})</b></text></td>'\
        .format(label=label, prob=prob)

    def _format_word_attributions(self):
        tags = ["<td>"]
        for word, importance in zip(self.words, self.word_attributions[:len(self.words)]):
            color = self._background_color(importance)
            unwrapped_tag = '<mark style="background-color: {color}; opacity:1.0; \
                        line-height:1.75"><font color="black"> {word}\
                        </font></mark>' \
                        .format(color=color, word=word)
            tags.append(unwrapped_tag)
        tags.append("</td>")
        return "".join(tags)

    def _background_color(self, importance):
        importance = max(-1, min(1, importance))
        if importance > 0:
            hue = 120
            sat = 75
            lig = 100 - int(30 * importance)
        else:
            hue = 0
            sat = 75
            lig = 100 - int(-40 * importance)
        return "hsl({}, {}%, {}%)".format(hue, sat, lig)


def visualize_text(text_records):
    """visualize text"""
    html = ["<table width: 100%, align : center>"]
    rows = ["<tr><th>Golden Label</th>"
            "<th>Predicted Label (Prob)</th>"
            "<th>Important scores</th>"]
    for record in text_records:
        rows.append(record.record_html())
    html.append("".join(rows))
    html.append("</table>")
    html = HTML("".join(html))
    display(html)
    return html.data


def visualize(interpret_res, true_labels=None, words=None):
    """
    interpret_res: List[TokenResult, InterpretResult], Interpretability Results
    true_labels: List[int], Golden labels for test examples
    words: List[List[str]], The word segmentation result of the test examples, the length of words is equal to the attributions
    """
    result_num = len(interpret_res)
    if true_labels is None:
        true_labels = [None] * result_num
    if words is None:
        words = [None] * result_num
    records = []
    for i in range(result_num):
        records.append(VisualizationTextRecord(interpret_res[i], true_label=true_labels[i], words=words[i]))
    html = visualize_text(records)
    return html