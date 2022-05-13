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
"""TokenInterpreter"""

import abc

from ..data_processor import InterpretResult
from ..common import merge_subword_special_idx
from ..common import get_word_attributions
from ..common import get_rationales_and_non_ratioanles
from ...base_interpret import Interpreter


class TokenInterpreter(Interpreter):
    """
        Interpreter is the base class for all interpretation algorithms.
    """

    def __init__(self, *args, **akwargs):
        Interpreter.__init__(self, *args, **akwargs)

    def __call__(self, *args, **kwargs):
        return self.interpret(*args, **kwargs)

    @abc.abstractmethod
    def interpret(self, **kwargs):
        """Main function of the interpreter."""
        raise NotImplementedError

    @abc.abstractmethod
    def _build_predict_fn(self, **kwargs):
        """Build self.predict_fn for interpreters."""
        raise NotImplementedError

    def alignment(self,
                  interpret_results,
                  contexts,
                  batch_words,
                  word_offset_maps,
                  subword_offset_maps,
                  special_tokens=[],
                  rationale_num=5):
        """Align the subword's attributions to the word. Return top words with the top ``rationale_num`` as rationale and the other words as non-rationale.
        Args:
            interpret_results ([data_class]): The Interpreter functions ouputs, like ``AttentionResult``, ``LIMEResult`` etc.
            contexts ([str]): The input text with speical_tokens to tokenizer, like ``[CLS] How are you? [SEP]``.
            batch_words ([[str]]): The word segmentation resutls of the contexts.
            word_offset_maps ([(int, int)]): The offset mapping of word segationment.
            subword_offset_maps ([(int, int)]): The offset mapping of subwords.
            special_tokens ([str], optional): The speical tokens which not be extracted as rationales. 
            rationale_num (int, optional): The number of rationales. Default: 5
        Returns:
            List[InterpretResult]:  a list of predicted labels, probabilities, interpretations, rationales etc.
        """

        result = []
        assert len(contexts) == len(batch_words) == len(word_offset_maps) == len(subword_offset_maps) == len(
            interpret_results
        ), f"The lenght of contexts, batch_words, word_offset_maps, subword_offset_maps, interpret_results should be equal."

        for i in range(len(contexts)):
            words = batch_words[i]
            context = contexts[i]
            word_offset_map = word_offset_maps[i]
            subword_offset_map = subword_offset_maps[i]
            interpret_result = interpret_results[i]
            assert subword_offset_map[-1][1] == word_offset_map[-1][
                1], "error offset_map, please check word_offset_maps and subword_offset_maps"

            # merge speical tokens for subword_offset_map
            subword_offset_map = merge_subword_special_idx(words, word_offset_map, subword_offset_map, special_tokens)

            attributions = interpret_result.attributions
            pred_label = interpret_result.pred_label
            pred_proba = interpret_result.pred_proba

            # get word attributions
            word_attributions = get_word_attributions(words, word_offset_map, subword_offset_map, attributions)
            # get ratioanles and non-rationales
            ratioanle_result = get_rationales_and_non_ratioanles(words,
                                                                 word_attributions,
                                                                 special_tokens=special_tokens,
                                                                 rationale_num=rationale_num)
            interpret_result = InterpretResult(words=words,
                                               word_attributions=word_attributions,
                                               pred_label=pred_label,
                                               pred_proba=pred_proba,
                                               rationale=ratioanle_result['rationale_ids'],
                                               non_rationale=ratioanle_result['non_rationale_ids'],
                                               rationale_tokens=ratioanle_result['rationale_tokens'],
                                               non_rationale_tokens=ratioanle_result['non_rationale_tokens'])
            result.append(interpret_result)
        return result