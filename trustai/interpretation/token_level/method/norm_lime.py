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
"""norm lime"""

from ..data_processor import NormLIMEResult
from .base_interpret import TokenInterpreter


class NormLIMEInterpreter(TokenInterpreter):
    """A wrap class of interpretdl.NormLIMENLPInterpreter,  please refer to ``interpretdl/interpreter/_normlime_base.py`` for details"""

    def __init__(self, paddle_model, preprocess_fn, unk_id, pad_id=None, device=None, batch_size=50) -> None:
        """
        Args:
            paddle_model (callable): A model with ``forward`` and possibly ``backward`` functions.
            preprocess_fn (Callable): A user-defined function that input raw string and outputs the a tuple of inputs to feed into the NLP model.
            unk_id (int): The word id to replace occluded words. Typical choices include "", <unk>, and <pad>.
            pad_id (int or None): The word id used to pad the sequences. If None, it means there is no padding. Default: None.
            device (str, optional): The device used for running `paddle_model`, options: ``cpu``, ``gpu``, ``gpu:0``, ``gpu:1`` etc. Default: None. 
            batch_size (int, optional): Number of samples to forward each time. Default: 50
        """
        TokenInterpreter.__init__(self, paddle_model, device)

        # build predict function
        self.normlime = self._build_predict_fn(paddle_model, device)

        self.batch_size = batch_size
        self.preprocess_fn = preprocess_fn
        self.unk_id = unk_id
        self.pad_id = pad_id

    def interpret(self, data, num_samples=500, temp_data_file='all_lime_weights.npz', save_path='normlime_weights.npy'):
        """Main function of the interpreter.
        Args:
            data ([type]): The inputs of the paddle_model.
            labels ([type], optional): The target label to analyze. If None, the most likely label will be used. Default: None.
            num_samples (int, optional): LIME sampling numbers. Larger number of samples usually gives more accurate interpretation. Default: 1000
            temp_data_file (str, optinal): The .npz file to save/load the dictionary where key is word ids joined by '-' and value is another dictionary with lime weights. Default: 'all_lime_weights.npz'
            save_path (str, optional): The .npy path to save the normlime weights. It is a dictionary where the key is label and value is segmentation ids with their importance. Default: 'normlime_weights.npy'

        Returns:
            [NormLIMEResult] NormLIME weights: {label_i: weights on features}

        """

        normlime_weights = self.normlime.interpret(data,
                                                   self.preprocess_fn,
                                                   unk_id=self.unk_id,
                                                   pad_id=self.pad_id,
                                                   num_samples=num_samples,
                                                   batch_size=self.batch_size,
                                                   temp_data_file=temp_data_file,
                                                   save_path=save_path)

        normresult = NormLIMEResult(attributions=normlime_weights)
        return normresult

    def _build_predict_fn(self, paddle_model, device='gpu'):
        try:
            from interpretdl import NormLIMENLPInterpreter
        except ImportError as e:
            import sys
            sys.stderr.write(
                '''Warning with import interpretdl: please install interpretdl firstly. cmd: pip install -U interpretdl'''
            )
            raise e
        return NormLIMENLPInterpreter(paddle_model, device)
