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
"""GradShapInterpreter"""

from ..data_processor import GradShapResult
from .base_interpret import TokenInterpreter


class GradShapInterpreter(TokenInterpreter):
    """A wrap class of interpretdl.GradShapInterpreter,  please refer to ``interpretdl/interpreter/gradient_shap.py`` for details"""

    def __init__(self,
                 paddle_model,
                 device='gpu',
                 n_samples=5,
                 noise_amount=0.1,
                 embedding_name="word_embeddings") -> None:
        """
        Args:
            paddle_model (callable): A model with ``forward`` and possibly ``backward`` functions.
            device (str, optional): The device used for running `paddle_model`, options: ``cpu``, ``gpu``. Default: gpu. 
            n_samples (int, optional): [description]. Defaults to 5.
            noise_amount (float, optional): Noise level of added noise to the embeddings. 
                The std of Guassian random noise is ``noise_amount * embedding.mean() * (x_max - x_min)``. Default: 0.1
            embedding_name (str, optional): name of the embedding layer at which the noises will be applied. 
                Defaults to 'word_embeddings'. The correct name of embedding can be found through ``print(model)``.
        """
        TokenInterpreter.__init__(self, paddle_model, device)

        # build predict function
        self.gradshap = self._build_predict_fn(paddle_model, device)

        self.n_samples = n_samples
        self.noise_amount = noise_amount
        self.embedding_name = embedding_name

    def interpret(self, data):
        """Main function of the interpreter.
        Args:
            data ([type]): The inputs of the paddle_model.
            labels ([type], optional): The target label to analyze. If None, the most likely label will be used. Default: None.
        Returns:
            List[GradShapResult]: a list of predicted labels, probabilities and interpretations.
        """

        if isinstance(data, (tuple, list)):
            bs = data[0].shape[0]
        else:
            bs = data.shape[0]

        pred_label, pred_proba, attributions = self.gradshap.interpret(data,
                                                                       n_samples=self.n_samples,
                                                                       noise_amount=self.noise_amount,
                                                                       embedding_name=self.embedding_name,
                                                                       return_pred=True)
        # returns
        rets = []
        for i in range(bs):
            shapresult = GradShapResult(attributions=attributions[i],
                                        pred_label=pred_label[i],
                                        pred_proba=pred_proba[i])
            rets.append(shapresult)
        return rets

    def _build_predict_fn(self, paddle_model, device='gpu'):
        try:
            from interpretdl import GradShapNLPInterpreter
        except ImportError as e:
            import sys
            sys.stderr.write(
                '''Warning with import interpretdl: please install interpretdl firstly. cmd: pip install -U interpretdl'''
            )
            raise e

        return GradShapNLPInterpreter(paddle_model, device)
