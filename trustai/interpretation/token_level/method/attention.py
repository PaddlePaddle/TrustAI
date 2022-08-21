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
"""attention interpreter"""

import paddle

from ..data_processor import AttentionResult
from .base_interpret import TokenInterpreter


class AttentionInterpreter(TokenInterpreter):
    """
    Attention Interpreter for NLP tasks.
    """

    def __init__(self, paddle_model, device=None, attention_name=None, predict_fn=None) -> None:
        """
        Args:
            paddle_model (callable): A model with ``forward`` and possibly ``backward`` functions.
            device (str, optional): The device used for running `paddle_model`, options: ``cpu``, ``gpu``, ``gpu:0``, ``gpu:1`` etc. Default: None. 
            attention_name(str, optional): The layer name of attention. The correct name of embedding can be found through ``print(model)``. Default: None.  
            predict_fn(callable, optional): If the paddle_model prediction has special process, the user can customize the prediction function.  Default: None.
        """
        TokenInterpreter.__init__(self, paddle_model, device)

        # build predict function
        self._build_predict_fn(attention_name=attention_name, predict_fn=predict_fn)

    def interpret(self, data):
        """Main function of the interpreter.
        Args:
            data ([type]): The inputs of the paddle_model.
            
        Returns:
            List[AttentionResult]: a list of predicted labels, probabilities, and interpretations.
        """

        if isinstance(data, (tuple, list)):
            bs = data[0].shape[0]
        else:
            bs = data.shape[0]

        attributions, pred_label, pred_proba = self._attention_interpret(data)

        # returns
        rets = []
        for i in range(bs):
            attresult = AttentionResult(attributions=attributions[i],
                                        pred_label=pred_label[i],
                                        pred_proba=pred_proba[i])
            rets.append(attresult)
        return rets

    def _build_predict_fn(self, attention_name=None, predict_fn=None):
        assert attention_name is not None or \
            predict_fn is not None, "At least One of attention_name and predict_fn is not None."

        if attention_name is None:
            self.predict_fn = predict_fn
            return

        def predict_fn(inputs, paddle_model=None):
            if paddle_model is None:
                paddle_model = self.paddle_model
            target_feature_map = []

            def hook(layer, input, output):
                target_feature_map.append(output)
                return output

            hooks = []
            for name, v in paddle_model.named_sublayers():
                if attention_name in name:
                    h = v.register_forward_post_hook(hook)
                    hooks.append(h)

            if isinstance(inputs, (tuple, list)):
                logits = paddle_model(*inputs)  # get logits, [bs, num_c]
            else:
                logits = paddle_model(inputs)  # get logits, [bs, num_c]

            bs = logits.shape[0]
            for h in hooks:
                h.remove()

            probas = paddle.nn.functional.softmax(logits, axis=1)  # get probabilities.
            preds = paddle.argmax(probas, axis=1)  # get predictions.
            # logits or probas
            preds = preds.reshape((bs, ))
            attention = target_feature_map[0].sum(1)[:, 0]
            return attention.numpy(), preds.numpy(), probas.numpy()

        self.predict_fn = predict_fn

    def _attention_interpret(self, data) -> tuple:
        attentions, labels, probas = self.predict_fn(data, paddle_model=self.paddle_model)
        return attentions, labels, probas
