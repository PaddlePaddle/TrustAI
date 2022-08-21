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
"""integrated interpreter"""

from functools import partial

import numpy as np
import paddle

from ..data_processor import IGResult
from .base_interpret import TokenInterpreter


class IntGradInterpreter(TokenInterpreter):
    """
    Integrated Gradients Interpreter for NLP tasks.
    More details regarding the Integrated Gradients method can be found in the original paper:
    https://arxiv.org/abs/1703.01365
    """

    def __init__(self,
                 paddle_model,
                 device=None,
                 embedding_name='word_embeddings',
                 batch_size=16,
                 predict_fn=None) -> None:
        """
        Args:
            paddle_model (callable): A model with ``forward`` and possibly ``backward`` functions.
            device (str, optional): The device used for running `paddle_model`, options: ``cpu``, ``gpu``, ``gpu:0``, ``gpu:1`` etc. Default: None. 
            embedding_name (str, optional): name of the embedding layer at which the steps will be applied. 
                Defaults to 'word_embeddings'. The correct name of embedding can be found through ``print(model)``.
            batch_size(int, optional): Number of samples to forward each time in integrated gradient interpretation. Default: 16.
            predict_fn(callable, optional): If the paddle_model prediction has special process, the user can customize the prediction function.  Default: None.
        """
        TokenInterpreter.__init__(self, paddle_model, device)

        # build predict function
        self._build_predict_fn(predict_fn=predict_fn)

        # batch size for single instance in integrated Gradients
        self.batch_size = batch_size
        self.embedding_name = embedding_name

    def interpret(self, data, labels=None, steps=1000):
        """Main function of the interpreter.
        Args:
            data ([type]): The inputs of the paddle_model.
            labels ([type], optional): The target label to analyze. If None, the most likely label will be used. Default: None.
            steps (int, optional): number of steps in the Riemman approximation of the integral. Default: 1000
            
        Returns:
            List[IGResult]: a list of predicted labels, probabilities and interpretations.

        """
        if isinstance(data, (tuple, list)):
            bs = data[0].shape[0]
            is_tuple = True
        else:
            bs = data.shape[0]
            is_tuple = False

        assert labels is None or \
                        (isinstance(labels, (list, np.ndarray)) and len(labels) == bs)
        rets = []
        for i in range(bs):
            if is_tuple:
                instance = tuple(field[i:i + 1] \
                            if isinstance(field, (list, np.ndarray, paddle.Tensor)) and len(field) == bs else field \
                                for field in data)
            else:
                instance = (data[i:i + 1], )
            label = None if labels is None else labels[i:i + 1]
            attributions, error_percent, pred_label, pred_proba = self._ig_interpret_instance(instance, label, steps)
            igresult = IGResult(attributions=attributions,
                                pred_label=pred_label,
                                pred_proba=pred_proba,
                                error_percent=error_percent)
            rets.append(igresult)
        return rets

    def _build_predict_fn(self, predict_fn=None):

        if predict_fn is not None:
            self.predict_fn = predict_fn
            return

        def predict_fn(inputs, label, left=None, right=None, steps=None, paddle_model=None, embedding_name=None):
            if paddle_model is None:
                paddle_model = self.paddle_model
            target_feature_map = []

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

            if isinstance(inputs, (tuple, list)):
                logits = paddle_model(*inputs)  # get logits, [bs, num_c]
            else:
                logits = paddle_model(inputs)  # get logits, [bs, num_c]

            bs = logits.shape[0]
            for h in hooks:
                h.remove()

            probas = paddle.nn.functional.softmax(logits, axis=1)  # get probabilities.
            preds = paddle.argmax(probas, axis=1)  # get predictions.
            if label is None:
                assert len(preds) == 1
                label = preds.numpy()[:1]  # label is an integer.

            labels = np.concatenate([label] * bs)

            # logits or probas
            labels = labels.reshape((bs, ))
            labels_onehot = paddle.nn.functional.one_hot(paddle.to_tensor(labels), num_classes=probas.shape[1])
            loss = paddle.sum(probas * labels_onehot, axis=1)
            loss.backward()

            gradients = target_feature_map[0].grad  # get gradients of "embedding".
            loss.clear_gradient()

            if isinstance(gradients, paddle.Tensor):
                gradients = gradients.numpy()
            return gradients, labels, target_feature_map[0].numpy(), probas.numpy()

        self.predict_fn = predict_fn

    def _ig_interpret_instance(self, instance, label, steps) -> tuple:
        end = steps
        bs = self.batch_size
        baseline_embedding = None
        target_embedding = None
        if label is None:
            _, label, _, _ = self.predict_fn(instance,
                                             label,
                                             paddle_model=self.paddle_model,
                                             embedding_name=self.embedding_name)
        total_gradients = []
        for i in range(end // bs + 1):
            left = bs * i
            right = min(end + 1, left + bs)
            cur_bs = right - left
            inputs = tuple(
                paddle.concat([field] * cur_bs, axis=0) if isinstance(field, paddle.Tensor) else field
                for field in instance)

            batch_gradients, _, batch_embedding, batch_probas = self.predict_fn(inputs,
                                                                                label,
                                                                                left,
                                                                                right,
                                                                                steps,
                                                                                paddle_model=self.paddle_model,
                                                                                embedding_name=self.embedding_name)
            total_gradients.append(batch_gradients)
            if left == 0:
                baseline_embedding = batch_embedding[0]
                baseline_proba = batch_probas[0]
            if right == end + 1:
                pred_embedding = batch_embedding[-1]
                pred_proba = batch_probas[-1]

        total_gradients = np.concatenate(total_gradients, axis=0)
        trapezoidal_gradients = (total_gradients[1:] + total_gradients[:-1]) / 2
        integral = trapezoidal_gradients.mean(0)
        ig_gradients = (pred_embedding - baseline_embedding) * integral
        ig_gradients = np.sum(ig_gradients, axis=-1)

        error_percentage = None
        if isinstance(pred_proba, np.ndarray):
            sum_attributions = np.sum(ig_gradients)
            delta_proba = pred_proba - baseline_proba
            error_percentage = 100 * (delta_proba - sum_attributions) / delta_proba
            if isinstance(label[0], int):
                error_percentage = error_percentage.reshape(-1)[label[0]]

        return ig_gradients, error_percentage, label[0], pred_proba
