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
"""
The code in this file (_lime_base.py) is largely simplified and modified from https://github.com/marcotcr/lime.
"""

from functools import partial
import math

import numpy as np
import paddle
import sklearn
from sklearn.linear_model import Ridge
from sklearn.utils import check_random_state

from ..data_processor import LimeResult
from .base_interpret import TokenInterpreter


class LimeBase(object):
    """
    Class for learning a locally linear sparse model from perturbed data
    """

    def __init__(self,
                 unk_id,
                 pad_id,
                 predict_fn,
                 kernel_width=0.25,
                 kernel=None,
                 verbose=False,
                 random_state=None,
                 batch_size=16,
                 model_regressor='Ridge',
                 distance_metric='cosine'):
        """Init function
        """

        if kernel is None:

            def kernel(d, kernel_width):
                """kernel"""
                return np.sqrt(np.exp(-(d**2) / kernel_width**2))

        self.kernel_fn = partial(kernel, kernel_width=kernel_width)
        self.verbose = verbose
        self.random_state = check_random_state(random_state)
        self.batch_size = batch_size
        self.unk_id = unk_id
        self.pad_id = pad_id
        self.predict_fn = predict_fn
        self.distance_metric = distance_metric
        self.model_regressor = model_regressor

    def interpret_instance(self, instance, interpret_class, num_samples, reg_force=1.0):
        """
        Generates interpretations for a prediction.
        """
        lime_data, lime_probas, lime_distances = self._gen_neighbors(instance, num_samples)
        lime_labels = lime_probas[:, interpret_class]
        lime_weight, lime_score = self._train_lime(lime_data, lime_labels, lime_distances, interpret_class)

        return lime_weight, lime_score

    def _train_lime(self, neighborhood_data, neighborhood_labels, distances, label):

        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels

        if self.model_regressor == 'Ridge':
            linear_model = Ridge(alpha=0, fit_intercept=True, normalize=True, random_state=self.random_state)
        else:
            raise ValueError(f"invalid key '{self.model_regressor}' for model_regressor.")

        linear_model.fit(neighborhood_data, labels_column, sample_weight=weights)
        # R^2 between linear_model.predict(X) and y.
        prediction_score = linear_model.score(neighborhood_data, labels_column, sample_weight=weights)
        # local_pred = linear_model.predict(neighborhood_data[0, used_features].reshape(1, -1))

        return linear_model.coef_, prediction_score

    def _gen_neighbors(self, instance, num_samples):
        word_ids = instance[0]
        word_ids_shape = word_ids.shape
        word_ids = word_ids.reshape((-1, ))

        n_features = len(word_ids) if self.pad_id is None else np.sum(word_ids.numpy() != self.pad_id)

        probas = []
        data = []

        for i in range(math.ceil(num_samples / self.batch_size)):
            if (i + 1) * self.batch_size < num_samples:
                cur_batch_size = self.batch_size
            else:
                cur_batch_size = num_samples - self.batch_size * i

            batch_samples = self.random_state.randint(0, 2, cur_batch_size * n_features).reshape(
                (cur_batch_size, n_features))
            # orignal sample
            if i == 0:
                batch_samples[0] = 1
            mask = paddle.to_tensor(batch_samples)
            mask = paddle.concat(
                [mask, paddle.ones((cur_batch_size, len(word_ids) - n_features), dtype=mask.dtype)], axis=1)
            inputs = list(paddle.concat([field] * cur_batch_size, axis=0) \
                        if isinstance(field, paddle.Tensor) else field for field in instance)
            batch_word_ids = inputs[0]
            inputs[0] = batch_word_ids * mask + self.unk_id * (1 - mask)
            batch_preds, batch_probas = self.predict_fn(tuple(inputs))
            probas.append(batch_probas)
            data.append(batch_samples)
        data = np.concatenate(data, axis=0)
        probas = np.concatenate(probas, axis=0)

        # Calculate the distance bewteen samples and origin sample
        distances = sklearn.metrics.pairwise_distances(data, data[:1], metric=self.distance_metric).ravel()
        return data, probas, distances


class LIMEInterpreter(TokenInterpreter):
    """
    LIME Interpreter for NLP tasks.
    More details regarding the LIME method can be found in the original paper:
    https://arxiv.org/abs/1602.04938
    """

    def __init__(self,
                 paddle_model,
                 unk_id,
                 pad_id=None,
                 device=None,
                 random_seed=None,
                 predict_fn=None,
                 batch_size=50,
                 distance_metric='cosine') -> None:
        """
        Args:
            paddle_model (callable): A model with ``forward`` and possibly ``backward`` functions.
            unk_id (int): The word id to replace occluded words. Typical choices include "", <unk>, and <pad>.
            pad_id (int or None): The word id used to pad the sequences. If None, it means there is no padding. Default: None.
            device (str, optional): The device used for running `paddle_model`, options: ``cpu``, ``gpu``, ``gpu:0``, ``gpu:1`` etc. Default: None. 
            random_seed (int): random seed. Defaults to None.
            predict_fn(callable, optional): If the paddle_model prediction has special process, the user can customize the prediction function.  Default: None.
            batch_size (int, optional): Number of samples to forward each time. Default: 50
            distance_metric (str, optional): he metric to use when calculating distance between instances in a feature array. Default: ``cosine``.
        """

        TokenInterpreter.__init__(self, paddle_model, device)
        self.paddle_model = paddle_model

        self._build_predict_fn(predict_fn=predict_fn)

        # use the default LIME setting
        self.lime_base = LimeBase(random_state=random_seed,
                                  batch_size=batch_size,
                                  unk_id=unk_id,
                                  pad_id=pad_id,
                                  predict_fn=self.predict_fn,
                                  distance_metric=distance_metric)
        self.lime_intermediate_results = {}

    def interpret(self, data, labels=None, num_samples=1000):
        """
        Main function of the interpreter.
        Args:
            data (str): The raw string for analysis.
            preprocess_fn (Callable): A user-defined function that input raw string and outputs the a tuple of inputs to feed into the NLP model.
            num_samples (int, optional): LIME sampling numbers. Larger number of samples usually gives more accurate interpretation. Default: 1000
    
        Returns:
            List[LimeResult]: a list of predicted labels, probabilities, and interpretations.
        """

        if isinstance(data, (tuple, list)):
            bs = data[0].shape[0]
            is_tuple = True
        else:
            bs = data.shape[0]
            is_tuple = False

        assert labels is None or \
                        (isinstance(labels, (list, np.ndarray)) and len(labels) == bs)

        pred_label, pred_proba = self.predict_fn(data)

        rets = []
        for i in range(bs):
            if labels is None:
                interpret_class = pred_label[i]
            else:
                interpret_class = labels[i]
            # input of lime base should be index of word or subword, like input_ids.
            # shape like (bs, seq_len)
            if is_tuple:
                instance = tuple(field[i:i + 1] \
                            if isinstance(field, (list, np.ndarray, paddle.Tensor)) and len(field) == bs else field \
                                for field in data)
            else:
                instance = (data[i:i + 1], )
            # only one example here
            lime_weight, r2_scores = self.lime_base.interpret_instance(instance,
                                                                       num_samples=num_samples,
                                                                       interpret_class=interpret_class)
            limeresult = LimeResult(attributions=lime_weight,
                                    pred_label=pred_label[i],
                                    pred_proba=pred_proba[i],
                                    lime_score=r2_scores)
            rets.append(limeresult)
        return rets

    def _build_predict_fn(self, predict_fn=None):
        if predict_fn is not None:
            self.predict_fn = predict_fn
            return

        def predict_fn(inputs, paddle_model=None):
            """predict_fn"""
            if paddle_model is None:
                paddle_model = self.paddle_model

            if isinstance(inputs, (tuple, list)):
                logits = paddle_model(*inputs)  # get logits, [bs, num_c]
            else:
                logits = paddle_model(inputs)  # get logits, [bs, num_c]
            probas = paddle.nn.functional.softmax(logits, axis=1)  # get probabilities.
            preds = paddle.argmax(probas, axis=1)  # get predictions.

            return preds.numpy(), probas.numpy()

        self.predict_fn = predict_fn
