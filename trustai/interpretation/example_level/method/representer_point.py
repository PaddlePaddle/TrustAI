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
"""representer point"""

import logging
import functools

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from scipy.stats.stats import pearsonr

from ...base_interpret import Interpreter
from ..common.utils import get_sublayer


class SoftmaxClassifier(nn.Layer):
    """
    Softmax classifier with cross-entropy loss.
    """

    def __init__(self, in_feature, out_feature, params):
        """
        Initialization.
        """
        super().__init__()
        self.linear = paddle.nn.Linear(in_feature, out_feature, bias_attr=False)
        self.linear.weight.set_value(params)

    def forward(self, features, labels):
        """
        Calculate loss for the loss function and L2 regularizer.
        """
        logits = self.linear(features)
        logits_max = paddle.max(logits, axis=1, keepdim=True)
        logits = logits - logits_max
        A = paddle.log(paddle.sum(paddle.exp(logits), axis=1))
        B = paddle.sum(logits * labels, axis=1)
        loss = paddle.sum(A - B)
        l2 = paddle.sum(paddle.square(self.linear.weight))
        return (loss, l2)


class RepresenterPointBase(nn.Layer):
    """
    Class for learning a representer point model
    """

    def __init__(self,
                 paddle_model,
                 optimizer_name='SGD',
                 classifier_layer_name='classifier',
                 learning_rate=5e-2,
                 lmbd=0.03,
                 epochs=40000,
                 correlation=True):
        """
        Initialization
        """
        super().__init__()
        weight, params = self._get_params(paddle_model, classifier_layer_name)
        self.softmax_classifier = SoftmaxClassifier(weight.shape[0] + 1, weight.shape[1], params)
        self.learning_rate = learning_rate
        self.lmbd = lmbd
        self.epochs = epochs
        self.optimizer = getattr(paddle.optimizer, optimizer_name)
        self.correlation = correlation

    def _get_params(self, paddle_model, classifier_layer_name):
        """
        Get the parameters of classifier_layer in model.
        """
        classifier = get_sublayer(paddle_model, classifier_layer_name)
        weight, bias = classifier.weight, classifier.bias
        params = paddle.concat([weight, paddle.unsqueeze(bias, axis=0)], axis=0)
        return weight, params

    def train(self, input_feature, input_logits):
        """
        Train a representer point model.
        """
        # input_feature is the feature of a given model, input_logits is the logits of input_feature
        input_feature = paddle.concat(
            [input_feature, paddle.ones((input_feature.shape[0], 1), dtype=input_feature.dtype)], axis=1)

        input_num = len(input_logits)
        min_loss = float('inf')
        optimizer = self.optimizer(learning_rate=self.learning_rate,
                                   parameters=self.softmax_classifier.linear.parameters())
        print('Training representer point model, it will take several minutes...')
        for epoch in range(self.epochs):
            classifier_loss, L2 = self.softmax_classifier(input_feature, input_logits)
            loss = L2 * self.lmbd + classifier_loss / input_num
            classifier_mean_loss = classifier_loss / input_num
            loss.backward()
            grad_loss = paddle.mean(paddle.abs(self.softmax_classifier.linear.weight.grad)).numpy()
            # save the W with the lowest grad_loss
            if grad_loss < min_loss:
                if epoch == 0:
                    init_grad = grad_loss
                min_loss = grad_loss
                best_W = self.softmax_classifier.linear.weight
                if min_loss < init_grad / 200:
                    logging.info(f'stopping criteria reached in epoch:{epoch}')
                    optimizer.clear_grad()
                    break
            optimizer.step()
            optimizer.clear_grad()

            if epoch % 1000 == 0:
                logging.info(
                    f'Eopch:{epoch:4d}\tloss:{loss.numpy()}\tphi_loss:{classifier_mean_loss.numpy()}\tgrad:{grad_loss}')

        # caluculate w based on the representer theorem's decomposition
        logits = paddle.matmul(input_feature, best_W)
        logits_max = paddle.max(logits, axis=1, keepdim=True)
        logits = logits - logits_max  # avoids numerical overflow
        softmax_value = F.softmax(logits)

        # derivative of softmax cross entropy
        weight_matrix = softmax_value - input_logits
        weight_matrix = weight_matrix / (-2.0 * self.lmbd * input_num)  # alpha

        best_w = paddle.matmul(paddle.t(input_feature), weight_matrix)  # alpha * f_i^T

        if self.correlation:
            # calculate y_p, which is the prediction based on decomposition of w by representer theorem
            logits = paddle.matmul(input_feature, best_w)  # alpha * f_i^T * f_t
            logits_max = paddle.max(logits, axis=1, keepdim=True)
            logits = logits - logits_max
            y_p = F.softmax(logits)

            print('L1 difference between ground truth prediction and prediction by representer theorem decomposition')
            print(F.l1_loss(input_logits, y_p).numpy())

            print('pearson correlation between ground truth  prediction and prediciton by representer theorem')
            corr, _ = (pearsonr(input_logits.flatten().numpy(), (y_p).flatten().numpy()))
            print(corr)
        return weight_matrix, best_w


class RepresenterPointModel(Interpreter):
    """
    Representer Point Model for NLP tasks.
    More details regarding the representer point method can be found in the original paper:
    https://proceedings.neurips.cc/paper/2018/file/8a7129b8f3edd95b7d969dfc2c8e9d9d-Paper.pdf
    """

    def __init__(self,
                 paddle_model,
                 train_dataloader,
                 device='gpu',
                 classifier_layer_name='classifier',
                 predict_fn=None,
                 learning_rate=5e-2,
                 lmbd=0.03,
                 epochs=40000):
        """
        Initialization.
        Args:
            paddle_model(callable): A model with ``forward``.
            train_dataloader(iterable): Dataloader of model's training data.
            device(str: default=gpu): Device type, and it should be ``gpu``, ``cpu`` etc.
            classifier_layer_name(str: default=classifier): Name of the classifier layer in paddle_model.
            predict_fn(callabel: default=None): If the paddle_model prediction has special process, user can customize the prediction function.
            learning_rate(float: default=5e-2): Learning rate.
            lmbd(float: default=0.03): The coefficient of l2 regularization.
            epochs(int: default=4000): The total epochs to trianing representer point model.
        """
        Interpreter.__init__(self, paddle_model, device)
        self.paddle_model = paddle_model
        self._build_predict_fn(predict_fn=predict_fn)
        self.classifier_layer_name = classifier_layer_name
        self.represerter_point = RepresenterPointBase(paddle_model,
                                                      classifier_layer_name=classifier_layer_name,
                                                      learning_rate=learning_rate,
                                                      lmbd=lmbd,
                                                      epochs=epochs)
        self.train_feature, self.train_logits, _ = self.extract_featue(paddle_model, train_dataloader)
        self.weight_matrix, self.best_W = self.represerter_point.train(self.train_feature, self.train_logits)

    def interpret(self, data, sample_num=3):
        """
        Select postive and negtive examples for a given data.
        Args:
            data(iterable): Dataloader to interpret.
            sample_sum(int: default=3): the number of positive examples and negtive examples selected for each instance.
        """
        pos_examples = []
        neg_examples = []
        val_feature, _, results = self.extract_featue(self.paddle_model, data)
        for index, target_class in enumerate(results):
            tmp = self.weight_matrix[:, target_class] * paddle.sum(
                self.train_feature * paddle.to_tensor(val_feature[index]), axis=1)
            idx = paddle.flip(paddle.argsort(tmp), axis=0)
            pos_idx = idx[:sample_num].tolist()
            neg_idx = idx[-sample_num:].tolist()
            pos_examples.append(pos_idx)
            neg_examples.append(neg_idx)
        return results.tolist(), pos_examples, neg_examples

    def _build_predict_fn(self, predict_fn=None):
        if predict_fn is not None:
            self.predict_fn = functools.partial(predict_fn, paddle_model=self.paddle_model)
            return

        def predict_fn(inputs, paddle_model=None):
            """predict_fn"""
            if paddle_model is None:
                paddle_model = self.paddle_model

            x_feature = []

            def forward_pre_hook(layer, input):
                """
                Hook for a given layer in model.
                """
                x_feature.extend(input[0])

            classifier = get_sublayer(paddle_model, self.classifier_layer_name)

            forward_pre_hook_handle = classifier.register_forward_pre_hook(forward_pre_hook)

            if isinstance(inputs, (tuple, list)):
                logits = paddle_model(*inputs)  # get logits, [bs, num_c]
            else:
                logits = paddle_model(inputs)  # get logits, [bs, num_c]

            forward_pre_hook_handle.remove()

            probas = paddle.nn.functional.softmax(logits, axis=1)  # get probabilities.
            preds = paddle.argmax(probas, axis=1).tolist()  # get predictions.
            x_feature = paddle.to_tensor(x_feature)
            return x_feature, probas, preds

        self.predict_fn = predict_fn

    @paddle.no_grad()
    def extract_featue(self, paddle_model, data_loader):
        print('Extracting feature for dataloader, it will take some time...')
        x_features, y_logits, y_preds = [], [], []

        for step, batch in enumerate(data_loader, start=1):
            x_feature, prob, pred = self.predict_fn(batch)
            x_features.extend(x_feature)
            y_logits.extend(prob)
            y_preds.extend(pred)
        x_features = paddle.to_tensor(x_features)
        y_logits = paddle.to_tensor(y_logits)
        y_preds = paddle.to_tensor(y_preds)
        return x_features, y_logits, y_preds
