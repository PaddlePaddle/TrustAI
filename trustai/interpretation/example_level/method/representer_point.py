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

from ...base_interpret import Interpreter
from .example_base_interpreter import ExampleBaseInterpreter
from ..common.utils import get_sublayer, get_struct_res, get_top_and_bottom_n_examples


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

    def forward(self, features, probas):
        """
        Calculate loss for the loss function and L2 regularizer.
        """
        logits = self.linear(features)
        logits_max = paddle.max(logits, axis=1, keepdim=True)
        logits = logits - logits_max
        A = paddle.log(paddle.sum(paddle.exp(logits), axis=1))
        B = paddle.sum(logits * probas, axis=1)
        loss = paddle.sum(A - B)
        l2 = paddle.sum(paddle.square(self.linear.weight))
        return (loss, l2)


class RepresenterPointBase(nn.Layer):
    """
    Class for learning a representer point model
    """

    def __init__(
        self,
        paddle_model,
        optimizer_name="SGD",
        classifier_layer_name="classifier",
        learning_rate=5e-2,
        lmbd=0.03,
        epochs=40000,
        correlation=True,
    ):
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

    def train(self, input_feature, input_probas):
        """
        Train a representer point model.
        """
        # input_feature is the feature of a given model, input_probas is the probabilities of input_feature
        input_feature = paddle.concat(
            [
                input_feature,
                paddle.ones((input_feature.shape[0], 1), dtype=input_feature.dtype),
            ],
            axis=1,
        )

        input_num = len(input_probas)
        min_loss = float("inf")
        optimizer = self.optimizer(
            learning_rate=self.learning_rate,
            parameters=self.softmax_classifier.linear.parameters(),
        )
        print("Training representer point model, it will take several minutes...")
        for epoch in range(self.epochs):
            classifier_loss, L2 = self.softmax_classifier(input_feature, input_probas)
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
                    logging.info(f"stopping criteria reached in epoch:{epoch}")
                    optimizer.clear_grad()
                    break
            optimizer.step()
            optimizer.clear_grad()

            if epoch % 1000 == 0:
                logging.info(
                    f"Eopch:{epoch:4d}\tloss:{loss.numpy()}\tphi_loss:{classifier_mean_loss.numpy()}\tgrad:{grad_loss}")

        # caluculate w based on the representer theorem's decomposition
        logits = paddle.matmul(input_feature, best_W)
        logits_max = paddle.max(logits, axis=1, keepdim=True)
        logits = logits - logits_max  # avoids numerical overflow
        softmax_value = F.softmax(logits)

        # derivative of softmax cross entropy
        weight_matrix = softmax_value - input_probas
        weight_matrix = weight_matrix / (-2.0 * self.lmbd * input_num)  # alpha

        if self.correlation:
            try:
                from scipy.stats.stats import pearsonr
            except ImportError as e:
                import sys
                sys.stderr.write(
                    '''Info about import scipy: please install scipy firstly. cmd: pip install scipy. We need to calculate the pearsonr correlation between the representre point model and the gived model'''
                )
                return weight_matrix
            best_w = paddle.matmul(paddle.t(input_feature), weight_matrix)  # alpha * f_i^T
            # calculate y_p, which is the prediction based on decomposition of w by representer theorem
            logits = paddle.matmul(input_feature, best_w)  # alpha * f_i^T * f_t
            logits_max = paddle.max(logits, axis=1, keepdim=True)
            logits = logits - logits_max
            y_p = F.softmax(logits)

            print("L1 difference between ground truth prediction and prediction by representer theorem decomposition")
            print(F.l1_loss(input_probas, y_p).numpy())

            print("pearson correlation between ground truth  prediction and prediciton by representer theorem")
            corr, _ = pearsonr(input_probas.flatten().numpy(), (y_p).flatten().numpy())
            print(corr)
        return weight_matrix


class RepresenterPointModel(ExampleBaseInterpreter):
    """
    Representer Point Model for NLP tasks.
    More details regarding the representer point method can be found in the original paper:
    https://proceedings.neurips.cc/paper/2018/file/8a7129b8f3edd95b7d969dfc2c8e9d9d-Paper.pdf
    """

    def __init__(
        self,
        paddle_model,
        train_dataloader,
        device=None,
        classifier_layer_name="classifier",
        predict_fn=None,
        learning_rate=5e-2,
        lmbd=0.03,
        epochs=40000,
    ):
        """
        Initialization.
        Args:
            paddle_model(callable): A model with ``forward``.
            train_dataloader(iterable): Dataloader of model's training data.
            device(str: default=None): Device type, and it should be ``cpu``, ``gpu``, ``gpu:0``, ``gpu:1``  etc.
            classifier_layer_name(str: default=classifier): Name of the classifier layer in paddle_model.
            predict_fn(callabel: default=None): If the paddle_model prediction has special process, user can customize the prediction function.
            learning_rate(float: default=5e-2): Learning rate.
            lmbd(float: default=0.03): The coefficient of l2 regularization.
            epochs(int: default=4000): The total epochs to trianing representer point model.
        """
        ExampleBaseInterpreter.__init__(self, paddle_model, device, predict_fn, classifier_layer_name)
        self.paddle_model = paddle_model
        self.classifier_layer_name = classifier_layer_name
        self.represerter_point = RepresenterPointBase(
            paddle_model,
            classifier_layer_name=classifier_layer_name,
            learning_rate=learning_rate,
            lmbd=lmbd,
            epochs=epochs,
        )
        self.train_feature, self.train_probas, _ = self.extract_feature_from_dataloader(train_dataloader)
        self.weight_matrix = self.represerter_point.train(self.train_feature, self.train_probas)

    def interpret(self, data, sample_num=3):
        """
        Select postive and negtive examples for a given data.
        Args:
            data(iterable): one batch of data to interpret.
            sample_num(int: default=3): the number of positive examples and negtive examples selected for each instance. Return all the training examples ordered by `influence score` if this parameter is -1.
        """
        if sample_num == -1:
            sample_num = len(self.train_feature)
        pos_examples = []
        neg_examples = []
        val_feature, _, preds = self.extract_feature(self.paddle_model, data)
        for index, target_class in enumerate(preds):
            tmp = self.weight_matrix[:, target_class] * paddle.sum(
                self.train_feature * paddle.to_tensor(val_feature[index]), axis=1)
            pos_idx, neg_idx = get_top_and_bottom_n_examples(tmp, sample_num=sample_num)
            pos_examples.append(pos_idx)
            neg_examples.append(neg_idx)
        preds = preds.tolist()
        res = get_struct_res(preds, pos_examples, neg_examples)
        return res

    @paddle.no_grad()
    def extract_feature(self, paddle_model, data):
        """        
        extract feature from one batch of data.
        """
        if self.paddle_model.training:
            self.paddle_model.eval()
        feature, prob, pred = self.predict_fn(data)
        return paddle.to_tensor(feature), paddle.to_tensor(prob), paddle.to_tensor(pred)

    def extract_feature_from_dataloader(self, dataloader):
        """
        extract feature from data_loader.
        """
        print("Extracting feature from given dataloader, it will take some time...")
        features, probas, preds = [], [], []

        for batch in dataloader:
            feature, prob, pred = self.extract_feature(self.paddle_model, batch)
            features.append(feature)
            probas.append(prob)
            preds.append(pred)
        return paddle.concat(features, axis=0), paddle.concat(probas, axis=0), paddle.concat(preds, axis=0)