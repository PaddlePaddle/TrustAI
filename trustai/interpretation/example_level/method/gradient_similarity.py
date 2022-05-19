# !/usr/bin/env python3
"""
gradient-based similarity method.
cosine and dot.
"""
import os
import functools
import warnings

import paddle
import paddle.nn.functional as F

from ..common.utils import get_sublayer
from .example_base_interpreter import ExampleBaseInterpreter


class GradientSimilarityModel(ExampleBaseInterpreter):
    """
    Gradient-based similarity method for NLP tasks.
    """

    def __init__(
        self,
        paddle_model,
        train_dataloader,
        device="gpu",
        classifier_layer_name="classifier",
        predict_fn=None,
        criterion=None,
        cached_train_grad="./cached_train_grad.tensor",
    ):
        """
        Initialization.
        Args:
            paddle_model(callable): A model with ``forward``.
            train_dataloader(iterable): Dataloader of model's training data.
            device(str: default=gpu): Device type, and it should be ``gpu``, ``cpu`` etc.
            classifier_layer_name(str: default=classifier): Name of the classifier layer in paddle_model.
            predict_fn(callabel: default=None): If the paddle_model prediction has special process, user can customize the prediction function.
            criterion(paddle.nn.layer.loss: default=None): criterion to calculate model loss.
            cached_train_grad(str: default="./cached_train_grad.tensor"): path of the cached train_dataloader gradient.
        """
        ExampleBaseInterpreter.__init__(self, paddle_model, device, predict_fn, classifier_layer_name)
        self.paddle_model = paddle_model
        self.classifier_layer_name = classifier_layer_name
        self.criterion = (criterion if criterion is not None else paddle.nn.loss.CrossEntropyLoss())
        if os.path.exists(cached_train_grad) and os.path.isfile(cached_train_grad):
            self.train_grad = paddle.load(cached_train_grad)
        else:
            self.train_grad, *_ = self.get_grad(paddle_model, train_dataloader)
            try:
                paddle.save(self.train_grad, cached_train_grad)
            except IOError as e:
                import sys
                sys.stderr.write("save cached_train_grad fail")

    def interpret(self, data, sample_num=3, sim_fn="dot"):
        """
        Select most similar and dissimilar examples for a given data using the `sim_fn` metric.
        Args:
            data(iterable): Dataloader to interpret.
            sample_sum(int: default=3): the number of positive examples and negtive examples selected for each instance.
            sim_fn(str: default=dot): the similarity metric to select examples.
        """
        examples = []
        val_feature, _, preds = self.get_grad(self.paddle_model, data)

        if sim_fn == "dot":
            similarity_fn = self._dot_similarity
        elif sim_fn == "cos":
            similarity_fn = self._cos_similarity
        else:
            warnings.warn("only support ['dot', 'cos']")
            exit()
        for index, target_class in enumerate(preds):
            tmp = similarity_fn(val_feature[index])
            example_index = self._get_similarity_index(tmp, sample_num=sample_num)
            examples.append(example_index)
        return preds.tolist(), examples

    def _get_similarity_index(self, scores, sample_num=3):
        """
        get index of the most similarity examples
        """
        index = paddle.flip(paddle.argsort(scores), axis=0)
        sim_index = index[:sample_num].tolist()
        dis_sim_index = index[-sample_num:].tolist()
        return sim_index, dis_sim_index

    def _dot_similarity(self, inputs):
        return paddle.sum(self.train_grad * paddle.to_tensor(inputs), axis=1)

    def _cos_similarity(self, inputs):
        return F.cosine_similarity(self.train_grad, paddle.to_tensor(inputs).unsqueeze(0))

    def _euc_similarity(self, inputs):
        return -paddle.linalg.norm(self.train_grad - paddle.to_tensor(inputs).unsqueeze(0), axis=-1).squeeze(-1)

    def get_grad(self, paddle_model, data_loader):
        """
        get grad for data_loader.
        """
        paddle_model.eval()
        print("Extracting gradient for given dataloader, it will take some time...")
        features, probas, preds, grads = [], [], [], []

        for step, batch in enumerate(data_loader, start=1):
            *input, label = batch
            _, prob, pred = self.predict_fn(input)
            loss = self.criterion(prob, label)
            loss.backward()
            grad = self._get_flat_param_grad()
            grads.append(grad)
            self._clear_all_grad()

            probas.extend(prob)
            preds.extend(pred)

        return (
            paddle.to_tensor(grads),
            paddle.to_tensor(probas),
            paddle.to_tensor(preds),
        )

    def _get_flat_param_grad(self):
        """
        get gradient
        """
        return paddle.concat([
            paddle.flatten(p.grad) for n, p in self.paddle_model.named_parameters() if self.classifier_layer_name in n
        ])

    def _clear_all_grad(self):
        """
        clear gradient
        """
        for p in self.paddle_model.parameters():
            p.clear_gradient()
