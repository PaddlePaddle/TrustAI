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

from ..common.utils import get_sublayer, dot_similarity, cos_similarity, euc_similarity, get_top_and_bottom_n_examples, get_struct_res
from .example_base_interpreter import ExampleBaseInterpreter


class GradientSimilarityModel(ExampleBaseInterpreter):
    """
    Gradient-based similarity method for NLP tasks.
    """

    def __init__(
        self,
        paddle_model,
        train_dataloader,
        device=None,
        classifier_layer_name="classifier",
        predict_fn=None,
        criterion=None,
        cached_train_grad=None,
    ):
        """
        Initialization.
        Args:
            paddle_model(callable): A model with ``forward``.
            train_dataloader(iterable): Dataloader of model's training data.
            device(str: default=None): Device type, and it should be ``cpu``, ``gpu``, ``gpu:0``, ``gpu:1``  etc.
            classifier_layer_name(str: default=classifier): Name of the classifier layer in paddle_model.
            predict_fn(callabel: default=None): If the paddle_model prediction has special process, user can customize the prediction function.
            criterion(paddle.nn.layer.loss: default=None): criterion to calculate model loss.
            cached_train_grad(str: default=None): Path of the cached train_dataloader gradient. In the first training time, it will take some time to generate the train_grad
        """
        ExampleBaseInterpreter.__init__(self, paddle_model, device, predict_fn, classifier_layer_name)
        self.paddle_model = paddle_model
        self.classifier_layer_name = classifier_layer_name
        self.criterion = (criterion if criterion is not None else paddle.nn.loss.CrossEntropyLoss())
        if cached_train_grad is not None and os.path.isfile(cached_train_grad):
            self.train_grad = paddle.load(cached_train_grad)
        else:
            self.train_grad, *_ = self.get_grad_from_dataloader(train_dataloader)
            if cached_train_grad is not None:
                try:
                    paddle.save(self.train_grad, cached_train_grad)
                except IOError:
                    import sys
                    sys.stderr.write("save cached_train_grad fail")

    def interpret(self, data, sample_num=3, sim_fn="cos"):
        """
        Select most similar and dissimilar examples for a given data using the `sim_fn` metric.
        Args:
            data(iterable): one batch of data to interpret.
            sample_num(int: default=3): the number of positive examples and negtive examples selected for each instance. Return all the training examples ordered by `influence score` if this parameter is -1.
            sim_fn(str: default=cos): the similarity metric to select examples. It should be ``cos`` or ``dot``.
        """
        if sample_num == -1:
            sample_num = len(self.train_grad)
        pos_examples = []
        neg_examples = []
        val_feature, _, preds = self.get_grad(self.paddle_model, data)

        if sim_fn == "dot":
            similarity_fn = dot_similarity
        elif sim_fn == "cos":
            similarity_fn = cos_similarity
        else:
            raise ValueError(f"sim_fn only support ['dot', 'cos'] in gradient simmialrity, but gets `{sim_fn}`")
        for index in range(len(preds)):
            tmp = similarity_fn(self.train_grad, paddle.to_tensor(val_feature[index]))
            pos_idx, neg_idx = get_top_and_bottom_n_examples(tmp, sample_num=sample_num)
            pos_examples.append(pos_idx)
            neg_examples.append(neg_idx)
        preds = preds.tolist()
        res = get_struct_res(preds, pos_examples, neg_examples)
        return res

    def get_grad(self, paddle_model, data):
        """
        get grad from one batch of data.
        """
        if paddle_model.training:
            paddle_model.eval()
        if isinstance(data, (tuple, list)):
            assert len(data[0]) == 1, "batch_size must be 1"
        else:
            assert len(data) == 1, "batch_size must be 1"
        _, prob, pred = self.predict_fn(data)
        loss = self.criterion(prob, paddle.to_tensor(pred))
        loss.backward()
        grad = self._get_flat_param_grad()
        self._clear_all_grad()
        return paddle.to_tensor(grad), paddle.to_tensor(prob), paddle.to_tensor(pred)

    def get_grad_from_dataloader(self, data_loader):
        """
        get grad from data_loader.
        """
        print("Extracting gradient for given dataloader, it will take some time...")
        probas, preds, grads = [], [], []

        for batch in data_loader:
            grad, prob, pred = self.get_grad(self.paddle_model, batch)
            grads.append(grad)
            probas.append(prob)
            preds.append(pred)

        grads = paddle.concat(grads, axis=0)
        probas = paddle.concat(probas, axis=0)
        preds = paddle.concat(preds, axis=0)
        return grads, probas, preds

    def _get_flat_param_grad(self):
        """
        get gradient
        """
        return paddle.concat([
            paddle.flatten(p.grad) for n, p in self.paddle_model.named_parameters() if self.classifier_layer_name in n
        ]).unsqueeze(axis=0)

    def _clear_all_grad(self):
        """
        clear gradient
        """
        for p in self.paddle_model.parameters():
            p.clear_gradient()
