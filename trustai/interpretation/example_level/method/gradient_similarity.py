# !/usr/bin/env python3
"""
gradient-based similarity method.
cosine and dot.
"""
import functools

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
        cached_train_grad=None,
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
            cached_train_grad(str: default=None): path of the cached train_dataloader gradient.
        """
        ExampleBaseInterpreter.__init__(
            self, paddle_model, device, predict_fn, classifier_layer_name
        )
        self.paddle_model = paddle_model
        self.classifier_layer_name = classifier_layer_name
        self.criterion = (
            criterion if criterion is not None else paddle.nn.loss.CrossEntropyLoss()
        )
        if cached_train_grad is not None:
            self.train_feature = paddle.load(cached_train_grad)
        else:
            self.train_feature, *_ = self.get_grad(paddle_model, train_dataloader)
            paddle.save(self.train_feature, "./cached_train_grad.tensor")

    def interpret(self, data, sample_num=3, sim_fn="dot"):
        """
        Select most similar examples for a given data using the `sim_fn` metric.
        Args:
            data(iterable): Dataloader to interpret.
            sample_sum(int: default=3): the number of positive examples and negtive examples selected for each instance.
            sim_fn(str: default=dot): the similarity metric to select examples.
        """
        examples = []
        val_feature, _, preds = self.get_grad(self.paddle_model, data)

        if sim_fn == "dot":
            for index, target_class in enumerate(preds):
                tmp = paddle.sum(
                    self.train_feature * paddle.to_tensor(val_feature[index]), axis=1
                )
                example_index = self._get_similarity_index(tmp, sample_num=sample_num)
                examples.append(example_index)
        if sim_fn == "cos":
            for index, target_class in enumerate(preds):
                tmp = F.cosine_similarity(
                    self.train_feature,
                    paddle.to_tensor(val_feature[index]).unsqueeze(0),
                )
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
        return paddle.concat(
            [
                paddle.flatten(p.grad)
                for n, p in self.paddle_model.named_parameters()
                if self.classifier_layer_name in n
            ]
        )

    def _clear_all_grad(self):
        """
        clear gradient
        """
        for p in self.paddle_model.parameters():
            p.clear_gradient()
