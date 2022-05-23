# !/usr/bin/env python3
"""
feature-based similarity method.
cosine, cot and euc.
"""
import functools
import warnings

import paddle
import paddle.nn.functional as F

from ..common.utils import get_sublayer, dot_similarity, cos_similarity, euc_similarity, get_top_and_bottom_n_examples
from .example_base_interpreter import ExampleBaseInterpreter


class FeatureSimilarityModel(ExampleBaseInterpreter):
    """
    Feature-based similarity method for NLP tasks.
    """

    def __init__(
        self,
        paddle_model,
        train_dataloader,
        device="gpu",
        classifier_layer_name="classifier",
        predict_fn=None,
    ):
        """
        Initialization.
        Args:
            paddle_model(callable): A model with ``forward``.
            train_dataloader(iterable): Dataloader of model's training data.
            device(str: default=gpu): Device type, and it should be ``gpu``, ``cpu`` etc.
            classifier_layer_name(str: default=classifier): Name of the classifier layer in paddle_model.
            predict_fn(callabel: default=None): If the paddle_model prediction has special process, user can customize the prediction function.
        """
        ExampleBaseInterpreter.__init__(self, paddle_model, device, predict_fn, classifier_layer_name)
        self.paddle_model = paddle_model
        self.classifier_layer_name = classifier_layer_name
        self.train_feature, _ = self.extract_featue(paddle_model, train_dataloader)

    def interpret(self, data, sample_num=3, sim_fn="cos"):
        """
        Select most similar and dissimilar examples for a given data using the `sim_fn` metric.
        Args:
            data(iterable): Dataloader to interpret.
            sample_sum(int: default=3): the number of positive examples and negtive examples selected for each instance.
            sim_fn(str: default=cos): the similarity metric to select examples. It should be ``cos``, ``dot`` or ``euc``.
        """
        pos_examples = []
        neg_examples = []
        val_feature, preds = self.extract_featue(self.paddle_model, data)
        if sim_fn == "dot":
            similarity_fn = dot_similarity
        elif sim_fn == "cos":
            similarity_fn = cos_similarity
        elif sim_fn == "euc":
            similarity_fn = euc_similarity
        else:
            raise ValueError(f"sim_fn only support ['dot', 'cos', 'euc'] in feature similarity, but gets `{sim_fn}`")
        for index, target_class in enumerate(preds):
            tmp = similarity_fn(self.train_feature, paddle.to_tensor(val_feature[index]))
            pos_idx, neg_idx = get_top_and_bottom_n_examples(tmp, sample_num=sample_num)
            pos_examples.append(pos_idx)
            neg_examples.append(neg_idx)
        return preds.tolist(), pos_examples, neg_examples

    @paddle.no_grad()
    def extract_featue(self, paddle_model, data_loader):
        """
        extract feature for data_loader.
        """
        paddle_model.eval()
        print("Extracting feature for given dataloader, it will take some time...")
        features, preds = [], []

        for step, batch in enumerate(data_loader, start=1):
            feature, _, pred = self.predict_fn(batch)
            features.extend(feature)
            preds.extend(pred)
        return paddle.to_tensor(features), paddle.to_tensor(preds)
