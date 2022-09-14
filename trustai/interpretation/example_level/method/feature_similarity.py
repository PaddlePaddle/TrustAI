# !/usr/bin/env python3
"""
feature-based similarity method.
cosine, cot and euc.
"""
import os
import sys
import functools
import warnings

import paddle
import paddle.nn.functional as F

from ..common.utils import get_sublayer, dot_similarity, cos_similarity, euc_similarity, get_top_and_bottom_n_examples, get_struct_res
from .example_base_interpreter import ExampleBaseInterpreter


class FeatureSimilarityModel(ExampleBaseInterpreter):
    """
    Feature-based similarity method for NLP tasks.
    """

    def __init__(
        self,
        paddle_model,
        train_dataloader,
        device=None,
        classifier_layer_name="classifier",
        predict_fn=None,
        cached_train_feature=None,
    ):
        """
        Initialization.
        Args:
            paddle_model(callable): A model with ``forward``.
            train_dataloader(iterable): Dataloader of model's training data.
            device(str: default=None): Device type, and it should be ``cpu``, ``gpu``, ``gpu:0``, ``gpu:1``  etc.
            classifier_layer_name(str: default=classifier): Name of the classifier layer in paddle_model.
            predict_fn(callabel: default=None): If the paddle_model prediction has special process, user can customize the prediction function.
        """
        ExampleBaseInterpreter.__init__(self, paddle_model, device, predict_fn, classifier_layer_name)
        self.paddle_model = paddle_model
        self.classifier_layer_name = classifier_layer_name

        if cached_train_feature is not None and os.path.isfile(cached_train_feature):
            self.train_feature = paddle.load(cached_train_feature)
        else:
            self.train_feature, _ = self.extract_feature_from_dataloader(train_dataloader)
            if cached_train_feature is not None:
                try:
                    paddle.save(self.train_feature, cached_train_feature)
                except IOError:
                    import sys
                    sys.stderr.write("save cached_train_feature fail")

    def interpret(self, data, sample_num=3, sim_fn="cos"):
        """
        Select most similar and dissimilar examples for a given data using the `sim_fn` metric.
        Args:
            data(iterable): one batch of data to interpret.
            sample_num(int: default=3): the number of positive examples and negtive examples selected for each instance. Return all the training examples ordered by `influence score` if this parameter is -1.
            sim_fn(str: default=cos): the similarity metric to select examples. It should be ``cos``, ``dot`` or ``euc``.
        """
        if sample_num == -1:
            sample_num = len(self.train_feature)
        pos_examples = []
        neg_examples = []
        val_feature, preds = self.extract_feature(self.paddle_model, data)
        if sim_fn == "dot":
            similarity_fn = dot_similarity
        elif sim_fn == "cos":
            similarity_fn = cos_similarity
        elif sim_fn == "euc":
            similarity_fn = euc_similarity
        else:
            raise ValueError(f"sim_fn only support ['dot', 'cos', 'euc'] in feature similarity, but gets `{sim_fn}`")
        for index in range(len(preds)):
            tmp = similarity_fn(self.train_feature, paddle.to_tensor(val_feature[index]))
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
        feature, _, pred = self.predict_fn(data)
        return paddle.to_tensor(feature), paddle.to_tensor(pred)

    def extract_feature_from_dataloader(self, dataloader):
        """
        extract feature from data_loader.
        """
        print("Extracting feature from given dataloader, it will take some time...")
        features, preds = [], []

        for batch in dataloader:
            feature, pred = self.extract_feature(self.paddle_model, batch)
            features.append(feature)
            preds.append(pred)
        return paddle.concat(features, axis=0), paddle.concat(preds, axis=0)