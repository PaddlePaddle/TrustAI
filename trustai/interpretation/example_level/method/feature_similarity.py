# !/usr/bin/env python3
"""
feature-based similarity method.
cosine, cot and euc.
"""
import functools
import warnings

import paddle
import paddle.nn.functional as F

from ..common.utils import get_sublayer
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

    def interpret(self, data, sample_num=3, sim_fn="dot"):
        """
        Select most similar and dissimilar examples for a given data using the `sim_fn` metric.
        Args:
            data(iterable): Dataloader to interpret.
            sample_sum(int: default=3): the number of positive examples and negtive examples selected for each instance.
            sim_fn(str: default=dot): the similarity metric to select examples.
        """
        examples = []
        val_feature, preds = self.extract_featue(self.paddle_model, data)
        if sim_fn == "dot":
            similarity_fn = self._dot_similarity
        elif sim_fn == "cos":
            similarity_fn = self._cos_similarity
        elif sim_fn == "euc":
            similarity_fn = self._euc_similarity
        else:
            warnings.warn("only support ['dot', 'cos', 'eud']")
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
        return paddle.sum(self.train_feature * paddle.to_tensor(inputs), axis=1)

    def _cos_similarity(self, inputs):
        return F.cosine_similarity(self.train_feature, paddle.to_tensor(inputs).unsqueeze(0))

    def _euc_similarity(self, inputs):
        return -paddle.linalg.norm(self.train_feature - paddle.to_tensor(inputs).unsqueeze(0), axis=-1).squeeze(-1)

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
