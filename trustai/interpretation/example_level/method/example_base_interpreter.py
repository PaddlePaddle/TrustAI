import abc
import functools

import paddle

from ...base_interpret import Interpreter
from ..common.utils import get_sublayer


class ExampleBaseInterpreter(Interpreter):
    """Interpreter is the base class for all interpretation algorithms.
    Args:
        paddle_model (callable): A model with ``forward`` and possibly ``backward`` functions.
        device (str): The device used for running `paddle_model`, options: ``cpu``, ``gpu``, ``gpu:0``, ``gpu:1`` etc. default: None
        predict_fn(callabel: default=None): If the paddle_model prediction has special process, user can customize the prediction function.
        classifier_layer_name(str: default=classifier): Name of the classifier layer in paddle_model.
    """

    def __init__(self,
                 paddle_model: callable,
                 device: str = None,
                 predict_fn=None,
                 classifier_layer_name="classifier",
                 **kwargs):
        Interpreter.__init__(self, paddle_model, device)
        self.paddle_model = paddle_model
        self.classifier_layer_name = classifier_layer_name
        self._build_predict_fn(predict_fn=predict_fn)

    def __call__(self, *args, **kwargs):
        return self.interpret(*args, **kwargs)

    def _build_predict_fn(self, predict_fn=None):
        if predict_fn is not None:
            self.predict_fn = functools.partial(predict_fn,
                                                classifier_layer_name=self.classifier_layer_name,
                                                paddle_model=self.paddle_model)
            return

        def predict_fn(inputs, classifier_layer_name=None, paddle_model=None):
            """predict_fn"""
            if paddle_model is None:
                paddle_model = self.paddle_model
            if classifier_layer_name is None:
                classifier_layer_name = self.classifier_layer_name

            cached_features = []

            def forward_pre_hook(layer, input):
                """
                Pre_hook for a given layer in model.
                """
                cached_features.extend(input[0])

            cached_logits = []

            def forward_post_hook(layer, input, output):
                """
                Post_hook for a given layer in model.
                """
                cached_logits.append(output)

            classifier = get_sublayer(paddle_model, classifier_layer_name)

            forward_pre_hook_handle = classifier.register_forward_pre_hook(forward_pre_hook)
            forward_post_hook_handle = classifier.register_forward_post_hook(forward_post_hook)

            if isinstance(inputs, (tuple, list)):
                res = paddle_model(*inputs)  # get logits, [bs, num_c]
            else:
                res = paddle_model(inputs)  # get logits, [bs, num_c]

            forward_pre_hook_handle.remove()
            forward_post_hook_handle.remove()

            logits = cached_logits[-1]
            if len(logits.shape) < 2:
                logits = logits.unsqueeze(0)

            probas = paddle.nn.functional.softmax(cached_logits[-1], axis=1)  # get probabilities.
            preds = paddle.argmax(probas, axis=1).tolist()  # get predictions.
            return paddle.to_tensor(cached_features), probas, preds

        self.predict_fn = predict_fn

    @abc.abstractmethod
    def interpret(self, **kwargs):
        """Main function of the interpreter."""
        raise NotImplementedError
