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
"""base interpreter"""

import abc
import sys
import numpy as np
import re
import warnings

from .python_utils import versiontuple2tuple


class Interpreter(abc.ABC):
    """Interpreter is the base class for all interpretation algorithms.
    Args:
        paddle_model (callable): A model with ``forward`` and possibly ``backward`` functions.
        device (str): The device used for running `paddle_model`, options: ``cpu``, ``gpu``, ``gpu:0``, ``gpu:1`` etc. default: None
    """

    def __init__(self, paddle_model: callable, device: str = None, **kwargs):
        self.device = device
        self.paddle_model = paddle_model
        self.predict_fn = None

        assert self.device is None or isinstance(self.device, str) and re.search(
            r"^cpu$|^gpu$|^gpu:\d+$", self.device
        ) is not None, "The format of the ``devices`` should be like ``cpu``, ``gpu``, ``gpu:0``, ``gpu:1`` etc."

        self._paddle_env_set()

    def __call__(self, *args, **kwargs):
        return self.interpret(*args, **kwargs)

    @abc.abstractmethod
    def interpret(self, **kwargs):
        """Main function of the interpreter."""
        raise NotImplementedError

    @abc.abstractmethod
    def _build_predict_fn(self, **kwargs):
        """Build self.predict_fn for interpreters."""
        raise NotImplementedError

    def _paddle_env_set(self):
        import paddle
        if self.device is not None:
            if not paddle.is_compiled_with_cuda() and self.device[:3] == 'gpu':
                warnings.warn("Paddle is not installed with GPU support. Change to CPU version now.")
                self.device = 'cpu'

            # globally set device.
            paddle.set_device(self.device)
            self.paddle_model.to(self.device)

        if versiontuple2tuple(paddle.__version__) >= (2, 2, 1):
            # From Paddle2.2.1, gradients are supported in eval mode.
            self.paddle_model.eval()
        else:
            # Former versions.
            self.paddle_model.train()
            for n, v in self.paddle_model.named_sublayers():
                if "batchnorm" in v.__class__.__name__.lower():
                    v._use_global_stats = True
                if "dropout" in v.__class__.__name__.lower():
                    v.p = 0
