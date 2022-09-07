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

import random

import numpy as np
import paddle

from args import parse_args
from selector import model, model_manager


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


def run(args):
    set_seed(args)

    # Prepare device and model
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    rank = paddle.distributed.get_rank()

    selector = model.Selector(args)
    if paddle.distributed.get_world_size() > 1:
        selector = paddle.DataParallel(selector)

    # Prepare model manager
    manager = model_manager.ModelManager(args, selector)

    if args.do_train:
        manager.train(rank)

    if args.do_predict and rank == 0:
        manager.test()


if __name__ == "__main__":
    args = parse_args()
    run(args)
