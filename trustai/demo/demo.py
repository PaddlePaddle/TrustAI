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
"""pretrain for demo"""

import logging
import logging.handlers
import os
import sys
import requests
import shutil
import tarfile
import warnings
import functools

from tqdm import tqdm
from paddle.io import DataLoader, BatchSampler
try:
    from paddlenlp.transformers import *
    from paddlenlp.datasets import load_dataset

except ImportError as e:
    sys.stderr.write(
        '''The demo module depends on paddlenlp, please install paddlenlp firstly. cmd: pip install -U paddlenlp. ''')
    exit(-1)

from .utils import DOWNLOAD_MODEL_PATH_DICT, MODEL_HOME, get_path_from_url
from .utils import LocalDataCollatorWithPadding, preprocess_function, get_path_from_url


class DEMO(object):

    def __init__(self, task_name, device: str = None):
        self.device = device
        assert self.device is None or isinstance(self.device, str) and re.search(
            r"^cpu$|^gpu$|^gpu:\d+$", self.device
        ) is not None, "The format of the ``devices`` should be like ``cpu``, ``gpu``, ``gpu:0``, ``gpu:1`` etc."

        self._paddle_env_set()
        self.task_name = task_name
        model_path = self.get_model_path(task_name)
        self.paddle_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.unk_id = self.tokenizer.unk_token_id
        self.pad_id = self.tokenizer.pad_token_type_id
        self.cls_id = self.tokenizer.cls_token_id
        self.mask_id = self.tokenizer.mask_token_id

    def get_model_path(self, model_name):
        try:
            model_url, md5sum = DOWNLOAD_MODEL_PATH_DICT[model_name]
        except KeyError:
            logging.warn(
                f"The model_name `{model_name}` is wrong, currently only the following models are supported : {', '.join(DOWNLOAD_MODEL_PATH_DICT.keys())}."
            )
            exit(-1)
        model_path = get_path_from_url(model_url, MODEL_HOME, md5sum=md5sum)
        return model_path

    def get_model(self):
        return self.paddle_model

    def get_tokenizer(self):
        return self.tokenizer

    def get_train_data_and_dataloader(self, batch_size=8, max_seq_length=256):
        task_name = self.task_name.split('/')
        if len(task_name) == 2:
            train_ds = load_dataset(task_name[0], name=task_name[1], splits=["train"])
        else:
            train_ds = load_dataset(task_name[0], splits=["train"])
        trans_func = functools.partial(preprocess_function,
                                       max_seq_length=max_seq_length,
                                       tokenizer=self.tokenizer,
                                       is_test=True)
        train_ds = train_ds.map(trans_func)
        train_batch_sampler = BatchSampler(train_ds, batch_size=batch_size, shuffle=False)
        collate_fn = LocalDataCollatorWithPadding(self.tokenizer)
        train_data_loader = DataLoader(dataset=train_ds, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
        return train_ds.data, train_data_loader,

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    def process(self, text, text_pair=None):
        tokenize_result = self.tokenizer(text, text_pair=text_pair, return_tensors='pd', padding=True)
        input_ids = tokenize_result['input_ids']
        token_type_ids = tokenize_result['token_type_ids']
        tokens = [self.tokenizer.convert_ids_to_tokens(_input_ids) for _input_ids in input_ids.tolist()]
        return tokens, (input_ids, token_type_ids)

    def _paddle_env_set(self):
        import paddle
        if self.device is not None:
            if not paddle.is_compiled_with_cuda() and self.device[:3] == 'gpu':
                warnings.warn("Paddle is not installed with GPU support. Change to CPU version now.")
                self.device = 'cpu'

            # globally set device.
            paddle.set_device(self.device)
            self.paddle_model.to(self.device)

    def __getitem__(self, key):
        return getattr(self, key)