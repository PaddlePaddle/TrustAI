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

import json
import math
import os

import paddle
from paddlenlp.metrics.squad import compute_prediction, squad_evaluate
from paddlenlp.transformers import AutoTokenizer, LinearDecayWithWarmup
from tqdm import tqdm
from utils import logger

from predictor import dataloader_factory


class ModelManager():

    def __init__(self, args, model):

        self.args = args
        self.logger = logger.Logger(args)
        self.global_step = 0

        # Init Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.tokenizer.add_tokens("[BSENT]", "[ESENT]")

        # Init model
        self.model = model
        if args.load_model_path is not None:
            self.model.set_state_dict(paddle.load(args.load_model_path))

        # Load Train/Dev data and prepare optimizer to train model
        if self.args.do_train:
            self.train_data_loader = dataloader_factory.get_dataloader(args, self.tokenizer, split="train")
            self.dev_data_loader, self.dev_raw_data = dataloader_factory.get_dataloader(args,
                                                                                        self.tokenizer,
                                                                                        split="dev",
                                                                                        return_raw_data=True)
            self.num_training_steps = args.max_steps if args.max_steps > 0 else len(
                self.train_data_loader) * args.num_train_epochs
            self.num_train_epochs = math.ceil(self.num_training_steps / len(self.train_data_loader))
            self.lr_scheduler = LinearDecayWithWarmup(args.learning_rate, self.num_training_steps,
                                                      args.warmup_proportion)
            # Generate parameter names needed to perform weight decay.
            # All bias and LayerNorm parameters are excluded.
            self.decay_params = [
                p.name for n, p in self.model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])
            ]
            self.optimizer = paddle.optimizer.AdamW(learning_rate=self.lr_scheduler,
                                                    epsilon=args.adam_epsilon,
                                                    parameters=self.model.parameters(),
                                                    weight_decay=args.weight_decay,
                                                    apply_decay_param_fun=lambda x: x in self.decay_params)

        # Load Test data
        if self.args.do_predict:
            self.test_data_loader, self.test_raw_data = dataloader_factory.get_dataloader(args,
                                                                                          self.tokenizer,
                                                                                          split="test",
                                                                                          return_raw_data=True)

    def model_step(self, train_data):
        loss, result = self.model(train_data, mode="train")
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.clear_grad()
        return loss, result

    def save_model(self, tokenizer=None):
        output_dir = os.path.join(self.args.output_dir, "model_%d" % self.global_step)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = self.model._layers if isinstance(self.model, paddle.DataParallel) else self.model
        paddle.save(model_to_save.state_dict(), os.path.join(output_dir, "model_state.pdparams"))
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        self.logger.info('Saving checkpoint to:' + output_dir)

    def save_prediction(self, all_predictions):
        output_dir = os.path.join(self.args.output_dir, "model_%d" % self.global_step)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, 'prediction.json'), "w", encoding='utf-8') as writer:
            writer.write(json.dumps(all_predictions, ensure_ascii=False, indent=4) + "\n")

    def train(self, rank):
        self.logger.info("Start Training....")
        highest_result = 0
        early_stop_count = 0
        for epoch in range(self.num_train_epochs):
            for step, train_data in enumerate(self.train_data_loader):
                self.global_step += 1
                # forward
                loss, _ = self.model_step(train_data)
                self.logger.logging_step(self.global_step, epoch, step, loss)

                # evaluate model and save best checkpoint
                if self.global_step % self.args.save_steps == 0 or self.global_step == self.num_training_steps:
                    if rank == 0:
                        performance_result, all_predictions = self.evaluate()
                        self.logger.add_performance(performance_result)
                        if performance_result["exact"] > highest_result:
                            highest_result = performance_result["exact"]
                            self.save_model(tokenizer=self.tokenizer)
                            self.save_prediction(all_predictions)
                            early_stop_count = 0
                        else:
                            early_stop_count += 1
                if self.global_step == self.num_training_steps:
                    break
                if self.args.early_stop and early_stop_count >= self.args.early_stop_nums:
                    self.logger.info("Early Stop!")
                    break
            if self.args.early_stop and early_stop_count >= self.args.early_stop_nums:
                break

        self.logger.save_performance()
        self.logger.info("Congratulations! You have finished selector training!")

    @paddle.no_grad()
    def evaluate(self, auto_metric=True):
        self.model.eval()

        all_start_logits = []
        all_end_logits = []
        for batch in tqdm(self.dev_data_loader, total=len(self.dev_data_loader)):
            start_logits_tensor, end_logits_tensor = self.model(batch, mode="dev")

            for idx in range(start_logits_tensor.shape[0]):
                all_start_logits.append(start_logits_tensor.numpy()[idx])
                all_end_logits.append(end_logits_tensor.numpy()[idx])

        all_predictions, _, _ = compute_prediction(self.dev_raw_data, self.dev_data_loader.dataset,
                                                   (all_start_logits, all_end_logits), False, self.args.n_best_size,
                                                   self.args.max_answer_length)

        self.model.train()
        if auto_metric:
            result = squad_evaluate(examples=[raw_data for raw_data in self.dev_raw_data],
                                    preds=all_predictions,
                                    is_whitespace_splited=False)
            return result, all_predictions
        return all_predictions

    @paddle.no_grad()
    def test(self):
        self.model.eval()

        all_start_logits = []
        all_end_logits = []
        for batch in tqdm(self.test_data_loader, total=len(self.test_data_loader)):
            start_logits_tensor, end_logits_tensor = self.model(batch, mode="dev")

            for idx in range(start_logits_tensor.shape[0]):
                all_start_logits.append(start_logits_tensor.numpy()[idx])
                all_end_logits.append(end_logits_tensor.numpy()[idx])

        all_predictions, _, _ = compute_prediction(self.test_raw_data, self.test_data_loader.dataset,
                                                   (all_start_logits, all_end_logits), False, self.args.n_best_size,
                                                   self.args.max_answer_length)

        self.model.train()
        self.save_prediction(all_predictions)
        return all_predictions
