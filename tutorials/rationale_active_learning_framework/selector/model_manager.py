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
from paddlenlp.transformers import AutoTokenizer, LinearDecayWithWarmup
from sklearn import metrics
from tqdm import tqdm
from utils import logger

from selector import dataloader_factory


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
            self.train_data_loader = dataloader_factory.get_dataloader(args,
                                                                       data_dir_path=args.data_dir,
                                                                       batch_size=args.batch_size,
                                                                       tokenizer=self.tokenizer,
                                                                       split="train",
                                                                       return_raw_data=True)
            self.dev_data_loader = dataloader_factory.get_dataloader(args,
                                                                     data_dir_path=args.data_dir,
                                                                     batch_size=args.batch_size,
                                                                     tokenizer=self.tokenizer,
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
            self.test_data_loader = dataloader_factory.get_dataloader(args,
                                                                      data_dir_path=args.data_dir,
                                                                      batch_size=args.batch_size,
                                                                      tokenizer=self.tokenizer,
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
        output_dir = os.path.join(self.args.output_dir, "best_model")
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

    def data_context_replace(self, filepath, pred_dict, output_path):
        with open(filepath, encoding="utf-8") as f:
            durobust = json.load(f)
        for i, article in enumerate(durobust["data"]):
            for j, paragraph in enumerate(article["paragraphs"]):
                for k, qa in enumerate(paragraph["qas"]):
                    ids = qa["id"]
                    if ids in pred_dict.keys():
                        durobust["data"][i]["paragraphs"][j]["context"] = pred_dict[ids]

        with open(output_path, 'w', encoding="utf8") as outfile:
            json.dump(durobust, outfile, ensure_ascii=False)

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
                        if self.args.use_loose_metric:
                            rs = performance_result["precision"] * 0.2 + 0.8 * performance_result["losse_recall"]
                        else:
                            rs = performance_result["recall"]

                        if rs > highest_result:
                            highest_result = rs
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
    def evaluate(self):
        self.model.eval()

        pred_list = []
        true_list = []
        all_predictions = {}
        if self.args.use_loose_metric:
            loose_pred_list = []
            loose_true_list = []
        for step, dev_data in enumerate(self.dev_data_loader):
            output = self.model(dev_data, mode="dev", tokenizer=self.tokenizer)
            pred_list += output["pred_label"]
            true_list += output["true_label"]
            if self.args.use_loose_metric:
                loose_pred_list += output["loose_pred_label"]
                loose_true_list += output["loose_true_label"]
            all_predictions.update(output["selected_context"])

        # compute metrics
        result = {}
        result["F1"] = metrics.f1_score(true_list, pred_list, average='binary')
        result["precision"] = metrics.precision_score(true_list, pred_list, average='binary')
        result["recall"] = metrics.recall_score(true_list, pred_list, average='binary')
        if self.args.use_loose_metric:
            result["losse_f1"] = metrics.f1_score(loose_true_list, loose_pred_list, average='binary')
            result["losse_precision"] = metrics.precision_score(loose_true_list, loose_pred_list, average='binary')
            result["losse_recall"] = metrics.recall_score(loose_true_list, loose_pred_list, average='binary')

        self.logger.logging_result(result)
        self.model.train()
        return result, all_predictions

    @paddle.no_grad()
    def test(self):
        self.model.eval()
        all_predictions = {}
        for step, test_data in tqdm(enumerate(self.test_data_loader), total=len(self.test_data_loader)):
            output = self.model(test_data, mode="dev", tokenizer=self.tokenizer)
            all_predictions.update(output["selected_context"])
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        self.data_context_replace(os.path.join(self.args.data_dir, "test.json"), all_predictions,
                                  os.path.join(self.args.output_dir, "test_prediction.json"))

        self.model.train()
