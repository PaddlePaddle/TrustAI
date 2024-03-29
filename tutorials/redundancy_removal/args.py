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

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", type=str, required=True, help="Name of pre-trained model.")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="The data directory should include `train` and `dev` set to train model and `test` set to test model.")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--batch_size", default=24, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=7e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=20, type=int, help="Total number of train epochs to perform.")
    parser.add_argument("--max_steps",
                        default=-1,
                        type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion",
                        default=0.0,
                        type=float,
                        help="Proportion of training steps to perform linear learning rate warmup for.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every X updates steps.")
    parser.add_argument("--load_model_path", type=str, default=None, help="The checkpoint directory where the model")
    parser.add_argument("--get_k_sentences", type=int, default=0, help="load checkpoint path")
    parser.add_argument("--set_k_sentences_ground_true", type=int, default=0, help="set k sentences ground true")
    parser.add_argument("--early_stop_nums", type=int, default=5, help="probability threshold for selecting sentences")
    parser.add_argument("--one_alpha", type=float, default=0.4, help="probability threshold for selecting sentences")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument('--device',
                        choices=['cpu', 'gpu'],
                        default="gpu",
                        help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--doc_stride",
                        type=int,
                        default=128,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_query_length", type=int, default=64, help="Max query length.")
    parser.add_argument("--max_answer_length", type=int, default=30, help="Max answer length.")
    parser.add_argument("--hidden_size", type=int, default=768, help="hidden size")
    parser.add_argument("--verbose", action='store_true', help="Whether to output verbose log.")
    parser.add_argument("--do_train", action='store_true', help="Whether to train the model.")
    parser.add_argument("--do_predict", action='store_true', help="Whether to predict.")
    parser.add_argument("--use_loose_metric", action='store_true', help="whether to use loose metric to choose model.")
    parser.add_argument("--use_similarity", action='store_true', help="whether to use similarity to choose sentence.")
    parser.add_argument("--early_stop", action='store_true', help="whether to use early stop.")
    args = parser.parse_args()
    return args
