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

import paddle
from paddlenlp.transformers import AutoModel
from utils import logger


class Selector(paddle.nn.Layer):

    def __init__(self, args):
        super(Selector, self).__init__()
        self.args = args
        self.logger = logger.Logger(args)
        self.base_model = AutoModel.from_pretrained(args.model_name)
        self.linear = paddle.nn.Linear(self.args.hidden_size, 2)
        self.criterion = paddle.nn.CrossEntropyLoss()

    def extract(self, idx_list: list, sequence_input, len_sentence=-1):
        """
        Extract specific id tensor from encoded sequence tensor with `idxlist`.

        Args:
            idx_list (list): a batch of specific index position list. shape=[batch_size, index_len]
            sequence_input (tensor): encoded sequence tensor. shape=[batch_size,seq_len,hidden_size].

        Returns:
            extracted_sequence_input: an extracted and padded sequence input with shape=[batch_size,max_idx_len,hidden_size]

        Example:
            .. code-block::

                from paddlenlp.transformers import AutoModelForConditionalGeneration

                # Name of built-in pretrained model
                model = AutoModelForConditionalGeneration.from_pretrained('bart-base')
                print(type(model))
                # <class 'paddlenlp.transformers.bart.modeling.BartForConditionalGeneration'>


                # Load from local directory path
                model = AutoModelForConditionalGeneration.from_pretrained('./my_bart/')
                print(type(model))
                # <class 'paddlenlp.transformers.bart.modeling.BartForConditionalGeneration'>
        """
        if len_sentence <= 0:
            len_sentence = max([len(idx_l) for idx_l in idx_list])
        sentence_output = None
        step_batch_size = sequence_input.shape[0]
        for i in range(step_batch_size):
            index = paddle.to_tensor(idx_list[i], dtype='int32')
            temp_out = paddle.gather(sequence_input[i], index, axis=0)

            padding_tensor = paddle.zeros([len_sentence - temp_out.shape[0], temp_out.shape[1]], dtype="float32")
            out = paddle.concat(x=[temp_out, padding_tensor], axis=0)

            if sentence_output is None:
                sentence_output = paddle.unsqueeze(out, axis=0)
            else:
                sentence_output = paddle.concat(x=[sentence_output, paddle.unsqueeze(out, axis=0)], axis=0)
        return sentence_output

    def forward(self, x, mode="train", tokenizer=None):

        if mode not in ["train", "dev", "test"]:
            self.logger.error("Unexpected mode, please select the mode in ['train','dev','test']")
            return

        # unpack data from dataloader input
        data = {
            "input_ids": x[0],
            "token_type_ids": x[1],
            "attention_mask": x[2],
            "label_list": x[3],
            "question_input_ids": x[4],
            "question_token_type_ids": x[5],
            "question_attention_mask": x[6],
            "start_list": x[7],
            "end_list": x[8],
            "ground_label_list": x[9]
        }
        if len(x) == 11:
            data["raw_data"] = x[10]
        # encode
        sequence_output, _ = self.base_model(input_ids=data["input_ids"],
                                             token_type_ids=data["token_type_ids"],
                                             attention_mask=data["attention_mask"])

        # extract BSENT and ESENT presentation as sentence presentation
        start_present = self.extract(data["start_list"], sequence_output)
        end_present = self.extract(data["end_list"], sequence_output)
        sentence_present = (start_present + end_present) / 2  # [batch_size, sentence_num, hidden_size]

        # Encode question and activate question attention
        _, pooled_output = self.base_model(
            input_ids=data["question_input_ids"],
            token_type_ids=data["question_token_type_ids"],
            attention_mask=data["question_attention_mask"],
        )

        # Decode to select sentences
        if self.args.use_similarity:
            question_presentation = paddle.unsqueeze(pooled_output, axis=1)
            question_presentation = paddle.expand(question_presentation, shape=sentence_present.shape)

            selected_logit = paddle.nn.functional.cosine_similarity(sentence_present, question_presentation, axis=2)
        else:
            selected_logit = self.linear(sentence_present)

        if mode == "train":
            loss = self.criterion(selected_logit, data["label_list"])
            return loss, selected_logit

        elif mode == "dev" or mode == "test":
            result = []

            # Strategy 1: use threshold α to loosen selector
            if self.args.one_alpha > 0:
                selected_prob = paddle.nn.functional.softmax(selected_logit, axis=2)
                squeezed_selected_prob = paddle.squeeze(paddle.chunk(selected_prob, chunks=2, axis=2)[1], axis=2)
                pred_label = paddle.tolist(paddle.cast(squeezed_selected_prob >= self.args.one_alpha, 'int32'))
            else:
                pred_label = paddle.tolist(paddle.argmax(selected_logit, axis=2))

            result = {}
            if "raw_data" in data:
                # use pred_label to select sentence.
                result["selected_context"] = {}
                for batch_idx, batch_selected_label in enumerate(pred_label):
                    temp_r = []
                    for sentence_idx, label in enumerate(batch_selected_label):
                        # skip padding sentence
                        if data["label_list"][batch_idx][sentence_idx] == -100:
                            continue
                        elif label == 1:
                            sentences = data["raw_data"][batch_idx]["sentences"]
                            # TODO: Remove get_k_sentences
                            temp_r += list(
                                range(max(0, sentence_idx - self.args.get_k_sentences),
                                      min(len(sentences), sentence_idx + self.args.get_k_sentences + 1)))
                    l2 = []
                    [l2.append(ttt) for ttt in temp_r if ttt not in l2]
                    temp_x = [data["raw_data"][batch_idx]["sentences"][ttt] for ttt in l2]
                    result["selected_context"][data["raw_data"][batch_idx]["id"]] = "".join(temp_x)

            # Assuming that a sentence with one answer is recalled, the sample answer is successfully recalled
            if self.args.use_loose_metric:
                loose_pred_label_list = []
                loose_ground_label_list = [1] * len(pred_label)
                # Find whether recall at least one answer
                for batch_idx, batch_ground_label in enumerate(data["ground_label_list"]):
                    recall_flag = 0
                    for sentence_idx, ground_label in enumerate(batch_ground_label):
                        if ground_label == 1 and pred_label[batch_idx][sentence_idx] == 1:
                            recall_flag = 1
                            break
                    loose_pred_label_list.append(recall_flag)
                result["loose_pred_label"] = loose_pred_label_list
                result["loose_true_label"] = loose_ground_label_list

            temp_pred_label = []
            temp_ground_label = []
            # Remove padding to evaluate
            for idx_sim, batch_sim in enumerate(pred_label):
                temp_pred_label += batch_sim[:len(data["ground_label_list"][idx_sim])]
                temp_ground_label += data["ground_label_list"][idx_sim]

            result["pred_label"] = temp_pred_label
            result["true_label"] = temp_ground_label

            return result
