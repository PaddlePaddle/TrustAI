from __future__ import print_function
from collections import OrderedDict
import io
import json
import sys
import ast
import argparse
import paddle
import paddlenlp
from paddlenlp.datasets import DatasetBuilder
from paddlenlp.metrics.squad import squad_evaluate
import collections
import time
import numpy as np
from functools import partial
from paddlenlp.data import Stack, Dict, Pad
from trustai.interpretation import get_word_offset
from paddle import tensor
from paddle.fluid import layers

class DuReader(DatasetBuilder):
    def _read(self, filename):
        with open(filename, 'r', encoding='utf8') as f:
            for line in f.readlines():
                example_dic = json.loads(line)
                id = example_dic['id']
                context = example_dic['context']
                question = example_dic['question']
                if 'sent_token' in example_dic:
                    sent_token = example_dic['sent_token']
                    yield {
                        'id': id,
                        'context': context,
                        'question': question,
                        'sent_token': sent_token
                    }


def _tokenize_chinese_chars(text):
    """
    :param text: input text, unicode string
    :return:
        tokenized text, list
    """

    def _is_chinese_char(cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    output = []
    buff = ""
    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp) or char == "=":
            if buff != "":
                output.append(buff)
                buff = ""
            output.append(char)
        else:
            buff += char

    if buff != "":
        output.append(buff)

    return output


def _normalize(in_str):
    """
    normalize the input unicode string
    """
    in_str = in_str.lower()
    sp_char = [
        u':', u'_', u'`', u'，', u'。', u'：', u'？', u'！', u'(', u')', u'“', u'”',
        u'；', u'’', u'《', u'》', u'……', u'·', u'、', u',', u'「', u'」', u'（', u'）',
        u'－', u'～', u'『', u'』', '|'
    ]
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


def find_lcs(s1, s2):
    """find the longest common subsequence between s1 ans s2"""
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    max_len = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > max_len:
                    max_len = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - max_len:p], max_len


def convert_to_dict(ref_ans):

    res = {}
    for ins in ref_ans:
        res[ins['id']]=ins
    return res


def evaluate_ch(ref_ans, pred_ans):
    """
    ref_ans: reference answers, dict
    pred_ans: predicted answer, dict
    return:
        f1_score: averaged F1 score
        em_score: averaged EM score
        total_count: number of samples in the reference dataset
        skip_count: number of samples skipped in the calculation due to unknown errors
    """
    f1 = 0
    em = 0
    total_count = 0
    skip_count = 0
    if isinstance(ref_ans, list):
        ref_ans = convert_to_dict(ref_ans)
    for query_id in ref_ans:
        sample = ref_ans[query_id]
        total_count += 1
        answers = sample['answers'][0]
        try:
            prediction = pred_ans[query_id]
        except:
            skip_count += 1
            continue
        if prediction == "" and answers == "":
            _f1 = 1.0
            _em = 1.0
        else:
            _f1 = calc_f1_score([answers], prediction)
            _em = calc_em_score([answers], prediction)
        f1 += _f1
        em += _em

    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count
    return {'f1':f1_score, 'em':em_score, 'total':total_count, 'skip':skip_count}


def calc_f1_score(answers, prediction):
    f1_scores = []
    for ans in answers:
        ans_segs = _tokenize_chinese_chars(_normalize(ans))
        prediction_segs = _tokenize_chinese_chars(_normalize(prediction))
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        prec = 1.0 * lcs_len / len(prediction_segs)
        rec = 1.0 * lcs_len / len(ans_segs)
        f1 = (2 * prec * rec) / (prec + rec)
        f1_scores.append(f1)
    return max(f1_scores)


def calc_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = _normalize(ans)
        prediction_ = _normalize(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em


class CrossEntropyLossForRobust(paddle.nn.Layer):
    def __init__(self):
        super(CrossEntropyLossForRobust, self).__init__()

    def forward(self, y, label):
        start_logits, end_logits = y
        start_position, end_position = label
        start_position = paddle.unsqueeze(start_position, axis=-1)
        end_position = paddle.unsqueeze(end_position, axis=-1)
        start_loss = paddle.nn.functional.cross_entropy(
            input=start_logits, label=start_position)
        end_loss = paddle.nn.functional.cross_entropy(
            input=end_logits, label=end_position)
        loss = (start_loss + end_loss) / 2
        return loss


def transform_old_format(data):
    res = {}
    for i in data[0]:
        res[i] = []
    for i in range(len(data)):
        for j in data[i]:
            res[j].append(data[i][j])
    return res


def compute_prediction(examples,
                       features,
                       predictions,
                       version_2_with_negative=False,
                       n_best_size=20,
                       max_answer_length=30,
                       null_score_diff_threshold=0.0):
    """
    Post-processes the predictions of a question-answering model to convert 
    them to answers that are substrings of the original contexts. This is 
    the base postprocessing functions for models that only return start and 
    end logits.

    Args:
        examples (list): List of raw squad-style data (see `run_squad.py 
            <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/
            machine_reading_comprehension/SQuAD/run_squad.py>`__ for more 
            information).
        features (list): List of processed squad-style features (see 
            `run_squad.py <https://github.com/PaddlePaddle/PaddleNLP/blob/
            develop/examples/machine_reading_comprehension/SQuAD/run_squad.py>`__
            for more information).
        predictions (tuple): The predictions of the model. Should be a tuple
            of two list containing the start logits and the end logits.
        version_2_with_negative (bool, optional): Whether the dataset contains
            examples with no answers. Defaults to False.
        n_best_size (int, optional): The total number of candidate predictions
            to generate. Defaults to 20.
        max_answer_length (int, optional): The maximum length of predicted answer.
            Defaults to 20.
        null_score_diff_threshold (float, optional): The threshold used to select
            the null answer. Only useful when `version_2_with_negative` is True.
            Defaults to 0.0.
    
    Returns:
        A tuple of three dictionaries containing final selected answer, all n_best 
        answers along with their probability and scores, and the score_diff of each 
        example.
    """
    assert len(
        predictions
    ) == 2, "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits = predictions
    #print(len(predictions[0]), len(features))
    assert len(predictions[0]) == len(
        features), "Number of predictions should be equal to number of features."

    # Build a map example to its corresponding features.
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[feature["example_id"]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    all_feature_index = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    # Let's loop over all the examples!
    for example_index, example in enumerate(examples):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example['id']]

        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get(
                "token_is_max_context", None)

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction[
                    "score"] > feature_null_score:
                min_null_prediction = {
                    "feature_index": (0, 0),
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1:-n_best_size - 1:
                                                     -1].tolist()
            end_indexes = np.argsort(end_logits)[-1:-n_best_size - 1:-1].tolist(
            )
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (start_index >= len(offset_mapping) or
                            end_index >= len(offset_mapping) or
                            offset_mapping[start_index] is None or
                            offset_mapping[end_index] is None or
                            offset_mapping[start_index] == (0, 0) or
                            offset_mapping[end_index] == (0, 0)):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(
                            str(start_index), False):
                        continue
                    prelim_predictions.append({
                        "feature_index": (start_index, end_index),
                        "offsets": (offset_mapping[start_index][0],
                                    offset_mapping[end_index][1]),
                        "score":
                        start_logits[start_index] + end_logits[end_index],
                        "start_logit": start_logits[start_index],
                        "end_logit": end_logits[end_index],
                    })
        if version_2_with_negative:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(
            prelim_predictions, key=lambda x: x["score"],
            reverse=True)[:n_best_size]

        # Add back the minimum null prediction if it was removed because of its low score.
        if version_2_with_negative and not any(p["offsets"] == (0, 0)
                                               for p in predictions):
            predictions.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0]:offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0 or (len(predictions) == 1 and
                                     predictions[0]["text"] == ""):
            predictions.insert(0, {
                "feature_index": (0, 0),
                "text": "empty",
                "start_logit": 0.0,
                "end_logit": 0.0,
                "score": 0.0
            })

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
            all_feature_index[example["id"]] = predictions[0]['feature_index']
        else:
            # Otherwise we first need to find the best non-empty prediction.
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # Then we compare to the null prediction using the threshold.
            score_diff = null_score - best_non_null_pred[
                "start_logit"] - best_non_null_pred["end_logit"]
            scores_diff_json[example["id"]] = float(
                score_diff)  # To be JSON-serializable.
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]
            all_feature_index[example["id"]] = predictions[i]['feature_index']

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [{
            k: (float(v)
                if isinstance(v, (np.float16, np.float32, np.float64)) else v)
            for k, v in pred.items()
        } for pred in predictions]

    return all_predictions, all_nbest_json, scores_diff_json, all_feature_index


@paddle.no_grad()
def evaluate(model, data_loader, is_test=False):
    model.eval()

    all_start_logits = []
    all_end_logits = []
    tic_eval = time.time()

    for batch in data_loader:
        input_ids, token_type_ids = batch
        start_logits_tensor, end_logits_tensor = model(input_ids,
                                                       token_type_ids)

        for idx in range(start_logits_tensor.shape[0]):
            if len(all_start_logits) % 1000 == 0 and len(all_start_logits):
                print("Processing example: %d" % len(all_start_logits))
                print('time per 1000:', time.time() - tic_eval)
                tic_eval = time.time()

            all_start_logits.append(start_logits_tensor.numpy()[idx])
            all_end_logits.append(end_logits_tensor.numpy()[idx])

    all_predictions, _, _, _ = compute_prediction(
        data_loader.dataset.data, data_loader.dataset.new_data,
        (all_start_logits, all_end_logits))

    eval_res = evaluate_ch(data_loader.dataset.data,all_predictions)
    
    model.train()
    return eval_res


def prepare_train_features(examples,tokenizer,doc_stride,max_seq_length):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    contexts = [examples[i]['context'] for i in range(len(examples))]
    questions = [examples[i]['question'] for i in range(len(examples))]

    tokenized_examples = tokenizer(
        questions,
        contexts,
        stride=doc_stride,
        max_seq_len=max_seq_length,
        return_dict=False)

    # Let's label those examples!
    for i, tokenized_example in enumerate(tokenized_examples):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_example["input_ids"]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offsets = tokenized_example['offset_mapping']

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_example['token_type_ids']

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = tokenized_example['overflow_to_sample']
        answers = examples[sample_index]['answers']
        answer_starts = examples[sample_index]['answer_starts']

        # Start/end character index of the answer in the text.
        start_char = answer_starts[0]
        end_char = start_char + len(answers[0])

        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1
        # Minus one more to reach actual text
        token_end_index -= 1

        # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
        if not (offsets[token_start_index][0] <= start_char and
                offsets[token_end_index][1] >= end_char):
            tokenized_examples[i]["start_positions"] = cls_index
            tokenized_examples[i]["end_positions"] = cls_index
        else:
            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
            # Note: we could go after the last offset if the answer is the last word (edge case).
            while token_start_index < len(offsets) and offsets[
                    token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples[i]["start_positions"] = token_start_index - 1
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples[i]["end_positions"] = token_end_index + 1

    return tokenized_examples

def prepare_validation_features(examples,tokenizer,doc_stride,max_seq_length):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    contexts = [examples[i]['context'] for i in range(len(examples))]
    questions = [examples[i]['question'] for i in range(len(examples))]

    tokenized_examples = tokenizer(
        questions,
        contexts,
        stride=doc_stride,
        max_seq_len=max_seq_length,
        return_dict=False)

    # For validation, there is no need to compute start and end positions
    for i, tokenized_example in enumerate(tokenized_examples):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_example['token_type_ids']

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = tokenized_example['overflow_to_sample']
        tokenized_examples[i]["example_id"] = examples[sample_index]['id']

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples[i]["offset_mapping"] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_example["offset_mapping"])
        ]

    return tokenized_examples


def training_mrc_model(model, 
                tokenizer,
                train_ds, 
                dev_ds,
                batch_size=12,
                epochs=3,
                learning_rate=5e-5,
                warmup_proportion=0.1,
                max_seq_length=512,
                doc_stride=512, 
                weight_decay=0.01,
                save_dir='save_model/base'):
    """
    An example of training an MRC model.
    """

    # Prepare data
    train_trans_func = partial(prepare_train_features, 
                            max_seq_length=max_seq_length, 
                            doc_stride=doc_stride,
                            tokenizer=tokenizer)

    train_ds.map(train_trans_func, batched=True, num_workers=4)

    dev_trans_func = partial(prepare_validation_features, 
                            max_seq_length=max_seq_length, 
                            doc_stride=doc_stride,
                            tokenizer=tokenizer)
                            
    dev_ds.map(dev_trans_func, batched=True, num_workers=4)

    # 定义BatchSampler
    train_batch_sampler = paddle.io.DistributedBatchSampler(
            train_ds, batch_size=batch_size, shuffle=True)

    dev_batch_sampler = paddle.io.BatchSampler(
        dev_ds, batch_size=batch_size, shuffle=False)


    # 定义batchify_fn
    train_batchify_fn = lambda samples, fn=Dict({
        "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
        "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
        "start_positions": Stack(dtype="int64"),
        "end_positions": Stack(dtype="int64")
    }): fn(samples)

    dev_batchify_fn = lambda samples, fn=Dict({
        "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
        "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id)
    }): fn(samples)

    # 构造DataLoader
    train_data_loader = paddle.io.DataLoader(
        dataset=train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=train_batchify_fn,
        return_list=True)

    dev_data_loader = paddle.io.DataLoader(
        dataset=dev_ds,
        batch_sampler=dev_batch_sampler,
        collate_fn=dev_batchify_fn,
        return_list=True)
    

    num_training_steps = len(train_data_loader) * epochs


    lr_scheduler = paddlenlp.transformers.LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_proportion)

    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)


    criterion = CrossEntropyLossForRobust()
    best_f1 = 0
    global_step = 0
    print("Training Starts:")
    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):

            global_step += 1
            input_ids, segment_ids, start_positions, end_positions = batch
            logits = model(input_ids=input_ids, token_type_ids=segment_ids)
            loss = criterion(logits, (start_positions, end_positions))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if global_step % 100 == 0 :
                print("global step %d, epoch: %d, batch: %d, loss: %.5f" % (global_step, epoch, step, loss))
            
        eval_res = evaluate(model=model, data_loader=dev_data_loader) 
        print("F1 on eval dataset:", eval_res['f1'])
        if best_f1 < eval_res['f1']:
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            best_f1 = eval_res['f1']

    print("best F1-score is %f!" % best_f1)


def attention_predict_fn(inputs,
                                      paddle_model,
                                      query_name="self_attn.q_proj",
                                      key_name="self_attn.k_proj",
                                      layer=11):
    """attention_predict_fn_on_paddlenlp"""
    query_feature = []
    key_feature = []

    def hook_for_query(layer, input, output):
        """hook_for_query"""
        query_feature.append(output)
        return output

    def hook_for_key(layer, input, output):
        """hook_for_key"""
        key_feature.append(output)
        return output

    hooks = []
    for name, v in paddle_model.named_sublayers():
        if str(layer) + '.' + query_name in name:
            h = v.register_forward_post_hook(hook_for_query)
            hooks.append(h)
        if str(layer) + '.' + key_name in name:
            h = v.register_forward_post_hook(hook_for_key)
            hooks.append(h)
    if isinstance(inputs, tuple):
        logits = paddle_model(*inputs)  # get logits, [bs, num_c]
    else:
        logits = paddle_model(input_ids=inputs[0], token_type_ids=inputs[1])  # get logits, [bs, num_c]
    bs = logits[0].shape[0]
    for h in hooks:
        h.remove()
    num_heads = paddle_model.ernie.config['num_attention_heads']
    hidden_size = paddle_model.ernie.config['hidden_size']
    head_dim = hidden_size // num_heads

    q = tensor.reshape(x=query_feature[0], shape=[0, 0, num_heads, head_dim])
    q = tensor.transpose(x=q, perm=[0, 2, 1, 3])
    k = tensor.reshape(x=key_feature[0], shape=[0, 0, num_heads, head_dim])
    k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
    attention = layers.matmul(x=q, y=k, transpose_y=True, alpha=head_dim**-0.5)
    
    attention = attention.sum(1)[:, 0]
    probas_start = paddle.nn.functional.softmax(logits[0], axis=1)  # get probabilities.
    preds_start = paddle.argmax(probas_start, axis=1)  # get predictions.

    probas_end = paddle.nn.functional.softmax(logits[1], axis=1)  # get probabilities.
    preds_end = paddle.argmax(probas_end, axis=1)  # get predictions.

    probas_start.stop_gradient = True
    probas_end.stop_gradient = True

    probas =[]
    for idx in range(len(probas_start)):#(preds_start.numpy(), preds_end.numpy())
        probas.append([probas_start[idx].numpy(), probas_end[idx].numpy()])
    preds =[]
    for idx in range(len(preds_start)):#(preds_start.numpy(), preds_end.numpy())
        preds.append([preds_start[idx].numpy(),preds_end[idx].numpy()])
    preds = np.array(preds)

    return attention.numpy(), preds, probas


def IG_predict_fn(inputs, label, left=None, right=None, steps=None, paddle_model=None,embedding_name=None):
    if paddle_model is None:
        paddle_model = self.paddle_model
    target_feature_map = []

    def hook(layer, input, output):
        if steps is not None:
            noise_array = np.arange(left, right) / steps
            output_shape = output.shape
            #assert right - left == output_shape[0]
            noise_tensor = paddle.to_tensor(noise_array, dtype=output.dtype)
            noise_tensor = noise_tensor.unsqueeze(axis=list(range(1, len(output_shape))))
            output = noise_tensor * output
        target_feature_map.append(output)
        return output

    hooks = []
    for name, v in paddle_model.named_sublayers():
        if embedding_name in name:
            h = v.register_forward_post_hook(hook)
            hooks.append(h)

    if isinstance(inputs, (tuple, list)):
        logits = paddle_model(*inputs)  # get logits, [bs, num_c]
    else:
        logits = paddle_model(inputs)  # get logits, [bs, num_c]

    bs = logits[0].shape[0]
    for h in hooks:
        h.remove()

    probas_start = paddle.nn.functional.softmax(logits[0], axis=1)  # get probabilities.
    preds_start = paddle.argmax(probas_start, axis=1)  # get predictions.

    probas_end = paddle.nn.functional.softmax(logits[1], axis=1)  # get probabilities.
    preds_end = paddle.argmax(probas_end, axis=1)  # get predictions.

    labels_onehot = paddle.nn.functional.one_hot(paddle.to_tensor(preds_start), num_classes=probas_start.shape[1])
    loss = paddle.sum(probas_start * labels_onehot, axis=1)
    labels_onehot = paddle.nn.functional.one_hot(paddle.to_tensor(preds_end), num_classes=probas_end.shape[1])
    loss += paddle.sum(probas_end * labels_onehot, axis=1)
    loss.backward()

    gradients = target_feature_map[0].grad  # get gradients of "embedding".
    loss.clear_gradient()

    if isinstance(gradients, paddle.Tensor):
        gradients = gradients.numpy()
    
    probas_start.stop_gradient = True
    probas_end.stop_gradient = True

    probas =[]
    for idx in range(len(probas_start)):#(preds_start.numpy(), preds_end.numpy())
        probas.append([probas_start[idx].numpy(), probas_end[idx].numpy()])
    preds =[]
    for idx in range(len(preds_start)):#(preds_start.numpy(), preds_end.numpy())
        preds.append([preds_start[idx].numpy()[0],preds_end[idx].numpy()[0]])
    preds = np.array(preds)
    return gradients, preds, target_feature_map[0].numpy(), probas


def trim_output(interp_results, data_ds, tokenizer):
    for i in range(len(interp_results)):
        start = data_ds[i]['input_ids'].index(tokenizer.sep_token_id)
        end = len(data_ds[i]['input_ids'])
        interp_results[i].attributions = interp_results[i].attributions[start:end]
        interp_results[i].pred_proba[0] = interp_results[i].pred_proba[0][start:end]
        interp_results[i].pred_proba[1] = interp_results[i].pred_proba[1][start:end]
    return interp_results


def pre_process(data, data_ds, tokenizer):
    # Add CLS and SEP tags to both original text and standard splited tokens
    contexts = []
    standard_split = []
    for idx in data:
        example = data[idx]
        contexts.append("[CLS]" + example['context'] + "[SEP]")
        standard_split.append(["[CLS]"] + example['sent_token'] + ["[SEP]"])

    # Get the offset map of tokenized tokens and standard splited tokens
    ori_offset_maps = []
    standard_split_offset_maps = []
    for i in range(len(contexts)):
        temp = data_ds[i]['offset_mapping'][data_ds[i]['input_ids'].index(tokenizer.sep_token_id):]
        temp[0] = (0,5)
        for j in range(1,len(temp)-1):
            temp[j] = (temp[j][0]+5,temp[j][1]+5)
        temp[-1] = (temp[-2][1], temp[-2][1]+5)
        ori_offset_maps.append(temp)

        standard_split_offset_maps.append(get_word_offset(contexts[i], standard_split[i]))
    return contexts, standard_split, ori_offset_maps, standard_split_offset_maps
