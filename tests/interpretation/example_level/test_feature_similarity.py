# !/usr/bin/env python3
import os
import sys
import unittest
from functools import partial

import paddle
import numpy as np
from paddlenlp.data import Stack, Tuple, Pad, Vocab, JiebaTokenizer
from paddlenlp.datasets import load_dataset, MapDataset
from paddlenlp.transformers import ErnieForSequenceClassification, ErnieTokenizer

sys.path.insert(0, '../')
sys.path.insert(0, '../../../')
from assets.utils import (
    create_dataloader,
    convert_example,
    create_dataloader_from_scratch,
    LSTMModel,
    preprocess_fn_lstm,
    get_sublayer,
)
from trustai.interpretation.example_level.method.feature_similarity import (
    FeatureSimilarityModel, )


class TestFeatureSimilarity(unittest.TestCase):

    def test_bert_model(self):
        MODEL_NAME = "ernie-1.0"
        DATASET_NAME = "chnsenticorp"
        paddle_model = ErnieForSequenceClassification.from_pretrained(MODEL_NAME, num_classes=2)
        tokenizer = ErnieTokenizer.from_pretrained(MODEL_NAME)
        state_dict = paddle.load(f"../assets/{DATASET_NAME}-{MODEL_NAME}/model_state.pdparams")
        paddle_model.set_dict(state_dict)

        train_ds, dev_ds, test_ds = load_dataset(DATASET_NAME, splits=["train", "dev", "test"])

        batch_size = 32
        max_seq_length = 128

        trans_func = partial(
            convert_example,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            is_test=True,
        )
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        ): [data for data in fn(samples)]

        train_data_loader = create_dataloader(
            train_ds,
            mode="train",
            batch_size=batch_size,
            batchify_fn=batchify_fn,
            trans_fn=trans_func,
            shuffle=False,
        )

        feature_sim_model = FeatureSimilarityModel(paddle_model, train_data_loader, classifier_layer_name="classifier")

    def test_predict_fn(self):

        def predict_fn(inputs, paddle_model, classifier_layer_name="classifier"):
            """predict_fn"""

            x_feature = []

            def forward_pre_hook(layer, input):
                """
                Hook for a given layer in model.
                """
                x_feature.extend(input[0])

            classifier = get_sublayer(paddle_model, classifier_layer_name)

            forward_pre_hook_handle = classifier.register_forward_pre_hook(forward_pre_hook)

            if isinstance(inputs, (tuple, list)):
                logits = paddle_model(*inputs)  # get logits, [bs, num_c]
            else:
                logits = paddle_model(inputs)  # get logits, [bs, num_c]

            forward_pre_hook_handle.remove()

            probas = paddle.nn.functional.softmax(logits, axis=1)  # get probabilities.
            preds = paddle.argmax(probas, axis=1)  # get predictions.
            x_feature = paddle.to_tensor(x_feature)
            return x_feature, probas, preds

        MODEL_NAME = "ernie-1.0"
        DATASET_NAME = "chnsenticorp"
        paddle_model = ErnieForSequenceClassification.from_pretrained(MODEL_NAME, num_classes=2)
        tokenizer = ErnieTokenizer.from_pretrained(MODEL_NAME)
        state_dict = paddle.load(f"../assets/{DATASET_NAME}-{MODEL_NAME}/model_state.pdparams")
        paddle_model.set_dict(state_dict)

        train_ds, dev_ds, test_ds = load_dataset(DATASET_NAME, splits=["train", "dev", "test"])

        batch_size = 32
        max_seq_length = 128

        trans_func = partial(
            convert_example,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            is_test=True,
        )
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
            Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        ): [data for data in fn(samples)]

        predict_fn_test = partial(predict_fn, paddle_model=paddle_model)

        train_data_loader = create_dataloader(
            train_ds,
            mode="train",
            batch_size=batch_size,
            batchify_fn=batchify_fn,
            trans_fn=trans_func,
            shuffle=False,
        )

        feature_sim_model = FeatureSimilarityModel(
            paddle_model,
            train_data_loader,
            classifier_layer_name="classifier",
            predict_fn=predict_fn_test,
        )

    def test_lstm_model(self):
        PARAMS_PATH = "../assets/chnsenticorp-bilstm/final.pdparams"
        VOCAB_PATH = "../assets/chnsenticorp-bilstm/bilstm_word_dict.txt"
        vocab = Vocab.from_json(VOCAB_PATH)
        tokenizer = JiebaTokenizer(vocab)
        label_map = {0: "negative", 1: "positive"}
        vocab_size = len(vocab)
        num_classes = len(label_map)
        pad_token_id = vocab.to_indices("[PAD]")

        DATASET_NAME = "chnsenticorp"
        paddle_model = LSTMModel(vocab_size, num_classes, direction="bidirect", padding_idx=pad_token_id)
        state_dict = paddle.load(PARAMS_PATH)
        paddle_model.set_dict(state_dict)

        train_ds, dev_ds, test_ds = load_dataset(DATASET_NAME, splits=["train", "dev", "test"])

        # train_ds = [d['text'] for d in list(train_ds)[:1200]]
        # train_ds = [d["text"] for d in list(train_ds)]
        # train_ds = MapDataset(train_ds)

        batch_size = 32
        max_seq_length = 128

        trans_func = partial(preprocess_fn_lstm, tokenizer=tokenizer, is_test=True)
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=pad_token_id),  # input
            Pad(axis=0, pad_val=pad_token_id),  # sequence_length
        ): [data for data in fn(samples)]

        train_data_loader = create_dataloader(
            train_ds,
            mode="train",
            batch_size=batch_size,
            batchify_fn=batchify_fn,
            trans_fn=trans_func,
            shuffle=False,
        )

        feature_sim_model = FeatureSimilarityModel(paddle_model,
                                                   train_data_loader,
                                                   classifier_layer_name="output_layer")


if __name__ == "__main__":
    unittest.main()
