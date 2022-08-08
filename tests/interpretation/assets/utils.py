"""
unit test
"""
import unittest

import paddle
import numpy as np
import paddle.nn as nn
import paddlenlp
from paddlenlp.data import Stack, Tuple, Pad


def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
    """
    Builds model inputs from a sequence or a pair of sequence for sequence classification tasks
    by concatenating and adding special tokens. And creates a mask from the two sequences passed
    to be used in a sequence-pair classification task.

    A BERT sequence has the following format:
    - single sequence: ``[CLS] X [SEP]``
    - pair of sequences: ``[CLS] A [SEP] B [SEP]``
    A BERT sequence pair mask has the following format:
    ::
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
    If only one sequence, only returns the first portion of the mask (0's).
    Args:
        example(obj:`list[str]`): List of input data, containing text and label if it have label.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer`
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization.
            Sequences longer than this will be truncated, sequences shorter will be padded.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.
    Returns:
        input_ids(obj:`list[int]`): The list of token ids.
        token_type_ids(obj: `list[int]`): List of sequence pair mask.
        label(obj:`numpy.array`, data type of int64, optional): The input label if not is_test.
    """
    possible_encoded_input_names = [
        "text",
        "sentence",
        "text_a",
        "text_b",
        "sentence1",
        "sentence2",
        "query",
        "title",
        "context",
    ]
    possible_label_names = ["label", "labels"]

    # Search a possible name
    encoded_input_names = []
    for n in possible_encoded_input_names:
        if n in example:
            encoded_input_names.append(n)

    encoded_label_name = None
    for n in possible_label_names:
        if n in example:
            encoded_label_name = n
            break
    if len(encoded_input_names) == 1:
        encoded_inputs = tokenizer(text=example[encoded_input_names[0]], max_seq_len=max_seq_length)
    elif len(encoded_input_names) == 2:
        encoded_inputs = tokenizer(
            text=example[encoded_input_names[0]],
            text_pair=example[encoded_input_names[1]],
            max_seq_len=max_seq_length,
        )
    else:
        raise ValueError("error input_names.")

    # encoded_inputs = tokenizer(text=example[encoded_input_name], max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example[encoded_label_name]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids


def create_dataloader(dataset, mode="train", shuffle=None, batch_size=1, batchify_fn=None, trans_fn=None):
    """create_dataloader"""
    if trans_fn:
        dataset = dataset.map(trans_fn)
    if shuffle is None:
        shuffle = True if mode == "train" else False
    # shuffle = True if mode == 'train' else False
    if mode == "train":
        batch_sampler = paddle.io.DistributedBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True,
    )


def preprocess_fn(data, tokenizer, with_label=False):
    """
    Preprocess input data to satisfy the demand of a given model.
    Args:
        data(obj:`list[str]`): List of input data, containing text and label if it have label.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer`
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
    Returns:
        input_ids(obj:`list[int]`): The list of token ids.
        token_type_ids(obj: `list[int]`): List of sequence pair mask.
    """
    examples = []
    if with_label:
        for text in data:
            input_ids, segment_ids, labels = convert_example(text, tokenizer, max_seq_length=128, is_test=False)
            examples.append((input_ids, segment_ids, labels))

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input id
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment id
            Stack(dtype="int64"),  # label
        ): fn(samples)
        input_ids, segment_ids, labels = batchify_fn(examples)
        return (
            paddle.to_tensor(input_ids, stop_gradient=False),
            paddle.to_tensor(segment_ids, stop_gradient=False),
            paddle.to_tensor(labels, stop_gradient=False),
        )
    else:
        for text in data:
            input_ids, segment_ids = convert_example(text, tokenizer, max_seq_length=128, is_test=True)
            examples.append((input_ids, segment_ids))

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input id
            Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment id
        ): fn(samples)
        input_ids, segment_ids = batchify_fn(examples)
        return paddle.to_tensor(input_ids, stop_gradient=False), paddle.to_tensor(segment_ids, stop_gradient=False)


def get_batches(data, batch_size=1):
    """
    seperate data into batches according to the batch_size.
    """
    batches = []
    one_batch = []
    for example in data:
        one_batch.append(example)
        if len(one_batch) == batch_size:
            batches.append(one_batch)
            one_batch = []
    if one_batch:
        # The last batch whose size is less than the config batch_size setting.
        batches.append(one_batch)
    return batches


def create_dataloader_from_scratch(data, tokenizer, batch_size=1, with_label=False):
    """
    Create dataloader from scratch.
    """
    dataloader = []
    # Seperates data into some batches.
    batches = get_batches(data, batch_size=batch_size)
    dataloader = [preprocess_fn(batch, tokenizer, with_label=with_label) for batch in batches]
    return dataloader


class LSTMModel(nn.Layer):
    """LSTMModel"""

    def __init__(
        self,
        vocab_size,
        num_classes,
        emb_dim=128,
        padding_idx=0,
        lstm_hidden_size=198,
        direction="forward",
        lstm_layers=1,
        dropout_rate=0.0,
        pooling_type=None,
        fc_hidden_size=96,
    ):
        super().__init__()
        self.embedder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=padding_idx)
        self.lstm_encoder = paddlenlp.seq2vec.LSTMEncoder(
            emb_dim,
            lstm_hidden_size,
            num_layers=lstm_layers,
            direction=direction,
            dropout=dropout_rate,
            pooling_type=pooling_type,
        )
        self.fc = nn.Linear(self.lstm_encoder.get_output_dim(), fc_hidden_size)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text, seq_len):
        """forward"""
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens, num_directions*lstm_hidden_size)
        # num_directions = 2 if direction is 'bidirect'
        # if not, num_directions = 1
        text_repr = self.lstm_encoder(embedded_text, sequence_length=seq_len)
        # Shape: (batch_size, fc_hidden_size)
        fc_out = paddle.tanh(self.fc(text_repr))
        # Shape: (batch_size, num_classes)
        logits = self.output_layer(fc_out)
        return logits


def preprocess_fn_lstm(text, tokenizer, is_test=False):
    """
    preprocess function for lstm-based model.
    """
    ids = tokenizer.encode(text["text"])
    texts = ids
    seq_lens = len(ids)
    if not is_test:
        label = np.array([text["label"]], dtype="int64")
        return texts, seq_lens, label
    else:
        return texts, seq_lens


def get_sublayer(model, sublayer_name="classifier"):
    """
    Get the sublayer named sublayer_name in model.
    Args:
        model (obj:`paddle.nn.Layer`): Any paddle model.
        sublayer_name (obj:`str`, defaults to classifier): The sublayer name.
    Returns:
        layer(obj:`paddle.nn.Layer.common.sublayer_name`):The sublayer named sublayer_name in model.
    """
    for name, layer in model.named_children():
        if name == sublayer_name:
            return layer
