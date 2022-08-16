"""
utils.py
"""
import json
import random
import functools

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddlenlp
from paddlenlp.data import Stack, Tuple, Pad


def predict(model, data, tokenizer, label_map, batch_size=1):
    """
    Predicts the data labels.
    Args:
        model (obj:`paddle.nn.Layer`): A model to classify texts.
        data (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
            A Example object contains `text`(word_ids) and `se_len`(sequence length).
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        label_map(obj:`dict`): The label id (key) to label str (value) map.
        batch_size(obj:`int`, defaults to 1): The number of batch.
    Returns:
        results(obj:`dict`): All the predictions labels.
    """
    examples = []
    for text in data:
        input_ids, segment_ids = convert_example(text, tokenizer, max_seq_length=128, is_test=True)
        examples.append((input_ids, segment_ids))

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input id
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment id
    ): fn(samples)

    # Seperates data into some batches.
    batches = []
    one_batch = []
    for example in examples:
        one_batch.append(example)
        if len(one_batch) == batch_size:
            batches.append(one_batch)
            one_batch = []
    if one_batch:
        # The last batch whose size is less than the config batch_size setting.
        batches.append(one_batch)

    results = []
    model.eval()
    for batch in batches:
        input_ids, segment_ids = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        segment_ids = paddle.to_tensor(segment_ids)
        logits = model(input_ids, segment_ids)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    """
    Given a dataset, it evals model and computes the metric.
    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()
    print("eval loss: %.5f, accu: %.5f" % (np.mean(losses), accu))
    model.train()
    metric.reset()
    return accu


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
        "text", "sentence", "text_a", "text_b", "sentence1", "sentence2", "query", "title", "context"
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
        encoded_inputs = tokenizer(text=example[encoded_input_names[0]],
                                   text_pair=example[encoded_input_names[1]],
                                   max_seq_len=max_seq_length)
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


def create_dataloader(dataset, mode='train', shuffle=None, batch_size=1, batchify_fn=None, trans_fn=None):
    """create_dataloader"""
    if trans_fn:
        dataset = dataset.map(trans_fn)
    if shuffle is None:
        shuffle = True if mode == 'train' else False

    # shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=batchify_fn, return_list=True)


def training_model(model, tokenizer, train_ds, dev_ds, learning_rate=5e-5, save_dir='assets/sst-2-ernie-2.0-en'):
    """
    An example of training an NLP model.
    """
    from paddlenlp.transformers import LinearDecayWithWarmup

    print('dataset labels:', train_ds.label_list)

    print('dataset examples:')
    for data in train_ds.data[:5]:
        print(data)

    batch_size = 32
    max_seq_length = 128
    epochs = 5  #3
    warmup_proportion = 0.1
    weight_decay = 0.01

    trans_func = functools.partial(convert_example, tokenizer=tokenizer, max_seq_length=max_seq_length)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]

    train_data_loader = create_dataloader(train_ds,
                                          mode='train',
                                          batch_size=batch_size,
                                          batchify_fn=batchify_fn,
                                          trans_fn=trans_func)

    dev_data_loader = create_dataloader(dev_ds,
                                        mode='dev',
                                        batch_size=batch_size,
                                        batchify_fn=batchify_fn,
                                        trans_fn=trans_func)

    num_training_steps = len(train_data_loader) * epochs
    lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_proportion)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=weight_decay,
        apply_decay_param_fun=lambda x: x in
        [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])])

    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    best_acc = 0
    global_step = 0
    print("Training Starts:")
    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, segment_ids, labels = batch
            logits = model(input_ids, segment_ids)
            loss = criterion(logits, labels)
            probs = F.softmax(logits, axis=1)
            correct = metric.compute(probs, labels)
            metric.update(correct)
            acc = metric.accumulate()

            global_step += 1
            if global_step % 100 == 0:
                print("global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f" %
                      (global_step, epoch, step, loss, acc))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
        acc = evaluate(model, criterion, metric, dev_data_loader)
        if best_acc < acc:
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            best_acc = acc
    print("best accuracy is %f!" % best_acc)


def aggregate_subwords_and_importances(subwords, subword_importances):
    """aggregate_subwords_and_importances"""
    # words
    agg_words = []
    agg_word_importances = []
    # subwords to word
    for j, w in enumerate(subwords):
        if '##' == w[:2]:
            agg_words[-1] = agg_words[-1] + w[2:]
            agg_word_importances[-1] = agg_word_importances[-1] + subword_importances[j]
        else:
            agg_words.append(w)
            agg_word_importances.append(subword_importances[j])

    words = agg_words
    word_importances = agg_word_importances
    return words, word_importances


def set_seed(seed):
    """
    Use the same data seed(for data shuffle) for all procs to guarantee data
    consistency after sharding.
    """
    random.seed(seed)
    np.random.seed(seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(seed)


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
            Stack(dtype="int64")  # label
        ): fn(samples)
        input_ids, segment_ids, labels = batchify_fn(examples)
        return paddle.to_tensor(input_ids, stop_gradient=False), paddle.to_tensor(
            segment_ids, stop_gradient=False), paddle.to_tensor(labels, stop_gradient=False)
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

    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 lstm_hidden_size=198,
                 direction='forward',
                 lstm_layers=1,
                 dropout_rate=0.0,
                 pooling_type=None,
                 fc_hidden_size=96):
        super().__init__()
        self.embedder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=padding_idx)
        self.lstm_encoder = paddlenlp.seq2vec.LSTMEncoder(emb_dim,
                                                          lstm_hidden_size,
                                                          num_layers=lstm_layers,
                                                          direction=direction,
                                                          dropout=dropout_rate,
                                                          pooling_type=pooling_type)
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


def load_data(file_path):
    """load data"""
    data = {}

    with open(file_path, 'r') as f:
        for line in f.readlines():
            if line.strip():
                example = json.loads(line)
                data[example[list(example.keys())[0]]] = example
    return data


def print_result(test_data, train_ds, res, data_name='chnsenticorp'):
    """
    print result
    """
    if data_name == 'chnsenticorp':
        for i in range(len(test_data)):
            print("test data")
            print(f"text: {test_data[i]['text']}\tpredict label: {res[i].pred_label}")
            print("examples with positive influence")
            for example, score in zip(res[i].pos_indexes, res[i].pos_scores):
                print(
                    f"text: {train_ds.data[example]['text']}\tgold label: {train_ds.data[example]['label']}\tscore: {score}"
                )
            print("examples with negative influence")
            for example, score in zip(res[i].neg_indexes, res[i].neg_scores):
                print(
                    f"text: {train_ds.data[example]['text']}\tgold label: {train_ds.data[example]['label']}\tscore: {score}"
                )
            print()
    elif data_name == 'qqp':
        for i in range(len(test_data)):
            print("test data")
            print(f"text: {test_data[i]['sentence1']}\t{test_data[i]['sentence2']}\tpredict label: {res[i].pred_label}")
            print("examples with positive influence")
            for example, score in zip(res[i].pos_indexes, res[i].pos_scores):
                print(
                    f"text: {train_ds.data[example]['sentence1']}\t{train_ds.data[example]['sentence2']}\tgold label: {train_ds.data[example]['labels']}\tscore: {score}"
                )
            print("examples with negative influence")
            for example, score in zip(res[i].neg_indexes, res[i].neg_scores):
                print(
                    f"text: {train_ds.data[example]['sentence1']}\t{train_ds.data[example]['sentence2']}\tgold label: {train_ds.data[example]['labels']}\tscore: {score}"
                )
            print()
    elif data_name == 'lcqmc':
        for i in range(len(test_data)):
            print("test data")
            print(f"text: {test_data[i]['query']}\t{test_data[i]['title']}\tpredict label: {res[i].pred_label}")
            print("examples with positive influence")
            for example, score in zip(res[i].pos_indexes, res[i].pos_scores):
                print(
                    f"text: {train_ds.data[example]['query']}\t{train_ds.data[example]['title']}\tgold label: {train_ds.data[example]['label']}\tscore: {score}"
                )
            print("examples with negative influence")
            for example, score in zip(res[i].neg_indexes, res[i].neg_scores):
                print(
                    f"text: {train_ds.data[example]['query']}\t{train_ds.data[example]['title']}\tgold label: {train_ds.data[example]['label']}\tscore: {score}"
                )
            print()
