import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default=None, help="input data file path")
parser.add_argument("--output_path", type=str, default=None, help="output data file path")
parser.add_argument("--bias_dir", type=str, default=None, help="bias data directory.")
parser.add_argument("--stopwords_path", type=str, default=None, help="stopwords data file path")
parser.add_argument('--num_classes', type=int, default=2, help='Number of classification.')
parser.add_argument('--alpha',
                    type=float,
                    default=0.01,
                    help='Hyperparameters for frequency of words when mode is lls_d_f.')
parser.add_argument('--mode',
                    type=str,
                    default='lls_d',
                    choices=['lls_d', 'lls_d_f'],
                    help='Hyperparameters for frequency of words.')

args = parser.parse_args()


def filter_stopwords(score_w, stop_words):
    for word in list(score_w.keys()):
        if word in stop_words:
            del score_w[word]
    return score_w


def word_score(d, num_classes):
    score_w = {}
    for k in d.keys():
        score_w[k] = abs(d[k] - 1 / num_classes)
    return score_w


def word_score_freq(d, d_cnt, num_classes, alpha):
    score_w = {}
    for k in d.keys():
        score_w[k] = abs(d[k] - 1 / num_classes) + alpha * d_cnt[k]
    return score_w


def lls_basic(score_w, id2words):
    sample_bias = {}
    for n in range(len(id2words)):

        sample_score = 0
        cnt = 0
        for word in id2words[str(n)]:
            if word in score_w:
                sample_score += score_w[word]
                cnt += 1
        if cnt != 0:
            sample_bias[n] = sample_score / cnt
    return sample_bias


def softxmax(sample_bias, a=0, b=0.15):
    """
    Score normalization
    """
    scores = []
    for k, v in sample_bias.items():
        scores.append(v)
    maxn, minn = max(scores), min(scores)
    sample_bias_norm = {}
    for k, sc in sample_bias.items():
        sc_softmax = a + (b - a) / (maxn - minn) * (sc - minn)
        sample_bias_norm[k] = (1 - sc_softmax)
    return sample_bias_norm


if __name__ == "__main__":

    # load data
    with open(args.stopwords_path, 'r') as f:
        stop_words = []
        for line in f.readlines():
            stop_words.append(line.strip())
    with open(os.path.join(args.bias_dir, 'id2words.json'), 'r') as f:
        id2words = json.load(f)
    with open(os.path.join(args.bias_dir, 'bias_word.json'), 'r') as f:
        d = json.load(f)
    with open(os.path.join(args.bias_dir, 'bias_word_cnt.json'), 'r') as f:
        d_cnt = json.load(f)
    with open(args.input_path, 'r') as f:
        lines = list(f)

    # get bias degree for example
    mode = args.mode
    if mode == 'lls_d':
        score_w = word_score(d, num_classes=2)
        score_w = filter_stopwords(score_w, stop_words)
        sample_bias = lls_basic(score_w, id2words)
        sample_bias_norm = softxmax(sample_bias)
    elif mode == 'lls_d_f':
        score_w = word_score_freq(d, d_cnt, num_classes=args.num_classes, alpha=args.alpha)
        score_w = filter_stopwords(score_w, stop_words)
        sample_bias = lls_basic(score_w, id2words)
        sample_bias_norm = softxmax(sample_bias)
    else:
        raise KeyError(f"Unknown mode: {mode}, mode should be chosen from [lls_d, lls_d_f].")

    # save result
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for n, line in enumerate(lines):
            if n in sample_bias_norm:
                f.write(line.strip() + '\t' + str(sample_bias_norm[n]) + '\n')
            else:
                f.write(line.strip() + '\t' + str(1) + '\n')