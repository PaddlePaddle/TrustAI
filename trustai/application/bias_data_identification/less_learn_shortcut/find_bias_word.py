import os
import json
import collections
import argparse

from LAC import LAC
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir",
                    default="./output",
                    type=str,
                    help="The output directory where the result will be written.")
parser.add_argument("--input_path", type=str, default=None, help="train data file path")
parser.add_argument('--num_classes', type=int, default=2, help='Number of classification.')
parser.add_argument('--cnt_threshold', type=int, default=3, help='Count threshold of statistical biased words')
parser.add_argument('--p_threshold', type=float, default=0.85, help='Probability threshold of statistical biased words')

args = parser.parse_args()


class BiasWord(object):
    """
    Statistic the biased words in the dataset
    """

    def __init__(self, segments, labels, num_classes=2, cnt_threshold=3, p_threshold=0.85):
        self.cnt_threshold = cnt_threshold
        self.p_threshold = p_threshold
        self.num_classes = num_classes
        self.segments = segments
        self.labels = labels

    def process(self):
        """
        process function
        """
        self._get_dict()
        self._search_bias_word()
        print("number of bias_words:", len(self.bias_words))
        return self.bias_words, self.bias_word_cnt, self.id2words

    def _get_dict(self):
        self.word2ids = collections.defaultdict(set)
        self.id2words = collections.defaultdict(set)
        for n, segs in enumerate(self.segments):
            for seg in segs:
                self.word2ids[seg].add(n)
            self.id2words[n] = set(segs)

    def _search_bias_word(self):
        self.bias_words = {}
        self.bias_word_cnt = {}
        for word, sentids in self.word2ids.items():
            if len(sentids) >= self.cnt_threshold:
                cnts = [0] * self.num_classes

                for sentid in sentids:
                    label = self.labels[sentid]
                    cnts[label] += 1
                assert sum(cnts) != 0
                max_cnt = max(cnts)
                p = max_cnt / sum(cnts)
                if p >= self.p_threshold:
                    self.bias_words[word] = p
                    self.bias_word_cnt[word] = len(sentids)


if __name__ == "__main__":
    # initialize tokenizer
    lac = LAC(mode='rank')

    # preprocess data, get segments、labels and lines
    segments = []
    labels = []
    lines = []
    with open(args.input_path, 'r') as f:
        for line in tqdm(list(f)):
            lines.append(line)
            query, title, label = line.strip().split('\t')
            seg_res = lac.run([query, title])
            query_segs = seg_res[0][0]
            title_segs = seg_res[1][0]
            segments.append(query_segs + title_segs)
            labels.append(int(label))

    # get bias_words
    biasword = BiasWord(segments, labels, num_classes=2, cnt_threshold=args.cnt_threshold, p_threshold=args.p_threshold)
    # b_words: biased words, dict
    # b_word_cnt: count of biased words, dict
    # id2words: sentence index to words, dict
    b_words, b_word_cnt, id2words = biasword.process()

    # save result to output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "bias_word.json"), 'w') as f:
        json.dump(b_words, f, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "bias_word_cnt.json"), 'w') as f:
        json.dump(b_word_cnt, f, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "id2words.json"), 'w') as f:
        for k, v in id2words.items():
            id2words[k] = list(v)
        json.dump(id2words, f, ensure_ascii=False)
