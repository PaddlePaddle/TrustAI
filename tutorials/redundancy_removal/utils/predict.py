import argparse
import json

from paddlenlp.metrics.squad import squad_evaluate


def get_data(filepath):
    with open(filepath, encoding="utf-8") as f:
        durobust = json.load(f)
    data = []
    for article in durobust["data"]:
        title = article.get("title", "")
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]  # do not strip leading blank spaces GH-2585
            for qa in paragraph["qas"]:
                answer_starts = [answer["answer_start"] for answer in qa.get("answers", '')]
                answers = [answer["text"] for answer in qa.get("answers", '')]
                # Features currently used are "context", "question", and "answers".
                # Others are extracted here for the ease of future expansions.
                data.append({
                    "title": title,
                    "context": context,
                    "question": qa["question"],
                    "id": qa["id"],
                    "answers": {
                        "answer_start": answer_starts,
                        "text": answers,
                    },
                })
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_data_dir", type=str, required=True, help="test data dir")
    parser.add_argument("--pred_data_dir", type=str, required=True, help="prediction data dir")
    args = parser.parse_args()
    raw_datasets = get_data(args.test_data_dir)

    with open(args.pred_data_dir, encoding="utf-8") as f:
        all_predictions = json.load(f)

    result = squad_evaluate(examples=raw_datasets, preds=all_predictions, is_whitespace_splited=False)
