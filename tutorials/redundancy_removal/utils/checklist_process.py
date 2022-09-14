import json
import tools
import argparse
import os


def process(input_data_path, output_data_path):
    with open(input_data_path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    for i, data in enumerate(obj['data']):
        x = []
        for j, paragraphs in enumerate(data["paragraphs"]):
            for k, ans in enumerate(paragraphs["qas"][0]["answers"]):
                answer = ans["text"]
                if answer != "" and len(tools.split_sentence(answer)) == 1:
                    x.append(paragraphs)
                else:
                    break
        obj["data"][i]["paragraphs"] = x

    with open(output_data_path, 'w+', encoding="utf8") as outfile:
        json.dump(obj, outfile, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_data_dir", type=str, required=True, help="checklist input data dir")
    parser.add_argument("--output_data_dir", type=str, required=True, help="checklist output data dir")
    args = parser.parse_args()
    process(os.path.join(args.input_data_dir, 'train.json'), os.path.join(args.output_data_dir, 'train.json'))
    process(os.path.join(args.input_data_dir, 'dev.json'), os.path.join(args.output_data_dir, 'dev.json'))
