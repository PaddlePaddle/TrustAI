#!/bin/bash
data_dir="./data/robust"
output_dir="./output"
load_selector_model_dir="./"
load_predictor_model_dir="./"
split="dev"

ARGS=`getopt -a -o :d:o:s:p:S:h -l data_dir:,output_dir:,selector_model_dir:,predictor_model_dir:,split:,help -- "$@"`
eval set -- "${ARGS}"
while true
do
      case "$1" in
      -d|--data_dir)
              data_dir="$2"
              shift
              ;;
      -o|--output_dir)
              output_dir="$2"
              shift
              ;;
      -s|--selector_model_dir)
              load_selector_model_dir="$2"
              shift
              ;;
      -p|--predictor_model_dir)
              load_predictor_model_dir="$2"
              shift
              ;;
      -S|--split)
              split="$2"
              shift
              ;;
      -h|--help)
              echo "help"
              ;;
      --)
              shift
              break
              ;;
      esac
shift
done 

if [ ! -d "./cache" ]; then
  mkdir "./cache"
fi
if [ ! -d "${output_dir}" ]; then
  mkdir "${output_dir}"
fi
if [ ! -d "${output_dir}/selected-test-data" ]; then
  mkdir "${output_dir}/selected-test-data"
fi
if [ $split = "dev" ]; then
  if [ ! -d "${output_dir}/tmp" ]; then
    mkdir "${output_dir}/tmp"
  fi
  cp "${data_dir}/dev.json" "${output_dir}/tmp/test.json"
  echo "Data path: ${data_dir}/dev.json"
  data_dir="${output_dir}/tmp"
else
  echo "Data path: ${data_dir}/test.json"
fi

echo "Output path: ${output_dir}/selected-test-data"

python -u run_select.py \
    --model_name hfl/roberta-wwm-ext \
    --do_predict \
    --batch_size 24 \
    --data_dir $data_dir \
    --load_model_path "${load_selector_model_dir}/model_state.pdparams" \
    --one_alpha 0.1 \
    --output_dir "${output_dir}/selected-test-data/" \
    --device gpu

mv "${output_dir}/selected-test-data/test_prediction.json" "${output_dir}/selected-test-data/test.json"

python -u run_predict.py \
    --model_name hfl/roberta-wwm-ext \
    --do_predict \
    --batch_size 24 \
    --max_seq_length 384 \
    --data_dir "${output_dir}/selected-test-data/" \
    --load_model_path "${load_predictor_model_dir}/model_state.pdparams" \
    --output_dir "${output_dir}/predict-result/" \
    --device gpu


