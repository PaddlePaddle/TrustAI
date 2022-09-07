data_dir="./data/robust"
output_dir="./output"
load_selector_model_dir="./"
load_predictor_model_dir="./"
# Read Parameter
# -d data dir path
# -o output_dir

# sh test.sh -d ./toy -o ./output -s ./output/selector/best_model -p ./output/predictor/model_20
while getopts ":d:o:s:p:" optname
do
    case "$optname" in
      "d")
        data_dir=$OPTARG
        ;;
      "o")
        output_dir=$OPTARG
        ;;    
      "s")
        load_selector_model_dir=$OPTARG
        ;;    
      "p")
        load_predictor_model_dir=$OPTARG
        ;;    
      ":")
        echo "No argument value for option $OPTARG"
        ;;
      "?")
        echo "Unknown option $OPTARG"
        ;;
      *)
        echo "Unknown error while processing options"
        ;;
    esac
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
echo "Data path: ${data_dir}"
echo "Output path: ${output_dir}"

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
