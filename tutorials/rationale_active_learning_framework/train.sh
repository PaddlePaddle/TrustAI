
data_dir="./data/robust"
output_dir="./output"

# Read Parameter
# -d data dir path
# -o output_dir
while getopts ":d:o:" optname
do
    case "$optname" in
      "d")
        data_dir=$OPTARG
        ;;
      "o")
        output_dir=$OPTARG
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

echo "Data path: ${data_dir}"
echo "Selector Output path: ${output_dir}/selector/"

# prepare dir

# clean cache
if [ ! -d "./cache" ]; then
  mkdir "./cache"
else
  rm  -rf "./cache"
  mkdir "./cache"
fi
if [ ! -d "${output_dir}" ]; then
  mkdir "${output_dir}"
fi
if [ ! -d "${output_dir}/selector" ]; then
  mkdir "${output_dir}/selector"
fi
if [ ! -d "${output_dir}/selected-data" ]; then
  mkdir "${output_dir}/selected-data"
fi
if [ ! -d "${output_dir}/predictor" ]; then
  mkdir "${output_dir}/predictor"
fi
if [ ! -d "${output_dir}/tmp" ]; then
  mkdir "${output_dir}/tmp"
fi

echo "########## Selector Training #############"

# train selector
python -u ./run_select.py \
    --model_name hfl/roberta-wwm-ext \
    --max_seq_length 512 \
    --batch_size 24 \
    --learning_rate 8e-5 \
    --num_train_epochs 100 \
    --logging_steps 10 \
    --save_steps 200 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir "${output_dir}/selector/" \
    --data_dir ${data_dir} \
    --set_k_sentences_ground_true 0 \
    --early_stop_nums 5 \
    --one_alpha -1 \
    --do_train \
    --use_loose_metric \
    --early_stop \
    --device gpu


echo "########## Dev Processing #############"
cp –f "${data_dir}/dev.json" "${output_dir}/tmp"
mv "${output_dir}/tmp/dev.json" "${output_dir}/tmp/test.json"

# predict selector
python -u ./run_select.py \
    --model_name hfl/roberta-wwm-ext \
    --max_seq_length 512 \
    --batch_size 24 \
    --load_model_path "${output_dir}/selector/best_model/model_state.pdparams" \
    --data_dir "${output_dir}/tmp/" \
    --output_dir "${output_dir}/selector/" \
    --set_k_sentences_ground_true 0 \
    --one_alpha 0.1 \
    --do_predict \
    --use_loose_metric \
    --device gpu

rm -rf "${output_dir}/tmp"
# Postprocess selected data
temp_dir="${data_dir}/*"
cp -f ${temp_dir} "${output_dir}/selected-data"
rm -f "${output_dir}/selected-data/dev.json"
origin_dev_path="${output_dir}/selector/test_prediction.json"
mv ${origin_dev_path} "${output_dir}/selected-data/dev.json"

echo "########## Predictor Training #############"

# Train predictor
python -u run_predict.py \
    --model_name hfl/roberta-wwm-ext \
    --max_seq_length 384 \
    --batch_size 24 \
    --learning_rate 5e-5 \
    --num_train_epochs 20 \
    --logging_steps 10 \
    --save_steps 100 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir "$output_dir/predictor/" \
    --data_dir "$output_dir/selected-data/" \
    --do_train \
    --device gpu

python -u run_predict.py \
    --model_name hfl/roberta-wwm-ext \
    --max_seq_length 384 \
    --batch_size 24 \
    --learning_rate 5e-5 \
    --num_train_epochs 8 \
    --logging_steps 10 \
    --save_steps 200 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir "$output_dir/predictor/" \
    --data_dir "$output_dir/selected-data/" \
    --do_train \
    --device gpu