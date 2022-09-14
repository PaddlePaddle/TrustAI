
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