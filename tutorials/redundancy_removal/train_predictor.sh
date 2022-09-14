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

echo "########## Predictor Training #############"

# Train predictor
python -u run_predict.py \
    --model_name hfl/roberta-wwm-ext \
    --max_seq_length 384 \
    --batch_size 24 \
    --learning_rate 5e-5 \
    --num_train_epochs 8 \
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
    --do_predict \
    --device gpu