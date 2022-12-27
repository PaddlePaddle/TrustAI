###
 # This script is used to finetune pretrained models
###

export CUDA_VISIBLE_DEVICES=3
LANGUAGE="ch"
timestamp=`date  +"%Y%m%d_%H%M%S"`
data_dir='./'
LEARNING_RATE=3e-5
MAX_SEQ_LENGTH=256

[ -d "logs" ] || mkdir -p "logs"
[ -d "outputs" ] || mkdir -p "outputs"
set -x

train_file=sample_100.tsv
dev_file=$train_file
train_size=100

batch_size=32
epoch=5
save_model_num=5
epoch_steps=$[$train_size/$batch_size]
save_steps=$[$epoch_steps*$epoch/${save_model_num}]

python3 ./train_pointwise.py  \
    --learning_rate $LEARNING_RATE \
    --max_seq_length $MAX_SEQ_LENGTH \
    --batch_size ${batch_size} \
    --epochs ${epoch} \
    --data_dir $data_dir \
    --train_set ${train_file} \
    --dev_set ${dev_file} \
    --eval_step ${save_steps} \
    --warmup_proportion 0.1 \
    --save_dir saved_model/${timestamp} >> logs/log_${timestamp}
    
