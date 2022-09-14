#!/bin/bash
# Download dataset and model parameters
set -e

if [ ! -d "./data" ]; then
  mkdir "./data"
fi

echo "Download DuReader-robust dataset"
wget --no-check-certificate https://dataset-bj.cdn.bcebos.com/dureader_robust/data/dureader_robust-data.tar.gz 
tar -zxvf dureader_robust-data.tar.gz 
mv dureader_robust-data data/robust
rm dureader_robust-data.tar.gz

echo "Download DuReader-checklist dataset"
wget --no-check-certificate https://dataset-bj.cdn.bcebos.com/lic2021/dureader_checklist.dataset.tar.gz
tar -zxvf dureader_checklist.dataset.tar.gz
mv dataset data/checklist
rm dureader_checklist.dataset.tar.gz
mkdir ./data/checklist_wo_no_answer
python ./utils/checklist_process.py --input_data_dir ./data/checklist --output_data_dir ./data/checklist_wo_no_answer