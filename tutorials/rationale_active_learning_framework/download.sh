#!/bin/bash
# Download dataset and model parameters
set -e

echo "Download DuReader-robust dataset"
mkdir data
wget --no-check-certificate https://dataset-bj.cdn.bcebos.com/dureader_robust/data/dureader_robust-data.tar.gz 
tar -zxvf dureader_robust-data.tar.gz 
mv dureader_robust-data data/robust
rm dureader_robust-data.tar.gz

echo "Download DuReader-checklist dataset"
wget --no-check-certificate https://dataset-bj.cdn.bcebos.com/lic2021/dureader_checklist.dataset.tar.gz
tar -zxvf dureader_checklist.dataset.tar.gz
mv dataset data/checklist
rm dureader_checklist.dataset.tar.gz