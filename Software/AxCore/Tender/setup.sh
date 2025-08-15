#!/bin/bash

# dataset
# wget https://huggingface.co/datasets/mit-han-lab/pile-val-backup/resolve/main/val.jsonl.zst
# mkdir -p ./data
# mv val.jsonl.zst ./data

# model setup

CWD=${PWD}
cd transformers/src/transformers/models

for model in llama opt;do
  mv ${model}/modeling_${model}.py ${model}/modeling_${model}_orig.py
  ln -s ${CWD}/models/modeling_${model}.py ${model}/modeling_${model}.py
done
