#!/usr/bin/env bash

# MODEL_TYPE = bert, xlnet, xlm, roberta, distilbert
# MODEL_NAME = bert-base-uncased, bert-large-uncased, bert-large-uncased-whole-word-masking

export TASK_NAME=rappler_vad
export DATA_DIR=data/rappler_vad
export MODEL_TYPE=bert_rnn_head
export MODEL_NAME=bert-base-uncased

conda activate emotion

python task_finetuning.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL_NAME \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $DATA_DIR \
  --save_steps 9999 \
  --max_seq_length 256 \
  --per_gpu_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir output/$TASK_NAME \
  --overwrite_output_dir \
  --overwrite_cache