#!/bin/sh
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
BERT_BASE_DIR=bert/model_chinese
DATA_DIR=CGED/char_level
TASK=cged

sudo python3 BERT_NER.py \
  --task_name=%TASK% \
  --do_train=true \
  --do_eval=true \
  --do_predict=false \
  --data_dir=%DATA_DIR% \
  --vocab_file=%BERT_BASE_DIR%/vocab.txt \
  --bert_config_file=%BERT_BASE_DIR%/bert_config.json \
  --init_checkpoint=%BERT_BASE_DIR%/bert_model.ckpt \
  --max_seq_length=16 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=output/%TASK%/