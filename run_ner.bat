@echo off
SET BERT_BASE_DIR=model_chinese
SET DATA_DIR=NERdata/
SET TASK=ner
py -3.6 BERT_NER.py ^
  --task_name=%TASK% ^
  --do_train=true ^
  --do_eval=true ^
  --do_predict=true ^
  --data_dir=%DATA_DIR% ^
  --vocab_file=%BERT_BASE_DIR%/vocab.txt ^
  --bert_config_file=%BERT_BASE_DIR%/bert_config.json ^
  --init_checkpoint=%BERT_BASE_DIR%/bert_model.ckpt ^
  --max_seq_length=64 ^
  --train_batch_size=32 ^
  --learning_rate=2e-3 ^
  --num_train_epochs=3.0 ^
  --output_dir=output/%TASK%/