@echo off
py -3.6 BERT_NER.py ^
--task_name="NER" ^
--do_train=True ^
--do_eval=True ^
--do_predict=True ^
--data_dir=NERdata ^
--vocab_file=uncased_L-12_H-768_A-12/vocab.txt ^
--bert_config_file=uncased_L-12_H-768_A-12/bert_config.json ^
--init_checkpoint=uncased_L-12_H-768_A-12/bert_model.ckpt ^
--max_seq_length=64 ^
--train_batch_size=32 ^
--learning_rate=2e-5 ^
--num_train_epochs=3.0 ^
--output_dir=./output/hihi/ 