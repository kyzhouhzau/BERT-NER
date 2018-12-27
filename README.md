# BERT-NER
Use google BERT to do CoNLL-2003 NER !


Try to implement NER work based on google's BERT code!

First <code>git clone https://github.com/google-research/bert.git</code>

Second <code>download file in this project</code>

    BERT
    |____ bert
    |____ BERT_NER.py
    |____ checkpoint
    |____ output


Third run:
```
  python3 BERT_NER.py   \
                    --task_name="NER"  \
                    --do_train=True   \
                    --do_eval=True   \
                    --do_predict=True \
                    --data_dir=NERdata/en   \
                    --column_sep=" " \
                    --vocab_file=../BERT-BiLSTM-CRF-NER/uncased_L-12_H-768_A-12/vocab.txt  \
                    --bert_config_file=../BERT-BiLSTM-CRF-NER/uncased_L-12_H-768_A-12/bert_config.json \
                    --init_checkpoint=../BERT-BiLSTM-CRF-NER/uncased_L-12_H-768_A-12/bert_model.ckpt   \
                    --max_seq_length=32   \
                    --train_batch_size=16   \
                    --learning_rate=2e-5   \
                    --num_train_epochs=6   \
                    --save_checkpoints_steps=1000  \
                    --output_dir=./output_en/result_dir/
 ```       

result:

The predicted result is placed in folder <code>./output/result_dir/</code>. It contain two files, <code>token_test.txt</code> is the tokens and <code>predicted_results.tx</code> is the labels for each token. If you want a more accurate evaluation result you can use script conlleval.pl for evaluation.

The following evaluation results differ from the evaluation results specified by conll2003. 

![](/picture.png)


#### 注：For the parameters of the above model, I have not made any modifications. All parameters are based on the BERT default parameters. The better parameters for this problem can be adjusted by yourselves.

The f_score evaluation codes come from:https://github.com/guillaumegenthial/tf_metrics/blob/master/tf_metrics/__init__.py

reference:
+ [https://github.com/google-research/bert](https://github.com/google-research/bert)
      
+ [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

