# BERT-NER
Use google BERT to do CoNLL-2003 NER !


Try to implement NER work based on google's BERT code!

First <code>git clone https://github.com/google-research/bert.git</code>

Second <code>download file in this project</code>

Third download bert snapshot, extract and rename folder checkpoint

    BERT
    |____ bert
    |____ NERdata
    |____ BERT_NER.py
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

The predicted result is placed in folder <code>./output/result_dir/</code>. It is in the file <code>predicted_results.tx</code>. 

The following evaluation results come from script conlleval.pl. 

 ```  
accuracy:  98.09%; precision:  90.01%; recall:  91.51%; FB1:  90.76
              LOC: precision:  91.81%; recall:  93.58%; FB1:  92.69  1698
             MISC: precision:  75.73%; recall:  81.34%; FB1:  78.43  754
              ORG: precision:  88.59%; recall:  89.70%; FB1:  89.14  1682
              PER: precision:  96.62%; recall:  95.67%; FB1:  96.14  1599
 ```  

#### 注：For the parameters of the above model, I have not made any modifications. All parameters are based on the BERT default parameters. The better parameters for this problem can be adjusted by yourselves.

The f_score evaluation codes come from:https://github.com/guillaumegenthial/tf_metrics/blob/master/tf_metrics/__init__.py

reference:
+ [https://github.com/google-research/bert](https://github.com/google-research/bert)
      
+ [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

