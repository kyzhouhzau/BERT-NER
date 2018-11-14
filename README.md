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
  python BERT_NER.py   \
                  --task_name="NER"  \ 
                  --do_train=true   \
                  --do_eval=True   \
                  --data_dir=NERdata   \
                  --vocab_file=checkpoint/vocab.txt  \ 
                  --bert_config_file=checkpoint/bert_config.json \  
                  --init_checkpoint=checkpoint/bert_model.ckpt   \
                  --max_seq_length=128   \
                  --train_batch_size=32   \
                  --learning_rate=2e-5   \
                  --num_train_epochs=3.0   \
                  --output_dir=./output/result_dir/ 
 ```       

result:

![](/picture2.png)


#### 注：For the parameters of the above model, I have not made any modifications. All parameters are based on the BERT default parameters. The better parameters for this problem can be adjusted by yourselves.

The f_score evaluation codes come from:https://github.com/guillaumegenthial/tf_metrics/blob/master/tf_metrics/__init__.py

reference:
+ [https://github.com/google-research/bert](https://github.com/google-research/bert)
      
+ [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

