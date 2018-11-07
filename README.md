# BERT-NER
Use google BERT to do CoNLL-2003 NER !


Try to implement NER work based on google's BERT code!
First <code>git clone https://github.com/google-research/bert.git</code>
Second <code>download file in this project</code>

    BERT
    |____ <strong>bert</strong>
    |____ BERT_NER.py
    |____ <strong>checkpoint</strong>
    |____ <strong>output</strong>


Third run:
  <code>python BERT_NER.py   \
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
                  --output_dir=./output/result_dir/ \</code>
                  
                  
result:
./picture.png


#### 注：I am a beginner of tensorflow, I don't know how to optimize the code better, and because I don't know if there is a multi-class evaluation function in tensorflow, I only call tf.metrics.accuracy for evaluation. All results are attached here. I hope to get your help!

reference:https://github.com/google-research/bert
          
          https://arxiv.org/abs/1810.04805

