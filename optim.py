from transformers import TFAutoModel
import keras_tuner as kt
from tensorflow import keras
import tensorflow as tf
import random

class ACP2HyperModel(kt.HyperModel):
    def __init__(self,
    train_text,
    train_label,
    test_text,
    test_label,
    vocab,
    dims : list = [128*4],
    hdim : list = [768//4],
                 ):
      super().__init__()
      self.dims = dims
      self.nheads= 4
      self.ngrams = [1]
      self.seq_len = 25
      self.hdim = [dims[-1]//4]

      i = random.randrange(len(self.ngrams))
      self.train_text = train_text
      self.train_label = train_label
      self.test_text = test_text
      self.vocab = vocab
      self.test_label = test_label
       
    def build(self, hp):
        dim = hp.Choice('dim', self.dims)
        self.dim = dim
        hdim = hp.Choice('hdim', self.hdim)
        fc_dim = hp.Choice('fc_dim', [dim])
        ngrams = hp.Choice('ngrams', self.ngrams)
        dropout = hp.Choice('dropout', [0.2, 0.3])
        n_layers_2 = hp.Choice('n_layers_2', [1, 2, 4])
        nheads = 4
        n_embeddings = 1
        seq_len = self.seq_len
        
        self.tokenizers = [
                AutoTokenizer.from_pretrained('AmelieSchreiber/esm2_t6_8M_finetuned_cafa5', padding='max_length', truncation=True) 
                for i in range(n_embeddings)]

        model_kwargs = {
                'tokenizers': self.tokenizers,
                'dim' : dim,
                'hdim' : hdim,
                'seq_len': seq_len,
                'embeddings_conf' : {
                    'trainable' : True
                    },
                'fc_dim': fc_dim,
                'n_embeddings' : n_embeddings
                }


        #self.esm = TFAutoModel.from_pretrained('google/t5-efficient-tiny', from_pt=True)
        self.esm_fc = RetentionEmbedding(len(self.tokenizers[-1].get_vocab()), dim)#Dense(dim)

        transformer_kwargs = {
                'num_layers' : 1,
                'd_model' : 128,
                'dff' : 512,
                'num_heads' : 8,
                'dropout_rate' : 0.1,
                'input_vocab_size': len(self.tokenizers[-1].get_vocab()),
                'target_vocab_size': 1,
                }

        self.layer = ACP2RetNet(model_kwargs)#Transformer(**transformer_kwargs)#
        
        x = Input((seq_len*n_embeddings, ))
        #o = #tf.split(x, n_embeddings, 1)[-1]
        #o = self.esm(x, decoder_input_ids=x).last_hidden_state
        o = self.esm_fc(x)
        outputs = self.layer([x, o])
        self.model = Model(inputs=x, outputs=outputs)
        self.optimizer = tf.keras.optimizers.Adam(1e-5)#CustomSchedule(dim))#, weight_decay=1e-5)
        
        self.model.compile(optimizer=self.optimizer,
              loss= tf.keras.losses.BinaryCrossentropy(),
              metrics=['binary_accuracy',\
                       tf.keras.metrics.Recall(),\
                       tf.keras.metrics.Precision(),
                       tf.keras.metrics.AUC()])
        return self.model

    def fit(self, hp, model, *args, **kwargs):
        train_text = list(map(lambda tokenizer: np.asarray(tokenizer.batch_encode_plus(train_dataset.text, padding='max_length', max_length=25, truncation=True)['input_ids']), self.tokenizers))
        val_text = list(map(lambda tokenizer: np.asarray(tokenizer.batch_encode_plus(test_dataset.text, padding='max_length', max_length=25, truncation=True)['input_ids']), self.tokenizers))
        train_text = tf.concat(train_text, 1)
        val_text = tf.concat(val_text, 1)
        train_label =  self.train_label
        val_y =  self.test_label
        return model.fit(train_text, train_label, validation_data=[val_text, val_y],
            *args,
            **kwargs)


    

