# the goal here is to implmement NAS for the ACP2 dataset
# I can do this using keras tuner
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
    dims : list = [64, 128, 256, 512],
    hdim : list = [32],
                 ):
      super().__init__()
      self.dims = dims
      self.nheads=4
      self.ngrams = [i for i in range(1, 25)]
      self.seq_len = 50
      self.hdim = hdim

      i = random.randrange(len(self.ngrams))
      self.train_text = train_text
      self.train_label = train_label
      self.test_text = test_text
      self.vocab = vocab
      self.test_label = test_label

       
    def build(self, hp):
        dim = hp.Choice('dim', self.dims)
        hdim = hp.Choice('hdim', self.hdim)
        n_layers = hp.Choice('n_layers', [i for i in range(1, 5)])
        ngrams = hp.Choice('ngrams', self.ngrams)
        #n_layers_2 = hp.Choice('n_layers', [0, 1, 2, 4])
        nheads = 4

        seq_len = self.seq_len
        
        embedding_layers = [
            PositionalEmbedding,
            Embedding
            ]

        self.tokenizer = TextVectorization(
          standardize=None,
          split='character',
          ngrams=ngrams,
          output_mode='int',
          output_sequence_length=seq_len,
          trainable=False,
          vocabulary=self.vocab)
        #self.tokenizer.adapt(self.train_text)
        

        i = random.randrange(len(embedding_layers))


        self.embedding_layer = embedding_layers[i](len(self.tokenizer.get_vocabulary()), dim)

        retention_layers = [
            Retention,
            RecurrentRetention,
            ChunkwiseRetention,
            ]

        i = random.randrange(len(retention_layers))

        retention_kwargs = [{
                'retention_layer': retention_layers[random.randrange(len(retention_layers))],
                'dim' : dim,
                'hdim' : hdim,
                'seq_len': seq_len
                }for i in range(n_layers)]

        self.msr_layer = Sequential([
          RetentionBlock(**retention_kwargs[i])
          for i in range(n_layers)
          ])

        self.fc = Sequential([
            #*[Dense(dim) for i in range(n_layers_2)],
            Flatten(),
            Dense(64, activation='relu', kernel_initializer='glorot_normal'),
            Dense(32, activation='relu', kernel_initializer='glorot_normal'),
            Dense(16, activation='relu', kernel_initializer='he_uniform'),
            Dense(1, activation='sigmoid')
            ])

        inputs = Input((seq_len, ))
        x = self.embedding_layer(inputs)
        x = self.msr_layer(x)
        x = self.fc(x)

        self.model = Model(inputs=inputs, outputs=x)
        self.optimizer = tf.keras.optimizers.AdamW(1e-3)
        self.model.compile(optimizer=self.optimizer,
              loss='mse',
              metrics=['binary_accuracy',\
                       tf.keras.metrics.Recall(),\
                       tf.keras.metrics.Precision(),
                       tf.keras.metrics.AUC()])
        return self.model

    def fit(self, hp, model, *args, **kwargs):
        train_text = self.tokenizer(self.train_text)
        train_label = self.train_label
        val_text = self.tokenizer(self.test_text)
        val_y = self.test_label
        return model.fit(train_text, train_label, validation_data=[val_text, val_y],
            *args,
            **kwargs)


    

