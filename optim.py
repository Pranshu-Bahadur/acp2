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
    dims : list = [64, 128, 256, 512, 1024],
    nheads : list = [4],
                 ):
      super().__init__()
      self.dims = dims
      self.nheads=nheads
      self.ngrams = [3, 4, 5]
      self.seq_len = [50]

      i = random.randrange(len(self.ngrams))
      self.train_text = train_text
      self.train_label = train_label
      self.test_text = test_text
      self.test_label = test_label



      

       
    def build(self, hp):
        dim = hp.Choice('dim', self.dims)
        nheads = hp.Choice('nheads', self.nheads)
        ngrams = hp.Choice('ngrams', self.ngrams)
        seq_len = 50
        
        embedding_layers = [
            PositionalEmbedding,
            Embedding
            ]

        self.tokenizer = TextVectorization(
          standardize=None,
          split='character',
          ngrams=ngrams,
          output_mode='int',
          output_sequence_length=50,
          trainable=False)
        self.tokenizer.adapt(self.train_text)
        

        i = random.randrange(len(embedding_layers))


        self.embedding_layer = embedding_layers[i](len(self.tokenizer.get_vocabulary()),
         dim)

        retention_layers = [
            Retention,
            RecurrentRetention,
            ChunkwiseRetention,
            ]

        i = random.randrange(len(retention_layers))

        retention_kwargs = {
                'retention_layer': retention_layers[i],
                'dim' : dim,
                'hdim' : 32,
                'seq_len': seq_len
                }
        self.msr_layer = RetentionBlock(**retention_kwargs)

        self.fc = Sequential([
            Flatten(),
            Dense(1, activation='sigmoid')
            ])

        inputs = Input((50, ))
        x = self.embedding_layer(inputs)
        x = self.msr_layer(x)
        x = self.fc(x)

        self.model = Model(inputs=inputs, outputs=x)
        self.optimizer = tf.keras.optimizers.AdamW(1e-3)
        self.model.compile(optimizer=self.optimizer,
              loss='binary_crossentropy',
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


    

