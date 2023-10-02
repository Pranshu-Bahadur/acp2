# the goal here is to implmement NAS for the ACP2 dataset
# I can do this using keras tuner
import keras_tuner as kt
from tensorflow import keras
import tensorflow as tf
import random

class ACP2HyperModel(kt.HyperModel):
    def __init__(self,
    tokenizer,
    dims : list = [64, 128, 256, 512, 1024],
    nheads : list = [4],
                 ):
      super().__init__()
      self.dims = dims
      self.nheads=nheads
      self.tokenizer = tokenizer

       
    def build(self, hp):
        dim = hp.Choice('dim', self.dims)
        nheads = hp.Choice('nheads', self.nheads)
        embedding_layers = [
            PositionalEmbedding,
            Embedding
            ]

        i = random.randrange(len(embedding_layers))


        self.embedding_layer = embedding_layers[i](len(tokenizer.get_vocabulary()),
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
                'seq_len': 50
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

