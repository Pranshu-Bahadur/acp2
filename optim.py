# the goal here is to implmement NAS for the ACP2 dataset
# I can do this using keras tuner
import keras_tuner as kt
from tensorflow import keras
import tensorflow as tf

class ACP2HyperModel(kt.HyperModel):
    def __init__(self,
                 dims : list = [64, 128, 256],
                 nheads : list = [2, 4, 8],
                 tokenizer=tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
       
    def build(self, hp):
        dim = hp.Choice('dim', dims)
        nheads = hp.Choice('nheads', nheads)
        embedding_layer = hp.Choice('embedding_layer', [
            PositionalEmbedding,
            Embedding
            ])

        embedding_kwargs = {
                'input_dim' : len(self.tokenizer.get_vocabulary()),
                'output_dim' : dim,
                }

        self.embedding_layer = embedding_layer(**embedding_kwargs)

        retention_layer = hp.Choice('retention_layer', [
            Retention,
            RecurrentRetention,
            ChunkwiseRetention,
            ])

        retention_kwargs = {
                'retention_layer': retention_layer,
                'dim' : dim,
                'hdim' : dim//nheads,
                'seq_len': 50
                }
        self.msr_layer = RetentionBlock(**retention_kwargs)

        self.fc = Sequential([
            Flatten(),
            Dense(1, activation='sigmoid')
            ])

        inputs = Input((1, ))
        x = self.tokenizer(x)
        x = self.embedding_layer(x)
        x = self.msr_layer(x)
        x = self.fc(x)

        self.model = Model(inputs=inputs, outputs=x)
        self.optimizer = tf.keras.optimizers.AdamW(1e-3)
        self.model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['binary_accuracy',\
                       tf.keras.metrics.Recall(),\
                       tf.keras.metrics.Precision(),
                       tf.keras.metrics.AUC()])

