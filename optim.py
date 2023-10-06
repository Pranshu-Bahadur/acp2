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
    dims : list = [128],
    hdim : list = [32],
                 ):
      super().__init__()
      self.dims = dims
      self.nheads= 4
      self.ngrams = [1]
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
        self.dim = dim
        hdim = hp.Choice('hdim', self.hdim)
        n_layers = hp.Choice('n_layers', [i for i in range(1, 3)])
        ngrams = hp.Choice('ngrams', self.ngrams)
        dropout = hp.Choice('dropout', [0.2, 0.3])
        n_layers_2 = hp.Choice('n_layers_2', [1, 2, 4])
        nheads = 4

        seq_len = self.seq_len
        self.tokenizers = [
          TextVectorization(
          split='character',
          output_mode='int',
          output_sequence_length=seq_len,
          standardize='strip_punctuation',
          ngrams=i+1,
          vocabulary=self.vocab[i]) for i in range(3)]

        model_kwargs = {
                'tokenizers': self.tokenizers,
                'dim' : dim,
                'hdim' : hdim,
                'seq_len': seq_len,
                'embeddings_conf' : {
                    'trainable' : True
                    }
                }
        self.layer = ACP2RetNet(model_kwargs)
        x = Input((seq_len*3, ))
        outputs = self.layer(x)
        self.model = Model(inputs=x, outputs=outputs)
        self.optimizer = tf.keras.optimizers.Adam(CustomSchedule(dim))#, weight_decay=1e-5)
        
        self.model.compile(optimizer=self.optimizer,
              loss= tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, label_smoothing=1e-5),
              metrics=['binary_accuracy',\
                       tf.keras.metrics.Recall(),\
                       tf.keras.metrics.Precision(),
                       tf.keras.metrics.AUC()])
        return self.model

    def fit(self, hp, model, *args, **kwargs):
        train_text = list(map(lambda tokenizer: tokenizer(self.train_text), self.tokenizers))
        val_text = list(map(lambda tokenizer: tokenizer(self.test_text), self.tokenizers))
        train_text = tf.concat(train_text, 1)
        val_text = tf.concat(val_text, 1)
        train_label =  self.train_label
        val_y =  self.test_label
        return model.fit(train_text, train_label, validation_data=[val_text, val_y],
            *args,
            **kwargs)


    

