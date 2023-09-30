import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten, Dropout, ReLU, LayerNormalization
from tensorflow_addons.layers import AdaptiveAveragePooling1D
import numpy
from itertools import product

from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten, Dropout, ReLU, LayerNormalization
from tensorflow_addons.layers import AdaptiveAveragePooling1D
import numpy
from itertools import product

class ACPClassifier(Model):
  def __init__(self,
               dim,
               seq_len,
               num_heads,
               vocabs,
               tokenizer,
               **kwargs):
    super(ACPClassifier, self).__init__()

    self.dim = dim
    self.seq_len = seq_len
    self.vocabs = vocabs
    self.tokenizer = tokenizer
    
    self.embedding_layers = {
        k:
        PositionalEmbedding(len(tokenizer.get_vocabulary()), dim)
        for k,v in self.vocabs.items()
    }

    print('embeddings initialized')

    self.retention_layer = MultiScaleRetention(dim,
                                               hdim=dim//num_heads,
                                               seq_len=seq_len)
    self.layer_norm = LayerNormalization()
    self.ffn = FeedForward(dim, dim, dropout_rate=0.1)
    self.fc = Sequential([
        AdaptiveAveragePooling1D(self.seq_len),
        Flatten(),
        Dense(1, activation='sigmoid'),
                          ])
  def _call_embeddings(self, x):
    embeddings = []
    for k, v in self.embedding_layers.items():
      self.tokenizer.set_vocabulary(self.vocabs[k])
      _input_ids = self.tokenizer(x)
      embedding = v(_input_ids)
      embeddings.append(embedding)
    return embeddings

  def _call_sequential_retention(self, embeddings):
    x = tf.vectorized_map(lambda x: self.retention_layer(x, x, x), embeddings)
    return x

  def _call_sequential_norm(self, embeddings):
    x = tf.vectorized_map(lambda x: self.layer_norm(x), embeddings)
    return x

  def _call_sequential_ffn(self, embeddings):
    x = tf.vectorized_map(lambda x: self.ffn(x), embeddings)
    return x

  def call(self, x):
    embeddings = self._call_embeddings(x)
    embeddings = tf.stack(embeddings)
    print(embeddings.shape)
    x = self._call_sequential_norm(embeddings)
    x = self._call_sequential_retention(x)
    x = self._call_sequential_ffn(x)
    x = self.fc(self.layer_norm(tf.reduce_mean(x)))
    return x
