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

    self.tokenizer.set_vocabulary(self.vocabs[len(vocabs)-1])

    self.embedding_layers = {
        k:
        PositionalEmbedding(len(tokenizer.get_vocabulary()), dim)
        for k,v in self.vocabs.items()
    }

    print('embeddings initialized')

    self.retention_layer = MultiScaleRetention(dim,
                                               hdim=dim//num_heads,
                                               seq_len=seq_len)
    retention_kwargs = {
            "dim" : dim,
            "hdim" :dim//num_heads,
            "seq_len": seq_len
            }

    _layer_names = ['Q', 'K', 'V']

    self.retention_layers = {k: 
                             MultiScaleRetention(**retention_kwargs)
                             for k in _layer_names
                             }
    self.layer_norm = LayerNormalization()
    self.ffn = FeedForward(dim, dim, dropout_rate=0.1)
    self.fc = Sequential([
        AdaptiveAveragePooling1D(self.seq_len),
        Flatten(),
        Dense(1, activation='sigmoid')
                          ])

    _indices = torch.arange(seq_len, dtype=torch.float)
    _decay_factors = 0.96875 ** (_indices.unsqueeze(1) - _indices)
    D = tf.ones((seq_len, seq_len), dtype='float32') * _decay_factors.numpy()
    self.D = tf.transpose(tf.linalg.band_part(D, 0, -1), perm=[1, 0])


  
  def _call_embeddings(self, x):
    embeddings = []
    for k, v in self.embedding_layers.items():
      self.tokenizer.set_vocabulary(self.vocabs[len(vocabs)-1])
      _input_ids = self.tokenizer(x)
      embedding = v(_input_ids)
      embeddings.append(embedding)
    return embeddings
  
  def _call_parallel_retention(self, embeddings):
    Q, K, V = [f(z, z, z) for f, z in zip(self.layers.values(), embeddings)]
    _, _, d = Q.shape
    x = Q@tf.transpose(K, perm=[0, 2, 1])
    x /= d**0.5
    D = self.D
    D /= tf.reduce_sum(D, 1)**0.5
    x = x*D
    x = tf.vectorized_map(lambda xs: tf.math.divide(xs, tf.maximum(tf.abs(tf.math.reduce_sum(xs, -1)), 1)), x)
    x = x@V
    return x


  def _call_sequential_retention(self, embeddings):
    x = tf.vectorized_map(lambda x: self.retention_layer(x, x, x), embeddings)
    return x

  def _call_sequential_norm(self, embeddings):
    x = tf.vectorized_map(lambda x: self.layer_norm(x), embeddings)
    return x

  def _call_sequential_ffn(self, embeddings):
    x = tf.vectorized_map(lambda x: self.ffn(x), embeddings)
    return x

  def call(self, x, training=False):
    embeddings = self._call_embeddings(x)
    x = self._call_parallel_retention(embeddings)
    x = self.layer_norm(x)
    x = self.ffn(x)
    x = self.fc(x)
    """

    if training:
        embeddings = tf.stack(embeddings)
        x = self._call_sequential_norm(embeddings)
        x = self._call_sequential_retention(x)
        x = self._call_sequential_ffn(x)
        x = self.fc(self.layer_norm(tf.reduce_mean(x, 0)))
    else:
        x = embeddings[-1]

    """
    return x
