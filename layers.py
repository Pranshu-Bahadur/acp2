
# Imports
import tensorflow as tf
from pandas import read_csv, DataFrame, concat
from tensorflow.keras import Model, Sequential
from tensorboard.plugins.projector import ProjectorConfig, visualize_embeddings
from tensorflow.keras.layers import TextVectorization, Input, Embedding, Conv1D,\
 MultiHeadAttention, LayerNormalization, Add, Dense, Flatten, BatchNormalization,\
  DepthwiseConv1D, MaxPooling1D,\
   GlobalAveragePooling1D, Concatenate, GroupNormalization, LSTM, GlobalMaxPooling1D, Activation,\
    Dropout, Attention, Dot, Bidirectional, GRU
from tensorflow.keras.metrics import AUC
import numpy as np
import torch
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Dense, ReLU, LayerNormalization, RNN, SimpleRNNCell
from tensorflow.keras.layers import Layer, Dense, LayerNormalization
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from torch.nn import RNN
import torch

class RetentionEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model, dropout=0.2, **kwargs):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size,
     d_model, mask_zero=True, **kwargs)
    self.vocab_size= vocab_size
    self.dropout = Dropout(dropout)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, input_ids, training=False):
    x = self.embedding(input_ids)
    return x

class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='gelu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])

  def call(self, x):
    return self.seq(x)

class Retention(Layer):
    def __init__(self, dim=128, nheads = 2, seq_len = 50, gamma = 0.9865, **kwargs):
        super().__init__()
        _dense_kwargs = {
                "use_bias" : False,
                "dtype" : "float32"
                }
        self._qkv_layers = [*repeat(Dense(dim, **_dense_kwargs), 3)]
        self.D = self._compute_decay(seq_len, gamma)
        self.seq_len = seq_len
        self.gamma = tf.cast(gamma, tf.float32)

    def call(self, x, training=False):
        Q, K, V = [f(z) for f, z in zip(self._qkv_layers, x)]
        _, _, d = Q.shape
        x = Q@tf.transpose(K, perm=[0, 2, 1])
        x /= d**0.5 #Normalization Trick 1
        D = self.D
        D /= tf.reduce_sum(D, 1)**0.5 #Normalization Trick 2
        x = x*D
        _norm_3 = lambda xs: tf.math.divide(xs, tf.maximum(tf.abs(tf.math.reduce_sum(xs, 1)), 1))
        x = tf.vectorized_map(_norm_3, x) #Normalization Trick 3
        x = x@V
        return x
    
    def _compute_decay(self, seq_len, gamma = 0.9865):
        _indices = torch.arange(seq_len, dtype=torch.float)
        _decay_factors = gamma ** (_indices.unsqueeze(1) - _indices)
        D = tf.ones((seq_len, seq_len), dtype='float32') * _decay_factors.numpy()
        return tf.transpose(tf.linalg.band_part(D, 0, -1), perm=[1, 0])

class MultiScaleRetention(Layer):
    def __init__(self, dim, hdim=100, seq_len=50, retention_layer=Retention, **kwargs):
      super(MultiScaleRetention, self).__init__()
      gamma = 1 - (2 ** (-5 - torch.arange(0, hdim)))
      gamma = gamma.numpy().tolist()
      self.dim = dim
      self.hdim = hdim
      self.heads = [retention_layer(dim=hdim, gamma=gamma[head], seq_len=seq_len, **kwargs) for head in range(dim // hdim)]
      self.gn = GroupNormalization(dim, scale=False)
      self.wg = Sequential([
            Dense(dim, use_bias=False, activation = 'swish', **kwargs),
        ])
      self.wo = Dense(dim, use_bias=False, **kwargs)

    def call(self, q, k, v):
      W = self.wg(q)
      x = [q, k, v]
      q, k, v = list(map(lambda val: tf.split(q, self.dim//self.hdim, 2), x))
      x = [headi([qi, ki, vi]) for headi, qi, ki, vi in zip(self.heads, q, k, v)]
      x = tf.concat(x, -1)
      Y = self.gn(x)
      x = self.wo(W * Y)
      return x

class RetentionEncoder(Layer):
    def __init__(self, dim=540, hdim=100, seq_len=50, retention_layer=Retention, **kwargs):
        super().__init__()
        self.layer_norm = LayerNormalization()
        self.msr = MultiScaleRetention(dim, hdim, seq_len, retention_layer=retention_layer)
        self.ffn = FeedForward(dim, dim)
    def call(self, x, training=False):
      xn = self.layer_norm(x)
      msr_x = self.msr(xn, xn, xn) + x
      x = self.ffn(self.layer_norm(msr_x)) + msr_x
      return x

class RetentionDecoder(Layer):
    def __init__(self, dim=540, hdim=100, seq_len=50, retention_layer=Retention, **kwargs):
        super().__init__()
        self.layer_norm = LayerNormalization()
        self.msr_1 = MultiScaleRetention(dim, hdim, seq_len, retention_layer=retention_layer)
        self.msr_2 = MultiScaleRetention(dim, hdim, seq_len, retention_layer=retention_layer)
        self.ffn = FeedForward(dim, dim)

    def call(self, x, x1, training=False):
      x1n = self.layer_norm(x1)
      x1 = self.msr_1(x1, x1, x1) + x1
      xn = self.layer_norm(x)
      x1n = self.layer_norm(x1)
      x = self.msr_2(xn, xn, x1n) + x1
      x = self.ffn(self.layer_norm(x)) + x
      return x






















