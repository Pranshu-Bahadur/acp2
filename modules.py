
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
from sklearn.metrics.pairwise import cosine_similarity
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
    def __init__(self, dim = 32, nheads = 2, seq_len = 50, gamma = 0.9865, **kwargs):
      super().__init__()

      _dense_kwargs = {
                "use_bias" : False,
                "dtype" : 'float32'
                }
      _layer_names = ['Q', 'K', 'V']
      self.r_layers = {k: Dense(dim, **_dense_kwargs) for k in _layer_names}

      _indices = torch.arange(seq_len, dtype=torch.float)
      _decay_factors = gamma ** (_indices.unsqueeze(1) - _indices)
      D = tf.ones((seq_len, seq_len), dtype='float32') * _decay_factors.numpy()
      self.D = tf.transpose(tf.linalg.band_part(D, 0, -1), perm=[1, 0])
      self.gamma = tf.cast(gamma, tf.float32)
      self.seq_len=seq_len


      _dense_kwargs = {
                "use_bias" : True,
                "dtype" : 'float32'
                }
      self.S = Dense(dim, **_dense_kwargs)

    def call(self, x, training=False):
      Q, K, V = [f(z) for f, z in zip(self.r_layers.values(), x)]
      _, _, d = Q.shape
      x = Q@tf.transpose(K, perm=[0, 2, 1])
      x /= d**0.5
      D = self.D
      D /= tf.reduce_sum(tf.abs(D))**0.5
      x = x*D
      x = tf.vectorized_map(lambda xs: tf.math.divide(xs, tf.maximum(tf.abs(tf.math.reduce_sum(xs, -1)), 1)), x)
      x = x@V
      return x
        

class RecurrentRetention(Layer):
    def __init__(self, dim = 100, nheads = 2, seq_len = 50, gamma = 0.9865, **kwargs):
        super(RecurrentRetention, self).__init__()
        _dense_kwargs = {
                "use_bias" : False,
                "dtype" : 'float32'
                }
        _layer_names = ['Q', 'K', 'V']
        self.r_layers = {k: Dense(dim, **_dense_kwargs) for k in _layer_names}

        _indices = torch.arange(seq_len, dtype=torch.float)
        _decay_factors = gamma ** (_indices.unsqueeze(1) - _indices)
        D = tf.ones((seq_len, seq_len), dtype='float32') * _decay_factors.numpy()
        self.D = tf.transpose(tf.linalg.band_part(D, 0, -1), perm=[1, 0])
        self.gamma = tf.cast(gamma, tf.float32)
        self.seq_len=seq_len

    def call(self, x):
      Q, K, V = [f(z) for f, z in zip(self.r_layers.values(), x)]
      _, _, d = Q.shape
      s = [Q[:, i, :]*0 for i in range(self.seq_len)]
      for t in range(1, self.seq_len):
        s[t] = (s[t-1]*self.gamma) + tf.einsum('ib, bj -> bj', tf.transpose(K[:, t, :], perm=[1, 0]), V[:, t , :])
      S = tf.stack(s)
      x = Q*tf.transpose(S, perm=[1, 0, 2])
      return x


class ChunkwiseRetention(Layer):
  def __init__(self, dim = 32, nheads = 2, seq_len = 25, gamma = 0.9865, **kwargs):
    super(ChunkwiseRetention, self).__init__()
    self.gamma = tf.cast(gamma, tf.float32)
    _dense_kwargs = {
                "use_bias" : False,
                "dtype" : 'float32'
    }
    _layer_names = ['Q', 'K', 'V']
    self.r_layers = {k: Dense(dim, **_dense_kwargs) for k in _layer_names}

    self.seq_len=seq_len
    self.dim = dim
    self.B = 10

    _indices = torch.arange(self.B, dtype=torch.float)
    _decay_factors = gamma ** (_indices.unsqueeze(1) - _indices)
    L = tf.ones((self.B, self.B), dtype='float32') * _decay_factors.numpy()
    self.L = tf.transpose(tf.linalg.band_part(L, 0, -1), perm=[1, 0])

    
    _indices = torch.arange(seq_len//self.B, dtype=torch.float)
    self.Z = tf.convert_to_tensor((gamma ** (self.B -_indices.unsqueeze(1) - 1)).squeeze(1).numpy())

  def call(self, x, training=False):

    Q, K, V = [tf.split(f(z), self.seq_len//self.B, 1) for f, z in zip(self.r_layers.values(), x)]
    Vz =  [vi*z for z, vi in zip(self.Z.numpy().tolist(), V)]
    X = [Vz[i]*0 for i in range(len(Q))]
    R = [(tf.transpose(K[i], perm=[0, 2, 1])@Vz[i]) for i in range(self.seq_len//self.B)]

    for i in range(1, self.seq_len//self.B):
      G = (self.gamma**self.B)*R[i-1]
      R[i] = (tf.transpose(K[i], perm=[0, 2, 1])@Vz[i]) + G
      X[i-1] = (Q[i]@R[i-1])*(self.gamma**(i+1))
    for i in range(len(Q)):
      S = tf.einsum('bij, bxk -> bik', Q[i], tf.transpose(K[i], perm=[0, 2, 1]))
      X[i] = ((S*self.L)@V[i])+X[i]
    X = tf.concat(X, 1)
    return X

class MultiScaleRetention(Layer):
    def __init__(self, dim, hdim=100, seq_len=50, retention_layer=ChunkwiseRetention, **kwargs):
      super(MultiScaleRetention, self).__init__()
      dims = dim
      gamma = 1 - (2 ** (-5 - torch.arange(0, hdim)))
      gamma = gamma.numpy().tolist()
      self.dim = dim
      self.hdim = hdim
      self.heads = [retention_layer(dim=hdim, gamma=gamma[head], seq_len=seq_len, **kwargs) for head in range(dim // hdim)]
      self.gn = GroupNormalization(hdim, scale=False)
      self.wg = Sequential([
            Dense(dims, use_bias=False, activation = 'swish', **kwargs),
        ])
      self.wo = Dense(dims, use_bias=False, **kwargs)

    def call(self, q, k, v):
      W = self.wg(q) #or k, v

      q = tf.split(q, self.dim//self.hdim, 2)
      k = tf.split(k, self.dim//self.hdim, 2)
      v = tf.split(v, self.dim//self.hdim, 2)

      x = [headi([qi, ki, vi]) for headi, qi, ki, vi in zip(self.heads, q, k, v)]
      x = tf.concat(x, -1)
      Y = self.gn(x)
      x = self.wo(W * Y)
      return x

class RetentionEncoder(Layer):
    def __init__(self, dim=540, nheads=2, hdim=100, seq_len=50, retention_layer=ChunkwiseRetention, **kwargs):
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
    def __init__(self, dim=540, nheads=2, hdim=100, seq_len=50, retention_layer=ChunkwiseRetention, **kwargs):
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






















