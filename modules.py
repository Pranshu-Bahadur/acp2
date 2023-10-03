
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

def PE(length, dim):
  dim /= 2

  posn = torch.arange(length).unsqueeze(1).float().numpy()
  dims = torch.arange(dim).unsqueeze(0).float().numpy()/dim

  posn_encoding = posn / (1e+4**dims)
  posn_encoding = tf.concat([np.sin(posn_encoding), np.cos(posn_encoding)], -1)
  return tf.cast(posn_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model, seq_len=50, **kwargs):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size,
     d_model, mask_zero=True, **kwargs)
    self.pos_encoding = PE(seq_len, dim=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, input_ids):
    length = tf.shape(input_ids)[1]
    x = self.embedding(input_ids)
    input_ids = tf.cast(input_ids, tf.int32)
    encodings = x + (self.pos_encoding[tf.newaxis, :length, :])

    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += encodings
    return x

class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()


class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff),
      tf.keras.layers.Dense(d_model, activation='relu'),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x)
    return x


class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = BaseAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x


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
      if training:
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
      else:
        Q, K, V = [f(z) for f, z in zip(self.r_layers.values(), x)]
        _, _, d = Q.shape
        s = [Q[:, i, :] for i in range(self.seq_len)]
        for t in range(1, self.seq_len):
          s[t] = (s[t-1]*self.gamma) + tf.einsum('ib, bj -> bj', tf.transpose(K[:, t, :], perm=[1, 0]), V[:, t , :])
        S = tf.stack(s)
        x = Q*tf.transpose(S, perm=[1, 0, 2])
        return x

class RecurrentRetention(Layer):
    def __init__(self, dim = 32, nheads = 2, seq_len = 50, gamma = 0.9865, **kwargs):
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
      s = [Q[:, i, :] for i in range(self.seq_len)]
      for t in range(1, self.seq_len):
        s[t] = (s[t-1]*self.gamma) + tf.einsum('ib, bj -> bj', tf.transpose(K[:, t, :], perm=[1, 0]), V[:, t , :])
      S = tf.stack(s)
      x = Q*tf.transpose(S, perm=[1, 0, 2])
      return x


class ChunkwiseRetention(Layer):
  def __init__(self, dim = 32, nheads = 2, seq_len = 50, gamma = 0.9865, **kwargs):
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
    self.B = 1

    _indices = torch.arange(self.B, dtype=torch.float)
    _decay_factors = gamma ** (_indices.unsqueeze(1) - _indices)
    L = tf.ones((self.B, self.B), dtype='float32') * _decay_factors.numpy()
    self.L = tf.transpose(tf.linalg.band_part(L, 0, -1), perm=[1, 0])

    
    _indices = torch.arange(seq_len//self.B, dtype=torch.float)
    self.Z = tf.convert_to_tensor((gamma ** (self.B -_indices.unsqueeze(1) - 1)).squeeze(1).numpy())

  def call(self, x):

    Q, K, V = [tf.split(f(z), self.seq_len//self.B, 1) for f, z in zip(self.r_layers.values(), x)]
    #d = x[-1].shape[-1]
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
    def __init__(self, dim, hdim=32, seq_len=50, retention_layer=ChunkwiseRetention, **kwargs):
      super(MultiScaleRetention, self).__init__()
      dims = dim
      gamma = 1 - (2 ** (-5 - torch.arange(0, hdim)))
      gamma = gamma.numpy().tolist()
      self.dim = dim
      self.hdim = hdim
      self.heads = [retention_layer(dim=hdim, gamma=gamma[head], seq_len=seq_len, **kwargs) for head in range(dim // hdim)]
      self.gn = GroupNormalization(scale=False)
      self.wg = Sequential([
            Dense(dims, use_bias=False, activation = 'swish', **kwargs),
        ])
      self.wo = Dense(dims, use_bias=False, **kwargs)

    def call(self, x):
      x, k, v = x, x, x
      W = self.wg(x)
      q = tf.split(x, self.dim//self.hdim, 2)
      k = tf.split(k, self.dim//self.hdim, 2)
      v = tf.split(v, self.dim//self.hdim, 2)
      x = [headi([qi, ki, vi]) for headi, qi, ki, vi in zip(self.heads, q, k, v)]
      x = tf.concat(x, -1)
      Y = self.gn(x)
      x = self.wo(W * Y)
      return x

class RetentionBlock(Layer):
    def __init__(self, dim=128, nheads=2, hdim=32, seq_len=50, retention_layer=ChunkwiseRetention, **kwargs):
        super().__init__()
        self.layer_norm = LayerNormalization()
        self.ffn = FeedForward(dim, dim)
        self.msr = MultiScaleRetention(dim, hdim, seq_len, retention_layer=retention_layer)

    def call(self, x, training=False):
      msr_x = self.msr(x)
      x = self.ffn(msr_x)
      return x
