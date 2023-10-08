from tensorflow.keras.layers import Layer
from tensorflow_addons.layers import AdaptiveAveragePooling1D, AdaptiveMaxPooling1D
import tensorflow_probability as tfp
exec(open('layers.py').read())

class ACP2RetNet(Layer):
  def __init__(self, conf, **kwargs):
    super().__init__()

      #@TODO: Add Memoization for states
    self.sentinel = tf.constant(0).ref()
    self.state = {self.sentinel : 0} 
    _tokenizers = conf.pop('tokenizers')
    _embeddings_conf = conf['embeddings_conf']

    self.embedding_layers = self._init_embedding_layers(_tokenizers,
              conf['dim'], **_embeddings_conf)

    self.n_embeddings = conf['n_embeddings']
    _arg_keys = ['dim', 'hdim', 'seq_len']
    _retention_args = list(map(lambda key: conf[key], _arg_keys))
    seq_len = _retention_args[-1]
    self.seq_len = seq_len
    self.encoder_layer_1 = RetentionEncoder(*_retention_args)
    #self.encoder_layer_2 = RetentionEncoder(*_retention_args)
    
    _retention_args[-1] = _retention_args[-1]*self.n_embeddings
    self.decoder_layer = RetentionDecoder(*_retention_args)

    self.final_layer = Sequential([
          #Dense(1),
          #LayerNormalization(),
          AdaptiveAveragePooling1D(1),
          Dense(1, activation = 'hard_sigmoid'),
          Dropout(0.2),
          ])

    _indices = torch.arange(seq_len, dtype=torch.float)
    _decay_factors = 0.96875 ** (_indices.unsqueeze(1) - _indices)
    D = tf.ones((seq_len, seq_len), dtype='float32') * _decay_factors.numpy()
    self.D = tf.transpose(tf.linalg.band_part(D, 0, -1), perm=[1, 0])
    print('init model done.')
    print(conf)

  def _init_embedding_layers(self, tokenizers, dim, **kwargs) -> list:
        
      vocab_size = lambda tokenizer: len(tokenizer.get_vocab())
      vocab_sizes = list(map(lambda tokenizer: vocab_size(tokenizer), tokenizers))  
      layers = [*repeat(RetentionEmbedding, len(vocab_sizes))]

      return list(map(lambda layer, vocab_size: layer(vocab_size, dim, **kwargs),
          layers, vocab_sizes))

  def _call_embedding_layers(self, x):
        return list(map(lambda embedding_layer, input_ids: embedding_layer(input_ids),
            self.embedding_layers, x))

  def _call_encoder_sequential(self, embeddings):
      x = tf.vectorized_map(lambda x: self.encoder_layer_1(x), embeddings)
      return x

  def _call_encoder_parallel(self, embeddings):
      Q, K, V = list(map(lambda x: self.encoder_layer_2(x), embeddings))
      _, _, d = Q.shape
      x = Q@tf.transpose(K, perm=[0, 2, 1])
      x /= d**0.5
      D = self.D
      D /= tf.reduce_sum(D, 1)**0.5
      x = x*D
      x = tf.vectorized_map(lambda xs: tf.math.divide(xs, tf.maximum(tf.abs(tf.math.reduce_sum(xs, -1)), 1)), x)
      x = x@V
      return x

  def elbo(self, x):
    x = tf.cast(x, tf.float32)
    log_likelihood = tf.reduce_sum(x, axis=-1)
    log_likelihood = tf.reduce_mean(log_likelihood)
    kl_divergence = tf.reduce_sum(
            tf.math.log(x + 1e-6) - tf.math.log(tf.cast(x.shape[-1], tf.float32)), -1)
    kl_divergence = tf.reduce_mean(kl_divergence)

    return log_likelihood - kl_divergence
      
        
  def call(self, x, training=False):
      x, o = x
      if len(self.state) == 1:
          self.state[self.sentinel] = tf.cast(o, tf.float32)
      x = tf.split(x, self.n_embeddings, -1)
      x = self._call_embedding_layers(x)
      x = tf.stack(x)
      x = self._call_encoder_sequential(x)
      x = tf.split(x, self.n_embeddings, 0)
      x = list(map(lambda xi: tf.squeeze(xi, 0), x))
      x = tf.concat(x, 1)
      if len(self.state) != 1 and self.elbo(x) is not None and not tf.is_symbolic_tensor(self.elbo(x)):
          o = self.state[max(list(map(lambda val: val.deref(),
                                      self.state.values()))).keys()]
      x = self.decoder_layer(x, o)
      if not tf.is_symbolic_tensor(self.elbo(x)):
          if self.elbo(x) > max(list(map(lambda val: val.deref().numpy(),
                                         self.state.keys()))): 
              self.state[self.elbo(x).ref()] = x
              print(self.elbo(x))
      x = self.final_layer(x)
      return x

class Transformer(tf.keras.Model):
  def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, target_vocab_size, dropout_rate=0.1):
    super().__init__()
    self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=input_vocab_size,
                           dropout_rate=dropout_rate)

    self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=target_vocab_size,
                           dropout_rate=dropout_rate)

    self.final_layer = Sequential([
          #Dense(target_vocab_size),
          AdaptiveAveragePooling1D(1),
          Dense(1, activation = 'sigmoid'),
          ])


  def call(self, inputs):
    # To use a Keras model with `.fit` you must pass all your inputs in the
    # first argument.
    context, x  = inputs

    context = self.encoder(context)  # (batch_size, context_len, d_model)

    x = self.decoder(x, context)  # (batch_size, target_len, d_model)

    # Final linear layer output.
    logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
      del logits._keras_mask
    except AttributeError:
      pass

    # Return the final output and the attention weights.
    return logits

