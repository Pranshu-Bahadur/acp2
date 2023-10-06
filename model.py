class ACP2RetNet(Model):
    def __init__(self, conf, **kwargs):
        super().__init__()

        #@TODO: Add Memoization for states
        
        _tokenizers = conf['tokenizers']
        _embeddings_conf = conf['embeddings_conf']

        self.embedding_layers = self._init_embedding_layers(_tokenizers,
                conf['dim'], **_embeddings_conf)

        _arg_keys = ['dim', 'hdim', 'seq_len']
        _retention_args = list(map(lambda key: conf[key], _arg_keys))
        self.encoder_layer_1 = RetentionEncoder(*_retention_args)
        self.encoder_layer_2 = RetentionEncoder(*_retention_args)

        self.decoder_layer = RetentionDecoder(*_retention_args)

        self.final_layer = Sequential([
            Flatten(),
            Dense(1, activation = 'sigmoid')
            ])

    def _init_embedding_layers(self, tokenizers, dim, **kwargs) -> list:
        
        vocab_size = lambda tokenizer: len(tokenizer.get_vocabulary())
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
        Q, K, V = list(map(lambda x: self.encoder_layer_2(x), embeddings)]
        _, _, d = Q.shape
        x = Q@tf.transpose(K, perm=[0, 2, 1])
        x /= d**0.5
        D = self.D
        D /= tf.reduce_sum(D, 1)**0.5
        x = x*D
        x = tf.vectorized_map(lambda xs: tf.math.divide(xs, tf.maximum(tf.abs(tf.math.reduce_sum(xs, -1)), 1)), x)
        x = x@V
        return x
          
    def call(self, x, training=False):
        x = self._call_embedding_layers(x)
        x = self._call_encoder_sequential(x)
        x = self._call_encoder_parallel(x)
        x = self.decoder_layer(x, x)
        x = self.final_layer(x)
        return x
