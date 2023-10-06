exec(open('layers.py').read())


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

        self.encoder_layer = RetentionEncoder(*_retention_args)
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

    def _call_embedding_layers(self):
        pass

