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
    self.tokenizer = tokenizer
    self.vocabs = vocabs
    
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
    self.ffn = FeedForward(dim, dim, dropout_rate=0.2)
    self.fc = Sequential([
        #AdaptiveAveragePooling1D(self.seq_len),
        Flatten(),
        Dense(1, activation='sigmoid'),
        Dropout(0.2)
                          ])



  def _call_embeddings(self, x):
    embeddings = []
    for k, v in self.embedding_layers.items():
      self.tokenizer.set_vocabulary(self.vocabs[k])
      _input_ids = tokenizer(x)
      embedding = v(_input_ids)
      embeddings.append(embedding)
    return embeddings


  def call(self, x):
    embeddings = self._call_embeddings(x)
    for embedding in embeddings:
      x = self.layer_norm(embedding)
      x = self.retention_layer(x, x, x) + x
      x = self.ffn(self.layer_norm(x)) + x
      x = self.layer_norm(x)
    x = self.fc(x)
    return x
