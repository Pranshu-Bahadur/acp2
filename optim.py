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
      self.seq_len = 25
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
        #embedding_type = hp.Choice('embedding_type', [0, 1])
        n_layers_2 = hp.Choice('n_layers_2', [1, 2, 4])
        nheads = 4

        seq_len = self.seq_len
        
        embedding_layers = [
            PositionalEmbedding,
            #Embedding
            ]

        self.tokenizer = TextVectorization(
          standardize=None,
          split='character',
          ngrams=ngrams,
          output_mode='int',
          output_sequence_length=seq_len,
          trainable=False)
          #vocabulary=self.vocab)
        self.tokenizer.adapt(self.train_text)

        #seq_len = len(self.tokenizer.get_vocabulary())
        

        self.embedding_layer = embedding_layers[0](len(self.tokenizer.get_vocabulary()), dim, dropout=dropout, trainable=True)

        retention_layers = [
            #Retention,
            #RecurrentRetention,
            ChunkwiseRetention,
            ]

        retention_kwargs = [{
                'retention_layer': retention_layers[0],
                'dim' : dim,
                'hdim' : hdim,
                'seq_len': seq_len,
                }for i in range(n_layers)]

        self.msr_layer = Sequential([
          RetentionBlock(**retention_kwargs[i])
          for i in range(n_layers)
          ])

        attention_layers = [
          BaseAttention,
          EncoderLayer
        ]

        resnet_kwargs = {
          'dim' : dim,
          'kernel_size': 2,
        }

        attention_kwargs = {
          'num_heads': 4,
          'd_model' : dim,
          'dff' : dim
        }

        #layer = attention_layers[hp.Choice('layer_attention', [i for i in range(len(attention_layers))])]

        self.attention_layer = EncoderLayer(**attention_kwargs)

        
        self.resnet_layers = Sequential([
          ResNetBlock(**resnet_kwargs)
          for i in range(n_layers_2)
          ])
        
        self.dropout = Dropout(0.2)

        self.fc = Sequential([
            #*[Dense(dim) for i in range(n_layers_2)],
            Flatten(),
            Dense(1, activation='sigmoid')
            ])
        inputs = Input((seq_len, ))
        #inputs = self.dropout(inputs)
        x = self.embedding_layer(inputs)
        #x = self.attention_layer(x)
        #x = self.resnet_layers(x)
        x = self.msr_layer(x)
        #
        x = self.fc(x)
        self.model = Model(inputs=inputs, outputs=x)
        self.optimizer = tf.keras.optimizers.Adam(1e-3)#CustomSchedule(dim))#, weight_decay=1e-5)
        
        self.model.compile(optimizer=self.optimizer,
              loss='binary_crossentropy',
              metrics=['binary_accuracy',\
                       tf.keras.metrics.Recall(),\
                       tf.keras.metrics.Precision(),
                       tf.keras.metrics.AUC()])
        return self.model

    def fit(self, hp, model, *args, **kwargs):
      
        train_text = self.tokenizer(self.train_text)
        train_label = self.train_label
        val_text = self.tokenizer(self.test_text)
        val_y = self.test_label
        return model.fit(train_text, train_label, validation_data=[val_text, val_y],
            *args,
            **kwargs)


    

