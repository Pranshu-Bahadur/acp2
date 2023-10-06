import keras_tuner as kt
from tensorflow import keras
import tensorflow as tf
import random

class ACP2HyperModel1(kt.HyperModel):
    def __init__(self,
    train_dataset : DataFrame,
    test_dataset : DataFrame):
      super().__init__()

      _vocabulary = generate_vocab(train_dataset.text, 3)
      self.tokenizer = self.build_tokenizer(3, _vocabulary)
      self.dims = dims
      self.nheads= 4
      self.ngrams = [1]
      self.seq_len = 50
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
        n_layers_2 = hp.Choice('n_layers_2', [1, 2, 4])
        nheads = 4

        seq_len = self.seq_len
          
        self.embeddings = [PositionalEmbedding(len(self.tokenizers[i].get_vocabulary()), dim, trainable=False) for i in range(3)]
        self.outputs = PositionalEmbedding(len(self.tokenizers[0].get_vocabulary()), dim, trainable=False)

        retention_layers = [
            #Retention,
            #RecurrentRetention,
            ChunkwiseRetention,
            ]

        retention_kwargs = {
                'retention_layer': retention_layers[0],
                'dim' : dim,
                'hdim' : hdim,
                'seq_len': seq_len,
                }

        self.msr_encoder = RetentionEncoder(**retention_kwargs)
        
        self.msr_decoder = RetentionDecoder(**retention_kwargs)
        self.fc = Sequential([
            Flatten(),
            Dense(1, activation='sigmoid')
            ])

        inputs = Input((50, ))
        o = self.outputs(inputs)
        #x = tf.split(inputs, 3, -1)
        x = self.embeddings[0](inputs)#[embedding(input_ids) for embedding, input_ids in zip(self.embeddings, x)]
        #x = tf.concat(x, 1)
        x = self.msr_encoder(x)
        x = self.msr_decoder(x, o)
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
        train_text = self.tokenizers[0](self.train_text)
        val_text = self.tokenizers[0](self.test_text)
        train_label =  self.train_label
        val_y =  self.test_label
        return model.fit(train_text, train_label, validation_data=[val_text, val_y],
            *args,
            **kwargs)

    def build_tokenizer(self, ngrams, vocabulary):
      _conf = {
        'ngrams': 1,
        'output_mode': 'int',
        'output_sequence_length': 50,
        'split': 'character',
        'standardize': 'strip_punctuation',
        'vocabulary' : vocabulary
        }
      return TextVectorization(**_conf)



    
