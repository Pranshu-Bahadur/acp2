# Code Example 1-Gram

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import itertools
from itertools import product, repeat
from numpy import asarray
from functools import reduce

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def generate_vocab(sequences, ngrams):
  _conf = {
    'ngrams': 1,
    'output_mode': 'int',
    'output_sequence_length': 50,
    'split': 'character',
    'standardize': 'strip_punctuation'
    }
  tokenizer = TextVectorization(**_conf)
  tokenizer.adapt(sequences)
  amino_acids = tokenizer.get_vocabulary()[2:]
  result = []
  for i in range(ngrams):
    ngram = product(*list(repeat(amino_acids, i+1)))
    ngram = ["".join(gram) for gram in ngram]
    result.append(ngram)
  return [reduce(lambda x, y: x+y, result[:i+1], result[0]) for i in range(ngrams)]

