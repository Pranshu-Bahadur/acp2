import tensorflow as tf
import pandas as pd
from pandas import read_csv, concat, DataFrame, merge, Series
from tensorflow.keras.layers import TextVectorization
from itertools import product

def seq_lens(df : DataFrame) -> Series:
    return df.text.str.len()

def reset_index(df : DataFrame, cols = ['label', 'text']) -> DataFrame:
    return df.reset_index()[cols]

def load_dataset(path : str, delimiter='\t', **kwargs) -> DataFrame:
    df = read_csv(path, delimiter=delimiter)[['label', 'text']]
    df.label = df.label.astype(int)
    df = reset_index(df[seq_lens(df) >= 0])
    return df

def describe_dataset(df : DataFrame, name='train') -> str:

  description = f'## {name} Dataset'
  description += '\n' + '--'*6 + '\n'
  
  description += f'\n### Size: {df.label.size}\n'
  
  description += '\n' + '--'*6 + '\n'
  
  description += f'\n### Sequence Length Descriptions:\n\n'
  _lens = seq_lens(df)
  metrics = product( [_lens], ['max', 'min', 'mean', 'mode', 'median'])
  seq_len_descs = {
      metric: pd.eval(f'_lens.{metric}()')
      for _lens, metric in metrics
  }
  mode = seq_len_descs['mode'].values[0]
  seq_len_descs['mode'] = mode
  description += f'{Series(seq_len_descs).to_markdown()}\n\n'
  description += '\n' + '--'*6 + '\n'
  description += '###Num occurences of mode: \n'
  description += f'{(_lens==mode).sum()}'
  description += '\n' + '--'*6 + '\n'
  description += '\n###Weight of mode:\n'
  description += f'\n{(_lens==mode).sum()/_lens.size}'
  description += '\n' + '--'*6 + '\n'
  description += '\n###Description of distbribution by sequence length:\n'
  description += f'{_lens.describe(percentiles=[0.1*i for i in range(1, 11)]).to_markdown()}' + 'n'*2
  return description
