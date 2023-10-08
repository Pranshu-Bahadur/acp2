from utils import get_dir_files
from pathlib import Path
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, TFAutoModel
from tensorflow_addons.layers import AdaptiveAveragePooling1D


# Import all .py files in this repo directory

repo = get_dir_files(f'{Path.cwd()}')
for file in repo:
  if 'main' not in str(file):
      exec(open(file).read())


# Load datasets as DataFrames

train_dataset = load_dataset(f'{Path.cwd()}/datasets/test.tsv')
test_dataset = load_dataset(f'{Path.cwd()}/datasets/train.tsv')

# Generate vocabulary of Amino Acids based on train dataset
vocabulary = generate_vocab(train_dataset.text, 3)

tokenizer = AutoTokenizer.from_pretrained('AmelieSchreiber/cafa_5_protein_function_prediction', padding='max_length', truncation=True)
X_train = tokenizer.batch_encode_plus(train_dataset.text, padding='max_length', max_length=25, truncation=True)['input_ids']


print(X_train[0])

X_test = tokenizer.batch_encode_plus(test_dataset.text, padding='max_length', max_length=25, truncation=True)['input_ids']

tuner = kt.BayesianOptimization(ACP2HyperModel(
    train_dataset.text,
    train_dataset.label,
    test_dataset.text,
    test_dataset.label,
    vocabulary),
    objective='val_binary_accuracy',
    max_trials=20,
    directory='my_dir_3',
    project_name='3DD3_V3.33$$$4554')

tuner.search(epochs=1000, batch_size=32, shuffle=True)

esm = TFAutoModel.from_pretrained("google/t5-efficient-base", from_pt=True)

esm.compile()

inputs = tf.cast(Input((25, )), tf.int64)
#inputs = list(map(lambda x: tf.squeeze(x, 0), inputs))
#inputs = [*repeat(inputs, 2)]
x = esm(inputs, decoder_input_ids=inputs)
x = x.last_hidden_state
x = Sequential([
    AdaptiveAveragePooling1D(1),
    Dense(1, activation = 'sigmoid')(x)
])

model = Model(inputs=inputs, outputs=x)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='binary_crossentropy',
              metrics=['binary_accuracy',\
                       tf.keras.metrics.Recall(),\
                       tf.keras.metrics.Precision(),
                       tf.keras.metrics.AUC()])

#X_train = tf.stack([X_train, X_train])
#X_test = tf.stack([X_test, X_test])

model.fit(np.asarray(X_train), train_dataset.label.values,
          validation_data=[np.asarray(X_test), test_dataset.label.values],
          epochs=1000, batch_size=32)
