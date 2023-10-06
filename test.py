exec(open('layers.py').read())
exec(open('model.py').read())
exec(open('utils.py').read())
exec(open('dataset.py').read())



dataset = load_dataset('train.tsv')

X = dataset.text

print(len(generate_vocab(X, 3)[0]))
