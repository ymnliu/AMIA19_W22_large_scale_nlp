import os
import click
from pathlib import Path
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk import word_tokenize
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Activation
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer

# Set environment variable in Docker to use correct directory
# if None specified, then default to local machine
try:  
   in_docker = os.environ["DOCKER"]
except:
   in_docker = None 

def get_predictive_model():

    encoder = LabelBinarizer()
    tokenizer = Tokenizer(num_words=5000)
    
    # set directries based on run-time environment
    if in_docker == 'True':
        model_dir = '/data/models/'
        data_dir = '/data/data/'
    else:
        model_dir = 'models/'
        data_dir = 'data/'

    # get model and convert to w2v
    glove_input_file = model_dir + 'w2v_glove_300.txt'

    word2vec_output_file = '/tmp/w2v.txt'
    glove2word2vec(glove_input_file, word2vec_output_file)
    wv_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

    # get stop words
    sw = data_dir + "stopwords.txt" 
    with open(sw) as f:
        stop_words = f.read().splitlines()

    # set parameters:
    maxlen = 100
    batch_size = 32
    embedding_dims = 50
    filters = 250
    kernel_size = 3
    hidden_dims = 250
    epochs = 20

    def get_input_seq(sentence):
        word_list = word_tokenize(sentence)
        word_list = [word.lower() for word in word_list if word.lower() not in stop_words]
        idx_seq = []

        for word in word_list:
            if wv_model.vocab.get(word):
                idx = wv_model.vocab.get(word).index
                idx_seq.append(idx)

        return idx_seq
    
    def get_input_word(sentence):
        word_list = word_tokenize(sentence)
        word_list = [word.lower() for word in word_list if word.lower() not in stop_words]
        idx_seq = []

        for word in word_list:
            if wv_model.vocab.get(word):
                idx = wv_model.vocab.get(word).index
                idx_seq.append(idx)

        return word_list

    # load prepartitioned train/test sets
    test = pd.read_csv(data_dir + "test.csv") # directories for use for local testing; change path accordingly
    train = pd.read_csv(data_dir + "train.csv")

    test['seq'] = [get_input_seq(sent) for sent in test.text]
    train['seq'] = [get_input_seq(sent) for sent in train.text]

    train_grouped_abbr = train.groupby('abbrev')
    test_grouped_abbr = test.groupby('abbrev')


    # Function to create model, required for KerasClassifier
    def create_cnn_model(output_dim, max_features):
        # create model
        model = Sequential()
        model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
        model.add(Dropout(0.2))

        # we add a Convolution1D, which will learn filters
        # word group filters of size filter_length:
        model.add(Conv1D(filters,
                         kernel_size,
                         padding='valid',
                         activation='relu',
                         strides=1))

        # we use max pooling:
        model.add(GlobalMaxPooling1D())

        # We add a vanilla hidden layer:
        model.add(Dense(hidden_dims))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))

        model.add(Dense(output_dim, activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'],)
        return model


    print_model_summary = True

    # Loop through different abbreviations.
   for abbr in train.abbrev.unique():
    #for abbr in ['MS']:
        if abbr == 'FISH':
            continue

        train_abbr = train_grouped_abbr.get_group(abbr)
        test_abbr = test_grouped_abbr.get_group(abbr)
        
        train_transfomed_label = encoder.fit_transform(train_abbr.expansion)
        test_transfomed_label = encoder.transform(test_abbr.expansion)

        X_train = sequence.pad_sequences(train_abbr.seq, maxlen=maxlen)

        y_train = train_transfomed_label

        X_test = sequence.pad_sequences(test_abbr.seq, maxlen=maxlen)
        y_test = test_transfomed_label
        
        print()
        print("##" * 20)
        print(" " * 20 + abbr)
        print("##" * 20)

        model = create_cnn_model(len(encoder.classes_), max(X_train.max(), X_test.max()) + 1)

        if print_model_summary:
            model.summary()
            print_model_summary = False

        history = model.fit(X_train, y_train,
                            epochs=epochs,
                            callbacks=[EarlyStopping(monitor='val_loss', verbose=1, patience=4)],
                            verbose=2,
                            validation_data=(X_test, y_test),
                            batch_size=batch_size)

        y_pred = model.predict(X_test)
        # get labels for preidictions
        lookup = encoder.inverse_transform(y_pred)

        output_dir = Path(data_dir + "output")
        output_dir.mkdir(parents=True, exist_ok=True)
        (pd.DataFrame({'predictions':lookup})).to_csv(output_dir / "cnn_{}.csv".format(abbr))

        y_test_idx = y_test.argmax(axis=1)
        target_names = [encoder.classes_[idx] for idx in set(y_test_idx)]

        print(classification_report(y_test_idx, y_pred.argmax(axis=1), target_names=target_names))
        

@click.command()
@click.option('-c', '--classifier', 'classifier', default='keras', help='Run predictive model for cnn/Keras classifier', type=click.STRING)

def get_classifier(classifier):
    """ Run Kera/CNN classifier """
    if classifier is None:
        exit(1)
    print('Running ', classifier) 
    get_predictive_model()

if __name__ == '__main__':
    classifier = get_classifier() 
