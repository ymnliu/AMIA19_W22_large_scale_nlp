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
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


# Seed value
# Apparently you may use different seed values at each stage
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value

# 2. Set the `python` built-in pseudo-random generator at a fixed value

# 3. Set the `numpy` pseudo-random generator at a fixed value

import numpy as np
np.random.seed(42)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.set_random_seed(42)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# set parameters:
maxlen = 100
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 10

try:
   in_docker = os.environ["DOCKER"]
except:
   in_docker = None

if in_docker == 'True':
    model_dir = '/data/models/'
    data_dir = '/data/data/'
else:
    model_dir = 'models/'
    data_dir = 'data/'

# get stop words
sw = data_dir + "stopwords.txt"
with open(sw) as f:
    stop_words = f.read().splitlines()


def get_input_seq(wv_model, sentence):
   word_list = word_tokenize(sentence)
   word_list = [word.lower() for word in word_list if word.lower() not in stop_words]
   idx_seq = []

   for word in word_list:
       if wv_model.vocab.get(word):
           idx = wv_model.vocab.get(word).index
           idx_seq.append(idx)

   return idx_seq


def get_sentence_vector(wv_model, sentence):
    word_list = word_tokenize(sentence)
    word_list = [word.lower() for word in word_list if word.lower() not in stop_words]
    word_vectors = []

    for x in word_list:
        try:
            w_vec = wv_model.get_vector(x)
            word_vectors.append(w_vec)
        except KeyError:
            pass

    return sum(word_vectors) / len(word_vectors)


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
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'], )
    return model


def define_ml_models():

    model_dict = {'svm': SVC(C=1.0, kernel='linear', degree=1),
                    'log': LogisticRegression(),
                    'mlp': MLPClassifier(),
                    'bag': BaggingClassifier(tree.DecisionTreeClassifier(random_state=1)),
                    'boost': AdaBoostClassifier(n_estimators=70),
                    'rf': RandomForestClassifier(),
                  }

    return model_dict

def run_predictive_model(model_name):

    encoder = LabelBinarizer()
    models = define_ml_models()

    # load prepartitioned train/test sets
    train = pd.read_csv(data_dir + "train.csv")
    test = pd.read_csv(data_dir + "test.csv")

    # get model and convert to w2v
    glove_input_file = model_dir + 'w2v_glove_300.txt'

    word2vec_output_file = '/tmp/w2v.txt'
    glove2word2vec(glove_input_file, word2vec_output_file)
    wv_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

    train['seq'] = [get_input_seq(wv_model, sent) for sent in train.text]
    test['seq'] = [get_input_seq(wv_model, sent) for sent in test.text]

    test['vec'] = [get_sentence_vector(wv_model, x) for x in test.text]
    train['vec'] = [get_sentence_vector(wv_model, x) for x in train.text]

    train_grouped_abbr = train.groupby('abbrev')
    test_grouped_abbr = test.groupby('abbrev')

    print_model_summary = True

    # Loop through different abbreviations.
    for abbr in train.abbrev.unique():
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

        output_dir = Path(data_dir + "output")
        output_dir.mkdir(parents=True, exist_ok=True)

        if model_name != 'cnn':
            X_train = np.array(list(train_abbr.vec))
            y_train = train_abbr.expansion

            X_test = np.array(list(test_abbr.vec))
            y_test = test_abbr.expansion

            model = models[model_name]
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            (pd.DataFrame({'predictions': y_pred})).to_csv(output_dir / "{}_{}.csv".format(model_name, abbr))
            print(classification_report(y_test, y_pred))

        else:
            model = create_cnn_model(len(encoder.classes_), max(X_train.max(), X_test.max()) + 1)

            if print_model_summary:
                model.summary()
                print_model_summary = False

            model.fit(X_train, y_train,
                                epochs=epochs,
                                shuffle=False,
                                callbacks=[EarlyStopping(monitor='val_loss', verbose=1, patience=4)],
                                verbose=2,
                                validation_data=(X_test, y_test),
                                batch_size=batch_size)

            y_pred = model.predict(X_test)
            # get labels for predictions
            lookup = encoder.inverse_transform(y_pred)

            (pd.DataFrame({'predictions': lookup})).to_csv(output_dir / "{}_{}.csv".format(model_name, abbr))
            y_test_idx = y_test.argmax(axis=1)
            target_names = [encoder.classes_[idx] for idx in set(y_test_idx)]

            # match labels -> target names
            le = LabelEncoder()
            le.fit(target_names)

            print(classification_report(y_test_idx, y_pred.argmax(axis=1), target_names=le.classes_))


@click.command()
@click.option('-c', '--classifier', 'classifier_name',
              default='svm', help='Run predictive model for cnn', type=click.STRING)
def run_wsd(classifier_name):
    """ Run Kera/CNN classifier """
    if classifier_name is None:
        exit(1)
    print('Running ', classifier_name)
    run_predictive_model(classifier_name)


if __name__ == '__main__':
    run_wsd()