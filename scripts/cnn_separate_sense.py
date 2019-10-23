from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk import word_tokenize
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

# get model and convert to w2v
glove_input_file = '../models/w2v_glove_300.txt' # directory for use in docker; change path accordingly
word2vec_output_file = '/tmp/w2v.txt'
glove2word2vec(glove_input_file, word2vec_output_file)
model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

# get stop words
sw = "../data/stopwords.txt" # directory for use in docker; change path accordingly
with open(sw) as f:
    stop_words = f.read().splitlines()


def get_sentence_vector(sentence):
    word_list = word_tokenize(sentence)
    word_list = [word.lower() for word in word_list if word.lower() not in stop_words]
    word_vectors = []

    for x in word_list:
        try:
            w_vec = model.get_vector(x)
            word_vectors.append(w_vec)
        except KeyError:
            pass

    return sum(word_vectors) / len(word_vectors)


# Function to create model, required for KerasClassifier
def create_model():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# load prepartitioned train/test sets
test = pd.read_csv("../data/test.csv") # directories for use in docker; change path accordingly
train = pd.read_csv("../data/AMIA_train_set.csv")

test['vec'] = [get_sentence_vector(x) for x in test.text]
train['vec'] = [get_sentence_vector(x) for x in train.text]

train_grouped_abbr = train.groupby('abbrev')
test_grouped_abbr = test.groupby('abbrev')

# Loop through different abbreviations.
for abbr in train.abbrev.unique():

    train_abbr = train_grouped_abbr.get_group(abbr)
    test_abbr = test_grouped_abbr.get_group(abbr)

    X_train = np.array(list(train_abbr.vec))
    y_train = train_abbr.expansion

    X_test = np.array(list(test_abbr.vec))
    y_test = test_abbr.expansion

    # set up SVM
    clf = SVC(C=1.0, kernel='linear', degree=1, probability=True).fit(X_train, y_train)

    pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, pred, labels=list(set(df.expansion)))
    print()
    print("##" * 20)
    print(" " * 20 + abbr)
    print("##" * 20)

    print(classification_report(y_test, pred))
    print()
    print(f'examples (first 5 cases)\t\t\t\t\t\ttrue_abbr\t\t\tpred_abbr')

    # Print first 5 cases
    i = 0
    for input_row, true_abbr, pred_abbr in zip(train_abbr.iterrows(), y_test, pred):

        sn_start = max(input_row[1].start - 25, 0)
        sn_end = min(input_row[1].end + 25, len(input_row[1].text))

        example_text = input_row[1].text[sn_start: sn_end]
        print(f'... {example_text} ...\t{true_abbr:<35}\t{pred_abbr}')

        if i == 5:
            break

        i += 1