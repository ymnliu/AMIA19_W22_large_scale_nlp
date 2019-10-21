import sys
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk.tokenize import TreebankWordTokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

# get model and convert to w2v
glove_input_file = '/data/w2v_glove_300.txt' # directory for use in docker; change path accordingly
word2vec_output_file = 'w2v.txt'
glove2word2vec(glove_input_file, word2vec_output_file)
model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

# get stop words
sw = "data/stopwords.txt" # directory for use in docker; change path accordingly
with open(sw) as f:
    stop_words = f.read().splitlines()

def sentence_vector(sentence):
    word_list = TreebankWordTokenizer().tokenize(sentence)
    word_list = [word for word in word_list if word not in stop_words]
    word_vectors = []
    for x in word_list:
        try:
            word_vectors.append(model[x])
        except KeyError:
            None    
    return sum(word_vectors)/len(word_vectors)

def vector_breakage(sentence):
    word_list = TreebankWordTokenizer().tokenize(sentence)
    word_list = [word for word in word_list if word not in stop_words]
    word_vectors_list = []
    for x in word_list:
        try:
            if len(model[x])==200:
                word_vectors_list.append(x)
        except:
            None
        else:
            None
    return word_vectors_list

# load prepartitioned train/test sets
test = pd.read_csv("data/test.csv") # directories for use in docker; change path accordingly
train = pd.read_csv("data/AMIA_train_set.csv")

# load full data set
frames = [test, train]
df = pd.concat(frames)
df = df[['text','expansion']]
df['vec'] = [sentence_vector(x) for x in df.text]
df.expansion.unique()

test = test[['text','expansion', 'case']]
train = train[['text','expansion']]
test['vec'] = [sentence_vector(x) for x in test.text]
train['vec'] = [sentence_vector(x) for x in train.text]

# vectorize
X = list(df.vec)
X = np.array(X)
y = df.expansion

X1 = list(train.vec)
X_train = np.array(X1)
y_train = train.expansion

X2 = list(test.vec)
X_test = np.array(X2)
y_test = test.expansion

# set up SVM
clf = SVC(C=1.0, kernel='linear', degree=1).fit(X_train, y_train)

# get CV predictions and evaluation data
pred = clf.predict(X_test)
cm = confusion_matrix(y_test, pred,labels=list(set(df.expansion)))
cross_val_scores = cross_val_score(clf, X, y, cv=7)

predicted_expansion = list(pred)
case = test['case'].tolist()

results = pd.DataFrame(
    {'case': case,
     'expansion': predicted_expansion
    })

print('PREDICTED RESULTS:')
print(results)
print('=========================================')

print('accuracy: {}'.format(cross_val_scores))
print()
print(set(df.expansion))
print([len(df[df.expansion == x]) for x in set(df.expansion)])
print()
print(cm)
print()
print(f1_score(y_test,pred,average = 'weighted'))
