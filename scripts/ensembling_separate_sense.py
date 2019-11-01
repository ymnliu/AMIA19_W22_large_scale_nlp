import os
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk import word_tokenize
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Set environment variable in Docker to use correct directory
# if None specified, then default to local machine
try:  
   in_docker = os.environ["DOCKER"]
except:
   in_docker = None 

def get_predictive_model():
    # get model and convert to w2v
    if in_docker == 'True':
        input_dir = '/data/models/'
        output_dir = '/data/data/'
    else:
        input_dir = 'models/'
        output_dir = 'data/'

    glove_input_file = input_dir + 'w2v_glove_300.txt'
    
    word2vec_output_file = '/tmp/w2v.txt'
    glove2word2vec(glove_input_file, word2vec_output_file)
    model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

    # get stop words

    sw = "data/stopwords.txt"
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


    # load prepartitioned train/test sets
    test = pd.read_csv("data/test.csv")
    train = pd.read_csv("data/train.csv")

    test['vec'] = [get_sentence_vector(x) for x in test.text]
    train['vec'] = [get_sentence_vector(x) for x in train.text]

    train_grouped_abbr = train.groupby('abbrev')
    test_grouped_abbr = test.groupby('abbrev')

    # load full data set
    frames = [test, train]
    df = pd.concat(frames)
    
    print("running voting for each acronym")
    # Loop through different abbreviations.
    for abbr in train.abbrev.unique():

        train_abbr = train_grouped_abbr.get_group(abbr)
        test_abbr = test_grouped_abbr.get_group(abbr)

        X_train = np.array(list(train_abbr.vec))
        y_train = train_abbr.expansion

        X_test = np.array(list(test_abbr.vec))
        y_test = test_abbr.expansion

        # Support Vector Machine
        svm = SVC(C=1.0, kernel='linear', degree=1)
        
        # Logistic Regression
        lr = LogisticRegression()

        # Multilayer Perceptron
        mlp = MLPClassifier()
        
        # Bagging
        bag = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
        
        # Boosting
        num_trees = 70
        boost = AdaBoostClassifier(n_estimators=num_trees, random_state=1032).fit(X_train, y_train)
        
        # Random Forest
        rf = RandomForestClassifier()
        
        estimators = [('svm', svm), ('logistic_regression', lr), ('mlp', mlp), ('bagging', bag), ('boosting', boost), ('random_forest', rf)]
        
        # ensembled classifier
        ensemble = VotingClassifier(estimators).fit(X_train, y_train)

        pred = ensemble.predict(X_test)
        (pd.DataFrame({'predictions':pred})).to_csv(output_dir + "ensemble_%s.csv" % (abbr))
        
        cm = confusion_matrix(y_test, pred, labels=list(set(df.expansion)))
        print()
        print("MODEL -> ENSEMBLE")
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

if __name__ == '__main__':
    print("Running ensemble")
    get_predictive_model()
