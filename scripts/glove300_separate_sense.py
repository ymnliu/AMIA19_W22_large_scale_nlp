import os
import click
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk import word_tokenize
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Set environment variable in Docker to use correct directory
# if None specified, then default to local machine
try:  
   in_docker = os.environ["DOCKER"]
except:
   in_docker = None 

def get_predictive_model(classifier):
    # get model and convert to w2v
    if in_docker == 'True':
        model_dir = '/data/models/'
        data_dir = '/data/data/'
    else:
        model_dir = 'models/'
        data_dir = 'data/'

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

    # Loop through different abbreviations.
    for abbr in train.abbrev.unique():

        train_abbr = train_grouped_abbr.get_group(abbr)
        test_abbr = test_grouped_abbr.get_group(abbr)

        X_train = np.array(list(train_abbr.vec))
        y_train = train_abbr.expansion

        X_test = np.array(list(test_abbr.vec))
        y_test = test_abbr.expansion
        
        if classifier == 'svm':
            # set up SVM
            clf = SVC(C=1.0, kernel='linear', degree=1).fit(X_train, y_train)

        elif classifier == 'log':
            clf = LogisticRegression().fit(X_train, y_train)

        elif classifier == 'mlp':
            clf = MLPClassifier().fit(X_train, y_train)
        
        elif classifier == 'bagging':
            clf = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1)).fit(X_train,y_train)
        
        elif classifier == 'boosting':
            num_trees = 70
            clf = AdaBoostClassifier(n_estimators=num_trees, random_state=seed).fit(X_train, y_train)
        
        elif classifier == 'rf':
            clf = RandomForestClassifier().fit(X_train, y_train)

        else:
            print('INVALID OPTION!')

        pred = clf.predict(X_test)
        (pd.DataFrame({'predictions':pred})).to_csv(output_dir + "%s_%s.csv" % (classifier,abbr))
        
        cm = confusion_matrix(y_test, pred, labels=list(set(df.expansion)))
        print()
        print("MODEL -> ", classifier)
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


@click.command()
@click.option('-c', '--classifier', 'classifier', default='svm', help='Run predictive model for: SupportVectorMachine (svm); LogisticRegresssion (log);  MultilayerPerceptron (mlp); RandomForest (rf); Bagging (bag); Boosting (boost) ; ', type=click.STRING)

def get_classifier(classifier):
    """ Run given classifier on GLoVe embedding model """
    if classifier is None:
        exit(1)
    print('Running ', classifier) 
    get_predictive_model(classifier)
    #return classifier

if __name__ == '__main__':
    classifier = get_classifier()


