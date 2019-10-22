import logging
import os
import gensim
from gensim import corpora
import csv
from nltk import word_tokenize
import sys

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def read_datasets(root='../data'):

    dataset = {}

    for tag in ['train', 'test']:
        res = []
        with open(os.path.join(root, tag + '.csv'), encoding='utf-8', errors='ignore') as f:
            csv_reader = csv.reader(f)
            line_count = 0

            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                    continue
                else:
                    res.append(row)
                    line_count += 1

            print(f'Processed {line_count} lines.')

            dataset[tag] = res

    return dataset


def build_dictionary(dataset, out_path='../models/casi.dict'):
    texts = []

    for tag in ['train', 'test']:
        for row in dataset[tag]:
            if len(row) < 6:
                print(row)
                continue

            texts.append([word.lower() for word in word_tokenize(row[5])])

    dictionary = corpora.Dictionary(texts)
    dictionary.save(out_path)  # store the dictionary, for future reference
    return dictionary


def trim_word_embeddings(dictionary, w2e_path, out_path):
    with open(out_path, 'w') as fo:
        with open(w2e_path) as f:
            for l in f:
                w = l.split()[0]
                if dictionary.token2id.get(w):
                    fo.write(l)


if __name__ == '__main__':
    dataset = read_datasets()
    casi_dict = build_dictionary(dataset)

    we_root = sys.argv[1]

    print(we_root)

    trim_word_embeddings(casi_dict,
                         os.path.join(we_root, 'glove.6B.50d.txt'),
                         '../models/w2v_glove_50.txt')

    trim_word_embeddings(casi_dict,
                         os.path.join(we_root, 'glove.6B.300d.txt'),
                         '../models/w2v_glove_300.txt')
