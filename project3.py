import glob
import io
import os
import pdb
import sys

import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk

import csv
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier


def get_entity(text):
    """Prints the entity inside of the text."""
    names = []
    for sent in sent_tokenize(text):
        for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                # print('****',chunk.label(), ' '.join(c[0] for c in chunk.leaves()))
                names.append(' '.join(c[0] for c in chunk.leaves()))
    return names


def doextraction(glob_text):
    """Get all the files from the given glob and pass them to the extractor."""
    sentences = []
    names = []
    files = glob.glob(glob_text)
    for thefile in files:
        with io.open(thefile, 'r', encoding='utf-8') as fyl:
            text = fyl.read()
            x = get_entity(text)
            for word in x:
                text = text.replace(word, u"\u2588" * len(word))
            sentences.append(text)
            names.extend(x)
    return sentences, names


def features(sentence):
    count = 0
    for i in sentence:
        if i == u"\u2588":
            count += 1

    return count


def filt_sentences(sentence):
    stop_words = set(stopwords.words("english"))
    filtered_sent = []
    words = word_tokenize(sentence)
    for w in words:
        if w not in stop_words:
            filtered_sent.append(w)

    return filtered_sent

def words(sentence):
    words=len(filt_sentences(sentence))
    return words


# To get the data
def get_data(url):
    headers = ['username', 'set', 'name', 'sentence']
    df = pd.read_csv(url, header=None, delimiter="\t", names=headers, quoting=csv.QUOTE_NONE, encoding='utf-8',on_bad_lines='skip')

    df['no_words'] = df['sentence'].apply(words)
    df['length_masked'] = df['sentence'].apply(features)
    # Few are spelled wrong so replacing with correct word.
    df['set'].mask(df['set'] == 'test', 'testing', inplace=True)
    df['set'].mask(df['set'] == 'teating', 'testing', inplace=True)
    df['set'].mask(df['set'] == 'train', 'training', inplace=True)
    # Splitting data into train,test and validation set.
    train_data = df[(df['set'] == 'training')]
    test_data = df[(df['set'] == 'testing')]
    validation_data = df[(df['set'] == 'validation')]

    return train_data, test_data, validation_data

def get_features(data):
    feature_list=[]
    for i in range(0,data.index.size):
        name=data.iloc[i][2]
        words=len(word_tokenize(name))
        white_s=name.count(" ")
        dict_features={
                      'words':data.iloc[i][4],
                      'length_names':data.iloc[i][5]}
        feature_list.append(dict_features)
    return feature_list


def get_scores(train, test):
    X_train = get_features(train)
    vec = DictVectorizer(sparse=False).fit_transform(X_train)
    Y_train = train['name'].values
    model = RandomForestClassifier()
    model.fit(vec, Y_train)

    X_test = get_features(test)
    v = DictVectorizer(sparse=False).fit_transform(X_test)
    Y_test = test['name'].values

    pred = model.predict(v)

    # a=accuracy_score(pred,Y_test)
    p = precision_score(pred, Y_test, average='micro')
    f = f1_score(pred, Y_test, average='weighted')
    r = recall_score(pred, Y_test, average='micro')

    print('Precision score: ', p)
    print('Recall score: ', r)
    print('F1 score: ', f)
    return pred


def main(dset):
    url = 'https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv'
    train, test, validation = get_data(url)
    if dset=='validation':
        p=get_scores(train,validation)
    if dset=='testing':
        p=get_scores(train,test)

import argparse
if __name__ == '__main__':
# main()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="validation or test data")

    args = parser.parse_args()
    if args.dataset:
        main(args.dataset)
