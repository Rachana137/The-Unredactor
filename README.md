# Cs5293sp22-project3

## The Unredactor

### RACHANA VELLAMPALLI
### rachana@ou.edu

 Whenever sensitive information is shared with the public, the data must go through a redaction process. That is, all sensitive names, places, and other sensitive information must be hidden. Documents such as police reports, court transcripts, and hospital records all contain sensitive information. Redacting this information is often expensive and time consuming.
In this Project3, The unredactor will take redacted documents and return the most likely candidates to fill in the redacted location.

## Packages Required

### Installing and Importing Packages

```bash
pipenv install nltk
pipenv install numpy
pipenv install pandas
pipenv install sklearn

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
nltk.download('stopwords')
```

# To Run the Program
if the evaluated data is validation run:
```bash
pipenv run python project3.py --dataset validation
                    
```
if the evaluated data is test data run:
```bash
pipenv run python project3.py --dataset testing
```
The output returns generated precision,recall and f1 score.
```
Precision score:  0.005063291139240506
Recall score:  0.005063291139240506
F1 score:  0.0072482832795565475
```

# Assumptions or Bugs
The assumptions made are assuming that redacted data in the each sentence are exactly one name not multiple names.
There are no bugs in the provided code.

# Tasks Involved

The key to this task is to (1) make it easy for a peer to use your code to execute the model on the validations set; (2) generate a precision, recall, and f1-score of the code for the dataset. 

# project3.py

In this project for feature extraction considered (1) The length of masked data in a sentence and (2) The number of words in a sentence.


The following methods are written in project3.py file and performs the below functions.


1. **features(sentence)**

    This function takes the sentence in as input and returns the length of the redacted data in the sentence.
    
2. **filt_sentences(sentence)**

    This function removes the stopwords in a sentence and returns the list of remaining words in the sentence.
    
3. **words(sentence)**

    This function takes in the sentence and returns the number of words in a sentence.
    ```bash
    def words(sentence):
    words=len(filt_sentences(sentence))
    return words
    ```
    
4. **get_data(url)**

    This function takes the url of the data as input and converts the data into dataframe. The features are added to the dataframe in new columns. It separates the training, testing and validation data and returns the data.
    
    Here, the url is 'https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv'. This takes the latest unredactor.tsv data.
    
    In the data, few are mispelled like for testing it's written test/teating. so I replaced them with testing.
    get_features(data)**
    
    The example of the dataframe for training/testing/validation:
    ```bash
    def get_data(url):
    ...
    ...
    ...

    return train_data, test_data, validation_data
    ```
    
5. **get_features(data)**

    This function returns the dictionary of the features.
    ```bash
    dict_features={
                      'words':data.iloc[i][4],
                      'length_names':data.iloc[i][5]
                      }
    ```
    
6. **get_scores(train, test)**

    This function takes in the train and test/validation dataframe and prints the precision,recall and f1 score.
    First, it takes in dictionary and vectorizes using DictVectorizer for X_train and X_test. In Y_train and Y_test the labeled data values are stored. 
    
    Here, The model is trained using RandomForestClassifier and performs precision,recall and f1-score on predicted labels and Y_test.
    
7. **doextraction(glob_text) & get_entity(text)**

    This functions are used for getting the data from text files and identifying the person names and redacting them by a block character u"\u2588".
    
    # References
    
    https://stackoverflow.com/questions/18016037/pandas-parsererror-eof-character-when-reading-multiple-csv-files-to-hdf5
