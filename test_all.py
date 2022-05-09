import pytest
import project3 as p
import pandas as pd


def test_features():
    masked="I couldn't image ██████████████ in a serious role, but his performance truly"
    length=p.features(masked)
    assert length==14

def test_filt_sentence():
    sentence="I couldn't image ██████████████ in a serious role, but his performance truly"
    l1=p.filt_sentences(sentence)
    assert type(l1)==list

def test_words():
    sentence="I couldn't image ██████████████ in a serious role, but his performance truly"
    l2=p.words(sentence)
    assert type(l2)==int
    
def test_get_features():
    df=pd.DataFrame()
    df['username']=['abc']
    df['set']=['training']
    df['name']=['aston kutcher']
    df['sentence']=["I couldn't image ██████████████ in a serious role, but his performance truly"]
    df['no_words']=[10]
    df['length_masked']=[14]
    l3=p.get_features(df)
    assert type(l3[0])==dict
    
def test_get_scores():
    url = 'https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv'
    train,test,valid=p.get_data(url)
    l3=p.get_scores(train,test)
    assert type(l3)!=None
    

