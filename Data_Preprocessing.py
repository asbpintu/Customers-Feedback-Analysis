# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 20:07:39 2023

@author: asbpi
"""


# Setting Path

import os
os.chdir(r'C:\Users\asbpi\Desktop\Nit_DS & AI\MY Projects\project_sentiment analysis')

# Import Packages

import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re

import spacy
import string
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# nltk.download('stopwords')
# nltk.download('punkt')

nlp = spacy.load("en_core_web_sm")
punct = string.punctuation
stem = PorterStemmer()
lemma = WordNetLemmatizer()

from wordcloud import WordCloud
from nltk.probability import FreqDist

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


'''
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Flatten, Dense, LSTM
'''

import warnings
warnings.filterwarnings('ignore')

# Reading Data

root = pd.read_csv(r'amazon_alexa.tsv' , delimiter = '\t' , quoting = 3)
data = root.copy()


# Ind and Dep variable


data = data.drop(['rating','date','variation'],axis = 1)
data.columns = ['reviews' , 'target']

x = data['reviews']
y = data['target']


# replace_text


def replace_text(rev):
    
    reviews = re.sub(r"what's", "what is ", rev)
    reviews = re.sub(r"\'s", " is", reviews)
    reviews = re.sub(r"\'ve", " have ", reviews)
    reviews = re.sub(r"can't", "cannot ", reviews)
    reviews = re.sub(r"n't", " not ", reviews)
    reviews = re.sub(r"i'm", "i am ", reviews)
    reviews = re.sub(r"\'re", " are ", reviews)
    reviews = re.sub(r"\'d", " would ", reviews)
    reviews = re.sub(r"\'ll", " will ", reviews)
    reviews = re.sub(r"\'scuse", " excuse ", reviews)
    reviews = re.sub('\W', ' ', reviews)
    reviews = re.sub('\s+', ' ', reviews)
    reviews = reviews.strip(' ')
    
    return reviews

for i in range(len(x)) :
    x[i] = replace_text(x[i])


# cleaned_text


def cleaned_text(rev):
      
    reviews = re.sub(r'\[[0-9]*\]', ' ',rev)
    reviews = re.sub(r'\s+', ' ', reviews)
    reviews = re.sub('[^a-zA-Z]', ' ', reviews )
    reviews = re.sub(r'\s+', ' ', reviews)
    reviews = re.sub(r'\W*\b\w{1,3}\b', "",reviews)
    reviews = reviews.strip()
    
  
    return reviews


for i in range(len(x)) :
    x[i] = cleaned_text(x[i])


# remove_stopwords


def remove_stopwords(rev):
    
    stop_words = set(stopwords.words('english'))
    
    tokens = word_tokenize(rev)
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token not in stop_words]
    reviews = ' '.join(tokens)
    
    return reviews

for i in range(len(x)) :
    x[i] = remove_stopwords(x[i])
    
    
# lemmatize  
    
  
def lemmatize(rev):
    
    doc = nlp(rev)
    reviews = [words.lemma_ for words in doc]
    reviews = ' '.join(reviews)
    
    return reviews


for i in range(len(x)) :
    x[i] = lemmatize(x[i])



# new data_frame for ann cnn and rnn

new_data = data.copy()

file_path = r'C:\Users\asbpi\Desktop\Nit_DS & AI\MY Projects\project_sentiment analysis\new_data.csv'
new_data.to_csv(file_path, index=False)

