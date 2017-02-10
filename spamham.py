#%matplotlib inline
import warnings
import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas
import sklearn
import cPickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.learning_curve import learning_curve

def split_into_tokens(message):
    message = unicode(message, 'utf8')  # convert bytes into proper unicode
    return TextBlob(message).words
    
def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]


messages = [line.rstrip() for line in open('./data/SMSSpamCollection')]
print len(messages)

'''
for message_no, message in enumerate(messages[:10]):
    print message_no, message
'''
    
messages = pandas.read_csv('./data/SMSSpamCollection', sep='\t', quoting=csv.QUOTE_NONE, names=["label", "message"])
print messages

print messages.groupby('label').describe()

messages['length'] = messages['message'].map(lambda text: len(text))
print messages.head()

messages.length.plot(bins=20, kind='hist')
print messages.length.describe()
print list(messages.message[messages.length > 900])
messages.hist(column='length', by='label', bins=50)

print messages.message.head()
print messages.message.head().apply(split_into_tokens)
print TextBlob("Hello world, how is it going?").tags  # list of (word, POS) pairs
print messages.message.head().apply(split_into_lemmas)
