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
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.learning_curve import learning_curve
import sys
from sknn.mlp import Regressor, Layer, Classifier

def split_into_tokens(message):
    message = unicode(message, 'utf8')  # convert bytes into proper unicode
    return TextBlob(message).words
    
def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]


check_messages = []
line = ''
if( len(sys.argv) >= 2 ):
	with open(sys.argv[1]) as f:
		for line in f:
			check_messages.append(line)
else:
	print " For multiple SMS testing < USAGE: python spamham.py <test_file_name> >"
	print "OR"
	while (line != 'q'):
		line = raw_input("Enter all texts now (Enter q as the last text message): ")
		check_messages.append(line)
	check_messages.pop()
	
messages = [line.rstrip() for line in open('./data/SMSSpamCollection')]
#messages = [line.rstrip() for line in open('./data/spam')]
#print len(messages)

'''
for message_no, message in enumerate(messages[:10]):
    print message_no, message
'''
    
messages = pandas.read_csv('./data/SMSSpamCollection', sep='\t', quoting=csv.QUOTE_NONE, names=["label", "message"])
#print messages

#print messages.groupby('label').describe()

messages['length'] = messages['message'].map(lambda text: len(text))
#print messages.head()

messages.length.plot(bins=20, kind='hist')
#print messages.length.describe()
#print list(messages.message[messages.length > 900])
messages.hist(column='length', by='label', bins=50)

#print messages.message.head()
#print messages.message.head().apply(split_into_tokens)
#print TextBlob("Hello world, how is it going?").tags  # list of (word, POS) pairs
#print messages.message.head().apply(split_into_lemmas)

bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['message'])

#print len(bow_transformer.vocabulary_)
'''
message4 = messages['message'][3]
print message4
bow4 = bow_transformer.transform([message4])
print bow4
print bow4.shape
print bow_transformer.get_feature_names()[6736]
print bow_transformer.get_feature_names()[8013]
'''

messages_bow = bow_transformer.transform(messages['message'])
#print 'sparse matrix shape:', messages_bow.shape
#print 'number of non-zeros:', messages_bow.nnz
#print 'sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))

tfidf_transformer = TfidfTransformer().fit(messages_bow)
messages_tfidf = tfidf_transformer.transform(messages_bow)
#print messages_tfidf.shape

spam_detector = MultinomialNB().fit(messages_tfidf, messages['label'])
all_predictions = spam_detector.predict(messages_tfidf)
#print all_predictions

#print 'accuracy', accuracy_score(messages['label'], all_predictions)
#print 'confusion matrix\n', confusion_matrix(messages['label'], all_predictions)
#print '(row=expected, col=predicted)'
'''
plt.matshow(confusion_matrix(messages['label'], all_predictions), cmap=plt.cm.binary, interpolation='nearest')
plt.title('confusion matrix')
plt.colorbar()
plt.ylabel('expected label')
plt.xlabel('predicted label')
print classification_report(messages['label'], all_predictions)
'''
msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)
#print len(msg_train), len(msg_test), len(msg_train) + len(msg_test)

msg_test = pandas.read_csv('./data/test', sep='\t', quoting=csv.QUOTE_NONE, names=["message"])
#print msg_test

#check_message = "Last two days left!! Hurry!!"
#print check_message

print
print
print
print
print


for mess in check_messages:
	bow4 = bow_transformer.transform([mess])
	messages_bow = bow_transformer.transform(messages['message'])
	tfidf_transformer = TfidfTransformer().fit(messages_bow)
	tfidf4 = tfidf_transformer.transform(bow4)
	print mess,
	print '-> predicted Spam/Ham:', spam_detector.predict(tfidf4)[0]
	print

'''
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

scores = cross_val_score(pipeline,  # steps to convert raw messages into models
                         msg_train,  # training data
                         label_train,  # training labels
                         cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                         scoring='accuracy',  # which scoring metric?
                         n_jobs=-1,  # -1 = use all cores = faster
                         )
                     
print scores
print scores.mean(), scores.std()
'''
