# =================================================================
# Class_Ex1:
#  Use the following dataframe as the sample data.
# Find the conditional probability of Char given the Occurrence.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q1' + 20 * '-')

import pandas as pd
import numpy as np

df = pd.DataFrame(
    {'Char': ['f', 'b', 'f', 'b', 'f', 'b', 'f', 'f'], 'Occurance': ['o1', 'o1', 'o2', 'o3', 'o2', 'o2', 'o1', 'o3'],
     'C': np.random.randn(8), 'D': np.random.randn(8)})

print(df.groupby(['Char', 'Occurance']).size().unstack(fill_value=0))

print(20 * '-' + 'End Q1' + 20 * '-')

# =================================================================
# Class_Ex2:
# Use the following dataframe as the sample data.
# Find the conditional probability occurrence of thw word given a sentiment.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q2' + 20 * '-')

df1 = pd.DataFrame({'Word': ['Good', 'Bad', 'Awesome', 'Beautiful', 'Terrible', 'Horrible'],
                    'Occurrence': ['One', 'Two', 'One', 'Three', 'One', 'Two'],
                    'sentiment': ['P', 'N', 'P', 'P', 'N', 'N'], })


print(df1.groupby(['Occurrence', 'sentiment']).size().unstack(fill_value=0))

print(20 * '-' + 'End Q2' + 20 * '-')
# =================================================================
# Class_Ex3:
# Read the data.csv file.
# Answer the following question
# 1- In this dataset we have a lot of responses in text and each response has a label.
# 2- Our goal is to correctly model the texts into its label.
# Hint: you need to read the text responses and perform preprocessing on it.
# such as normalization, legitimation, cleaning, stopwords removal and POS tagging.
# then use any methods you learned in the lecture to convert each response into meaningful numbers.
# 3- Apply Naive bayes and look at appropriate evaluation metric.
# 4- Explain your results very carefully.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q3' + 20 * '-')

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

#read in data

data = pd.read_csv('data.csv', encoding='latin1')
print(data)

#preprocess text column
stop_words = stopwords.words('english')
lemmatizer = nltk.WordNetLemmatizer()

nltk.download('averaged_perceptron_tagger')

#create preprocessing function
#be aware that removing punctuation and stopwords and overlemmatizing CAN reduce accuracy of model predictions
#here, linking the pos tags to wordnet tags in an attempt to make lemmatization contextually aware actually makes model perform poorer
def preprocess(text):
    #convert text to text lowercase (normalize)
    text = text.lower()
    #remove punctuation w/regular expression
    text = re.sub(r'[^\w\s]', '', text)
    #tokenize words
    words = word_tokenize(text)
    #part of speech tagging
    pos_tags = nltk.pos_tag(words)
    #lemmatize non-stop words based on pos tags
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags if word not in stop_words]
    return " ".join(lemmatized_words)

def get_wordnet_pos(tag):
    #map POS tags to wordnet tags
    tag = tag[0].upper()
    tag_dict = {"J": nltk.corpus.wordnet.ADJ,
                "N": nltk.corpus.wordnet.NOUN,
                "V": nltk.corpus.wordnet.VERB,
                "R": nltk.corpus.wordnet.ADV}
    return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)

#call preprocessing function on 'text' column
data['text'] = data['text'].apply(preprocess)

#encode labels
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

print(data)

#test/train split
X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#vectorize to convert each response into meaningful numbers
vectorizer = TfidfVectorizer()  # You can use other vectorizers as well
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

#apply naive bayes classifier
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_vectorized, y_train)

#evaluate model on test
y_pred = naive_bayes.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

#explain results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_rep)

#save model to pickle
#REMEMBER - PICKLE SAVES TO AWS, NOT LOCAL. YOU HAVE TO DOWNLOAD IT.
with open('naive_bayes_model.pkl', 'wb') as model_file:
    pickle.dump(naive_bayes, model_file)

print(20 * '-' + 'End Q3' + 20 * '-')
# =================================================================
# Class_Ex4:
# Use Naive bayes classifier for this problem,
# Write a text classification pipeline to classify movie reviews as either positive or negative.
# Find a good set of parameters using grid search. hint: grid search on n gram
# Evaluate the performance on a held out test set.
# hint1: use nltk movie reviews dataset
# from nltk.corpus import movie_reviews

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q4' + 20 * '-')

import nltk
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk import word_tokenize
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

#import dataset
nltk.download('movie_reviews')

#load dataset
positive_docs = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids('pos')]
negative_docs = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids('neg')]
reviews = positive_docs + negative_docs
labels = [1] * len(positive_docs) + [0] * len(negative_docs)

#preprocess reviews text
stop_words = stopwords.words('english')
lemmatizer = nltk.WordNetLemmatizer()

#create preprocessing function
#be aware that removing punctuation and stopwords and overlemmatizing CAN reduce accuracy of model predictions
#here, linking the pos tags to wordnet tags in an attempt to make lemmatization contextually aware actually makes model perform poorer
def preprocess(text):
    #convert text to text lowercase (normalize)
    text = text.lower()
    #remove punctuation w/regular expression
    text = re.sub(r'[^\w\s]', '', text)
    #tokenize words
    words = word_tokenize(text)
    #part of speech tagging
    pos_tags = nltk.pos_tag(words)
    #lemmatize non-stop words based on pos tags
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags if word not in stop_words]
    return " ".join(lemmatized_words)

def get_wordnet_pos(tag):
    #map POS tags to wordnet tags
    tag = tag[0].upper()
    tag_dict = {"J": nltk.corpus.wordnet.ADJ,
                "N": nltk.corpus.wordnet.NOUN,
                "V": nltk.corpus.wordnet.VERB,
                "R": nltk.corpus.wordnet.ADV}
    return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)

#call preprocessing function on reviews text
reviews = [preprocess(review) for review in reviews]

#test/train split
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)

#Create text classification pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

#grid search for optimal parameters
param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],  # Try different n-gram ranges
    'clf__alpha': [0.1, 1.0, 10.0]  # Try different alpha values
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

#use best parameters from grid search
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

#evaluate model on test
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=['negative', 'positive'])

#explain results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_rep)

#save model to pickle
#REMEMBER - PICKLE SAVES TO AWS, NOT LOCAL. YOU HAVE TO DOWNLOAD IT.

with open('best_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

print(20 * '-' + 'End Q4' + 20 * '-')
# =================================================================
# Class_Ex5:
# Calculate accuracy percentage between two lists
# calculate a confusion matrix
# Write your own code - No packages
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q5' + 20 * '-')

import numpy as np

y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 1, 0, 1, 0, 1]

accuracy = np.sum(y_true == y_pred) / len(y_true)
print("Accuracy:", accuracy)

confusion_matrix = np.array([[np.sum((y_true == 1) & (y_pred == 1)),
                           np.sum((y_true == 0) & (y_pred == 1))],
                          [np.sum((y_true == 1) & (y_pred == 0)),
                           np.sum((y_true == 0) & (y_pred == 0))]])
print("Confusion Matrix:\n", confusion_matrix)

print(20 * '-' + 'End Q5' + 20 * '-')
# =================================================================
# Class_Ex6:
# Read the data.csv file.
# Answer the following question
# 1- In this dataset we have a lot of responses in text and each response has a label.
# 2- Our goal is to correctly model the texts into its label.
# Hint: you need to read the text responses and perform preprocessing on it.
# such as normalization, legitimation, cleaning, stopwords removal and POS tagging.
# then use any methods you learned in the lecture to convert each response into meaningful numbers.
# 3- Apply Logistic Regression  and look at appropriate evaluation metric.
# 4- Apply LSA method and compare results.
# 5- Explain your results very carefully.

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q6' + 20 * '-')

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import pickle
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


#read in data
data = pd.read_csv('data.csv', encoding='latin1')

#preprocess text column
stop_words = stopwords.words('english')
lemmatizer = nltk.WordNetLemmatizer()

nltk.download('averaged_perceptron_tagger')

#create preprocessing function
#be aware that removing punctuation and stopwords and overlemmatizing CAN reduce accuracy of model predictions
#here, linking the pos tags to wordnet tags in an attempt to make lemmatization contextually aware actually makes model perform poorer
def preprocess(text):
    #convert text to text lowercase (normalize)
    text = text.lower()
    #remove punctuation w/regular expression
    text = re.sub(r'[^\w\s]', '', text)
    #tokenize words
    words = word_tokenize(text)
    #part of speech tagging
    pos_tags = nltk.pos_tag(words)
    #lemmatize non-stop words based on pos tags
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags if word not in stop_words]
    return " ".join(lemmatized_words)

def get_wordnet_pos(tag):
    #map POS tags to wordnet tags
    tag = tag[0].upper()
    tag_dict = {"J": nltk.corpus.wordnet.ADJ,
                "N": nltk.corpus.wordnet.NOUN,
                "V": nltk.corpus.wordnet.VERB,
                "R": nltk.corpus.wordnet.ADV}
    return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)

#call preprocessing function on 'text' column
data['text'] = data['text'].apply(preprocess)

#logistic Regression pipeline with tfidf vectorization
pipe_logreg = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

pipe_logreg.fit(data['text'], data['label'])
y_pred = pipe_logreg.predict(data['text'])

#print accuracy for comparison
print("Logistic Regression Accuracy:", accuracy_score(data['label'], y_pred))

#lsa pipeline with tfidf vectorization
lsa = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('lsa', TruncatedSVD(n_components=100)),
    ('clf', LogisticRegression())
])

lsa.fit(data['text'], data['label'])
y_pred = lsa.predict(data['text'])

#print accuracy for comparison
print("LSA Logistic Regression Accuracy:", accuracy_score(data['label'], y_pred))

print(20 * '-' + 'End Q6' + 20 * '-')

# =================================================================
# Class_Ex7:
# Use logistic regression classifier for this problem,
# Write a text classification pipeline to classify movie reviews as either positive or negative.
# Find a good set of parameters using grid search. hint: grid search on n-gram
# Evaluate the performance on a held out test set.
# hint1: use nltk movie reviews dataset
# from nltk.corpus import movie_reviews
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q7' + 20 * '-')

import nltk
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk import word_tokenize
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

#import dataset
nltk.download('movie_reviews')

#load dataset
positive_docs = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids('pos')]
negative_docs = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids('neg')]
reviews = positive_docs + negative_docs
labels = [1] * len(positive_docs) + [0] * len(negative_docs)

#preprocess reviews text
stop_words = stopwords.words('english')
lemmatizer = nltk.WordNetLemmatizer()

#create preprocessing function
#be aware that removing punctuation and stopwords and overlemmatizing CAN reduce accuracy of model predictions
#here, linking the pos tags to wordnet tags in an attempt to make lemmatization contextually aware actually makes model perform poorer
def preprocess(text):
    #convert text to text lowercase (normalize)
    text = text.lower()
    #remove punctuation w/regular expression
    text = re.sub(r'[^\w\s]', '', text)
    #tokenize words
    words = word_tokenize(text)
    #part of speech tagging
    pos_tags = nltk.pos_tag(words)
    #lemmatize non-stop words based on pos tags
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags if word not in stop_words]
    return " ".join(lemmatized_words)

def get_wordnet_pos(tag):
    #map POS tags to wordnet tags
    tag = tag[0].upper()
    tag_dict = {"J": nltk.corpus.wordnet.ADJ,
                "N": nltk.corpus.wordnet.NOUN,
                "V": nltk.corpus.wordnet.VERB,
                "R": nltk.corpus.wordnet.ADV}
    return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)

#call preprocessing function on reviews text
reviews = [preprocess(review) for review in reviews]

#test/train split
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)

#Create text classification pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

#grid search for optimal parameters
param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],  # Try different n-gram ranges
    'clf__C': [0.1, 1.0, 10.0]  # Try different c values
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

#use best parameters from grid search
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

#evaluate model on test
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=['negative', 'positive'])

#explain results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_rep)

#save model to pickle
#REMEMBER - PICKLE SAVES TO AWS, NOT LOCAL. YOU HAVE TO DOWNLOAD IT.

with open('best_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

print(20 * '-' + 'End Q7' + 20 * '-')
