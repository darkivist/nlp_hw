# Paul Kelly
# DATS 6312
# Homework 5
# 10-9-23

import re

# E.1
#open email.txt
with open('Email.txt') as f:
    text = f.read()

#finds all email addresses in the text
emails = re.findall(r'[\w\.-]+@[\w\.-]+', text)
#prints results
print(emails)

# E.2
import collections
#open war_and_peace.txt
with open('war_and_peace.txt') as f:
    text = f.read()

#finds all names ending in ski
names = re.findall(r'\b\w*ski\b', text)
#creates ordered set of results
names = sorted(set(names))
#prints set
print(names)

# E.3

#creates text
text = '12 0 mph is a very high speed in the 6 6 interstate.'
#regex to remove spaces between digits
text = re.sub(r'(\d+) (\d+)', r'\1\2', text)
#printing result
print(text)

#creates text
text = 'The price is (100) dollars.'
#regex to find any parentheses with a number inside and replace with (xxxxx)
text = re.sub(r'\(\d+\)', '(xxxxx)', text)
#printing result
print(text)

#creates text
text = 'My cats are named Bert and Blakely.'
#regex to find all words ending in ly.
words = re.findall(r'\w+ly\b', text)
#printing result
print(words)

#creates text
text = 'I say "goodbye" and you say "hello"'
#regex that finds text inside the quotes
quotes = re.findall(r'\"(.+?)\"', text)
#printing result
print(quotes)

#regex that finds words with three, four, or five characters
words = re.findall(r'\b\w{3,5}\b', text)
#printing result
print(words)

#creates text
text = "This was a comma, now is a hyphen."
#regex that replaces commas with hyphens in a string
string = re.sub(r',', '-', text)
print(string)

url = 'https://www.yahoo.com/news/football/wew/2021/09/02/odell--famer-rrrr-on-one-tr-littleball--norman-stupid-author/'
date = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', url)
if date:
    year, month, day = date.groups()
    print(year, month, day)

# =================================================================
# Class_Ex1:
# Write a function that checks a string contains only a certain set of characters
# (all chars lower and upper case with all digits).
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q1' + 20 * '-')

import string

def check_chars(text):
  allowed = string.ascii_letters + string.digits
  return set(text) <= set(allowed)

string1 = "hello123WORLD"
print(check_chars(string1)) # True

string2 = "hello123WORLD@#"
print(check_chars(string2)) # False

print(20 * '-' + 'End Q1' + 20 * '-')

# =================================================================
# Class_Ex2:
# Write a function that matches a string in which a followed by zero or more b's.
# Sample String 'ac', 'abc', abbc'
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q2' + 20 * '-')

def match_ab(text):
  return re.match(r'a(b*)', text)

print(match_ab("ac")) # match object
print(match_ab("abc")) # match object
print(match_ab("abbc")) # match object

print(match_ab("abcd")) # None
print(match_ab("b")) # None

print(20 * '-' + 'End Q2' + 20 * '-')
# =================================================================
# Class_Ex3:
# Write Python script to find numbers between 1 and 3 in a given string.

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q3' + 20 * '-')


text = "123 python 456"

numbers = re.findall(r'[1-3]', text)
print(numbers) # ['1', '2', '3']

print(20 * '-' + 'End Q3' + 20 * '-')
# =================================================================
# Class_Ex4:
# Write a Python script to find the position of the substrings within a string.
# text = 'Python exercises, JAVA exercises, C exercises'
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q4' + 20 * '-')

text = 'Python exercises, JAVA exercises, C exercises'
print(text.index('exercises')) # 7

print(20 * '-' + 'End Q4' + 20 * '-')
# =================================================================
# Class_Ex5:
# Write a Python script to find if two strings from a list starting with letter 'C'.
# words = ["Cython CHP", "Java JavaScript", "PERL S+"]
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q5' + 20 * '-')

words = ["Cython CHP", "Java JavaScript", "PERL S+"]

c_words = [w for w in words if w.startswith('C')]
print(c_words) # ['Cython CHP']

print(20 * '-' + 'End Q5' + 20 * '-')

# =================================================================
# Class_Ex6:
# Write a Python script to remove everything except chars and digits from a string.
# USe sub method
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q6' + 20 * '-')

text = "Hello 123 world!!"
cleaned = re.sub(r'[^a-zA-Z0-9]', '', text)
print(cleaned) # Hello123world

print(20 * '-' + 'End Q6' + 20 * '-')
# =================================================================
# Class_Ex7:
# Scrape the following website
# https://en.wikipedia.org/wiki/Natural_language_processing
# Find the tag which related to the text. Extract all the textual data.
# Tokenize the cleaned text file.
# print the len of the corpus and pint couple of the sentences.
# Calculate the words frequencies.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q7' + 20 * '-')

from urllib import request
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.probability import FreqDist

#grab and decode html
url = "https://en.wikipedia.org/wiki/Natural_language_processing"

html = request.urlopen(url).read().decode('utf8')

#clean html
text = BeautifulSoup(html, 'html.parser').get_text()

tokens = word_tokenize(text)
print(len(tokens))
print(tokens[:2])

fdist = FreqDist(tokens)
print(fdist.most_common(10))

print(20 * '-' + 'End Q7' + 20 * '-')
# =================================================================
# Class_Ex8:
# Grab any text from Wikipedia and create a string of 3 sentences.
# Use that string and calculate the ngram of 1 from nltk package.
# Use BOW method and compare the most 3 common words.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q8' + 20 * '-')

from nltk import ngrams
from nltk.tokenize import TreebankWordTokenizer
from collections import Counter

text = "Stanley Kubrick was an American film director, producer, screenwriter and photographer. Widely considered one of the greatest filmmakers of all time, his films—nearly all of which are adaptations of novels or short stories—span a number of genres and are known for their intense attention to detail, innovative cinematography, extensive set design and dark humor. Kubrick was raised in the Bronx, New York City, and attended William Howard Taft High School from 1941 to 1945."

bigrams = ngrams(text.split(), 2)
print(list(bigrams))

# Bag of words
tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(text.lower())

bag_of_words = Counter(tokens)
print(bag_of_words.most_common(3))

print(20 * '-' + 'End Q8' + 20 * '-')
# =================================================================
# Class_Ex9:
# Write a python script that accepts any string and do the following.
# 1- Tokenize the text
# 2- Doe word extraction and clean a text. Use regular expression to clean a text.
# 3- Generate BOW
# 4- Vectorized all the tokens.
# 5- The only package you can use is numpy and re.
# all sentences = ["sentence1", "sentence2", "sentence3",...]
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q9' + 20 * '-')

import re
import numpy as np
from nltk.tokenize import word_tokenize

text = "Stanley Kubrick was an American film director, producer, screenwriter and photographer. Widely considered one of the greatest filmmakers of all time, his films—nearly all of which are adaptations of novels or short stories—span a number of genres and are known for their intense attention to detail, innovative cinematography, extensive set design and dark humor. Kubrick was raised in the Bronx, New York City, and attended William Howard Taft High School from 1941 to 1945."

# Tokenize and clean
tokens = word_tokenize(text)
tokens = [re.sub(r'[^a-zA-Z0-9]', '', w) for w in tokens]
print(tokens)

# BOW
bag_of_words = Counter(tokens)

# Vectorize
vocab = sorted(bag_of_words)
vectorized = [vocab.index(w) for w in tokens if w in vocab]

print(vectorized)

print(20 * '-' + 'End Q9' + 20 * '-')
# =================================================================
# Class_Ex10:
# Grab any text (almost a paragraph) from Wikipedia and call it text
# Preprocessing the text data (Normalize, remove special char, ...)
# Find total number of unique words
# Create an index for each word.
# Count number of the words.
# Define a function to calculate Term Frequency
# Define a function calculate Inverse Document Frequency
# Combining the TF-IDF functions
# Apply the TF-IDF Model to our text
# you are allowed to use just numpy and nltk tokenizer
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q10' + 20 * '-')

# Insert code to scrape Wikipedia text

import re
from nltk.tokenize import word_tokenize

text = "Stanley Kubrick was an American film director, producer, screenwriter and photographer. Widely considered one of the greatest filmmakers of all time, his films—nearly all of which are adaptations of novels or short stories—span a number of genres and are known for their intense attention to detail, innovative cinematography, extensive set design and dark humor. Kubrick was raised in the Bronx, New York City, and attended William Howard Taft High School from 1941 to 1945."

# Preprocess
tokens = word_tokenize(text.lower())
tokens = [re.sub(r'[^a-zA-Z0-9]', '', w) for w in tokens]

# Unique words
unique_words = set(tokens)
print(len(unique_words))

# Index and count
word_index = {w: i for i, w in enumerate(unique_words)}
word_count = {w: tokens.count(w) for w in unique_words}

# TF and IDF functions
def tf(term, doc):
    return doc.count(term) / len(doc)

def idf(term, docs):
     return np.log(len(docs) / (1 + list(docs).count(term))) + 1

# TF-IDF
tfidf = {t: tf(t, tokens) * idf(t, tokens) for t in unique_words}

print(tfidf)

print(20 * '-' + 'End Q10' + 20 * '-')
# =================================================================
# Class_Ex11:
# Grab arbitrary paragraph from any website.
# Creat  a list of stopwords manually.  Example :  stopwords = ['and', 'for', 'in', 'little', 'of', 'the', 'to']
# Create a list of ignore char Example: ' :,",! '
# Write a LSA class with the following functions.
# Parse function which tokenize the words lower cases them and count them. Use dictionary; keys are the tokens and value is count.
# Clac function that calculate SVD.
# TFIDF function
# Print function which print out the TFIDF matrix, first 3 columns of the U matrix and first 3 rows of the Vt matrix
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q11' + 20 * '-')


import re
from nltk.tokenize import word_tokenize
from sklearn.decomposition import TruncatedSVD
import numpy as np

text = "Stanley Kubrick was an American film director, producer, screenwriter and photographer. Widely considered one of the greatest filmmakers of all time, his films—nearly all of which are adaptations of novels or short stories—span a number of genres and are known for their intense attention to detail, innovative cinematography, extensive set design and dark humor. Kubrick was raised in the Bronx, New York City, and attended William Howard Taft High School from 1941 to 1945."

class LSA:

    def __init__(self, text):
        self.text = text
        self.stopwords = ['and', 'for', 'in', 'little', 'of', 'the', 'to']
        self.ignore = ' :,",! '

    def parse(self):
        tokens = word_tokenize(self.text.lower())
        tokens = [re.sub(r'[^a-zA-Z0-9]', '', w) for w in tokens if w not in self.stopwords and w not in self.ignore]
        self.freq = {w: tokens.count(w) for w in set(tokens)}

    def calc_lsa(self):
        A = np.array([[self.freq[w] for w in self.freq]] * len(self.freq))
        self.lsa = TruncatedSVD(n_components=2).fit_transform(A)

    def print_lsa(self):
        print(self.lsa[:, :3])


lsa = LSA(text)
lsa.parse()
lsa.calc_lsa()
lsa.print_lsa()

print(20 * '-' + 'End Q11' + 20 * '-')
# =================================================================
# Class_Ex12:
# Use the following doc
#  = ["An intern at OpenAI", "Developer at OpenAI", "A ML intern", "A ML engineer" ]
# Calculate the binary BOW.
# Use LSA method and distinguish two different topic from the document. Sent 1,2 is about OpenAI and sent3, 4 is about ML.
# Use pandas to show the values of dataframe and lsa components. Show there is two distinct topic.
# Use numpy take the absolute value of the lsa matrix sort them and use some threshold and see what words are the most important.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q12' + 20 * '-')

#it seems like there are THREE concepts here: openAI, ML, and interns

import pandas as pd

doc1 = "An intern at OpenAI"
doc2 = "Developer at OpenAI"
doc3 = "A ML intern"
doc4 = "A ML engineer"

doc_complete = [doc1, doc2, doc3, doc4]

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(binary=True, stop_words='english')
X =vectorizer.fit_transform(doc_complete)

from sklearn.preprocessing import Normalizer
X = Normalizer().fit_transform(X)


from sklearn.decomposition import TruncatedSVD
lsa = TruncatedSVD(n_components=2, n_iter=100)
lsa.fit(X)
terms = vectorizer.get_feature_names_out()

for i,comp in enumerate(lsa.components_):
    termsInComp = zip(terms,comp)
    sortedterms = sorted(termsInComp, key=lambda x: x[1],reverse=True)[:10]
    print("Concept %d:" % i)
    for term in sortedterms:
        print(term[0])
    print(" ")

# Convert the document-term matrix and LSA components to Pandas DataFrames
dtm_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
lsa_components_df = pd.DataFrame(lsa.components_, columns=vectorizer.get_feature_names_out())

# Print the DataFrames
print("Document-Term Matrix:")
print(dtm_df)

print("\nLSA Components:")
print(lsa_components_df)

#########

# Get the LSA components
lsa_components = lsa.components_

# Create a DataFrame to store the LSA components
lsa_components_df = pd.DataFrame(lsa_components, columns=vectorizer.get_feature_names_out())

# Take the absolute values of the LSA components
lsa_components_abs = np.abs(lsa_components_df)

# Set a threshold to determine word importance
threshold = 0.2  # Adjust this threshold as needed

# Iterate through each concept and find the important words
important_words_by_concept = {}
for concept_idx in range(len(lsa_components_abs)):
    concept = lsa_components_abs.iloc[concept_idx]
    important_words = concept[concept >= threshold].index.tolist()
    important_words_by_concept[f"Concept {concept_idx + 1}"] = important_words

# Print the important words for each concept
for concept, words in important_words_by_concept.items():
    print(f"{concept}: {', '.join(words)}")

print(20 * '-' + 'End Q12' + 20 * '-')
# =================================================================
