# Paul Kelly
# DATS 6312
# Homework 4
# 10-2-23

import spacy
nlp = spacy.load("en_core_web_sm")
import pandas as pd

######### E1 #########
#i
#load the data
df = pd.read_csv('data.csv')

#ii
#tokenize one of the titles and extract attributes
#select row 0 and title field
title = df.iloc[0]['title']
#make title field the doc
doc = nlp(title)

#create output dataframe
attrs = ['text', 'idx', 'lemma_', 'is_punct', 'is_space', 'shape_', 'pos_', 'tag_']
rows = [[token.text, token.idx, token.lemma_, token.is_punct, token.is_space,
         token.shape_, token.pos_, token.tag_] for token in doc]

df_attrs = pd.DataFrame(rows, columns=attrs)

print(df_attrs)

#iii
#find named entities
for ent in doc.ents:
    print(ent.text, ent.label_)

#iv
#select a different row in the dataframe
title = df.iloc[7]['title']
doc = nlp(title)

#noun phrase chunking
chunks = []
for chunk in doc.noun_chunks:
    chunks.append((chunk.text, chunk.root.text, chunk.root.pos_, [token.text for token in chunk]))

print(chunks)

#v
#set sentence as new doc
doc = nlp('"The Shining" is a motion picture directed by American filmmaker Stanley Kubrick.')
#analyze grammatical structure of sentence with dependency parsing
print([(token.text, token.dep_, token.head.text, token.head.pos_, [child for child in token.children]) for token in doc])

#vi
#load spacy large
nlp = spacy.load('en_core_web_lg')

#set words as docs
word1 = nlp("Japanese")
word2 = nlp("Scottish")
word3 = nlp("Cantonese")

#compare word similarities
print(word1.similarity(word2))
print(word3.similarity(word2))

######### E2 #########

#load tweets data
df = pd.read_csv('data1.csv')

nlp = spacy.load('en_core_web_sm')
tweet = df.iloc[1656]['text']
doc = nlp(tweet)

#find named entities
for ent in doc.ents:
    print(ent.text, ent.label_)

#explain them
print(spacy.explain("GPE"))
print(spacy.explain("WORK_OF_ART"))
print(spacy.explain("ORG"))
print(spacy.explain("NORP"))
print(spacy.explain("PERSON"))

#redact person name from the tweet

entities = [(ent.text, ent.label_) for ent in doc.ents]

for ent_text, ent_label in entities:
    if ent_label == 'PERSON':
        tweet = tweet.replace(ent_text, '[REDACTED]')

#print redacted tweet
print(tweet)

######### E3 #########
#load large model
nlp = spacy.load('en_core_web_lg')

#apply part of speech tags
doc1 = nlp("'The Shining' is a motion picture directed by American filmmaker Stanley Kubrick.")
for token in doc1:
    print(token.text, token.pos_)

#show syntactic dependencies
for token in doc:
    print(token.text, token.pos_, token.dep_, token.head.text)

#show named entities

doc2 = nlp("Apple is looking at buying U.K. startup for 1 billion dollar")

for ent in doc2.ents:
    print(ent.text, ent.label_)

#show similarities between two sentences

print(doc1.similarity(doc2))

# =================================================================
# Class_Ex1:
# Import spacy abd from the language class import english.
# Create a doc object
# Process a text : This is a simple example to initiate spacy
# Print out the document text from the doc object.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q1' + 20 * '-')

print(20 * '-' + 'End Q1' + 20 * '-')
# =================================================================
# Class_Ex2:
# Solve Ex1 but this time use German Language.
# Grab a sentence from german text from any website.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q2' + 20 * '-')

print(20 * '-' + 'End Q2' + 20 * '-')
# =================================================================
# Class_Ex3:
# Tokenize a sentence using sapaCy.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q3' + 20 * '-')

print(20 * '-' + 'End Q3' + 20 * '-')
# =================================================================
# Class_Ex4:
# Use the following sentence as a sample text. and Answer the following questions.
# "In 2020, more than 15% of people in World got sick from a pandemic ( www.google.com ). Now it is less than 1% are. Reference ( www.yahoo.com )"
# 1- Check if there is a token resemble a number.
# 2- Find a percentage in the text.
# 3- How many url is in the text.

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q4' + 20 * '-')

print(20 * '-' + 'End Q4' + 20 * '-')
# =================================================================
# Class_Ex5:
# Load small web english model into spaCy.
# USe the following text as a sample text. Answer the following questions
# "It is shown that: Google was not the first search engine in U.S. tec company. The value of google is 100 billion dollar"
# 1- Get the token text, part-of-speech tag and dependency label.
# 2- Print them in a tabular format.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q5' + 20 * '-')

print(20 * '-' + 'End Q5' + 20 * '-')

# =================================================================
# Class_Ex6:
# Use Ex 5 sample text and find all the entities in the text.

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q6' + 20 * '-')

print(20 * '-' + 'End Q6' + 20 * '-')
# =================================================================
# Class_Ex7:
# Use SpaCy and find adjectives plus one or 2 nouns.
# Use the following Sample text.
# Features of the iphone applications include a beautiful design, smart search, automatic labels and optional voice responses.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q7' + 20 * '-')

print(20 * '-' + 'End Q7' + 20 * '-')
# =================================================================
# Class_Ex8:
# Use spacy lookup table and find the hash id for a cat
# Text : I have a cat.
# Next use the id and find the strings.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q8' + 20 * '-')

print(20 * '-' + 'End Q8' + 20 * '-')
# =================================================================
# Class_Ex9:
# Create a Doc object for the following sentence
# Spacy is a nice toolkit.
# Use the methods like text, token,... on the Doc and check the functionality.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q9' + 20 * '-')

print(20 * '-' + 'End Q9' + 20 * '-')
# =================================================================
# Class_Ex10:
# Use spacy and process the following text.
# Newyork looks like a nice city.
# Find which token is proper noun and which one is a verb.
#

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q10' + 20 * '-')

print(20 * '-' + 'End Q10' + 20 * '-')
# =================================================================
# Class_Ex11:
# Read the list of countries in a json format.
# Use the following text as  sample text.
# Czech Republic may help Slovakia protect its airspace
# Use statistical method and rule based method to find the countries.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q11' + 20 * '-')

import spacy
import json
nlp = spacy.load("en_core_web_sm")
text = "Czech Republic may help Slovakia protect its airspace"
doc = nlp(text)
countries = []

for ent in doc.ents:
    if ent.label_ == "GPE":
        countries.append(ent.text)

result = {"countries": countries}

json_result = json.dumps(result)

print(json_result)

print(20 * '-' + 'End Q11' + 20 * '-')
# =================================================================
# Class_Ex12:
# Use spacy attributions and answer the following questions.
# Define the getter function that takes a token and returns its reversed text.
# Add the Token property extension "reversed" with the getter function
# Process the text and print the results.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q12' + 20 * '-')

print(20 * '-' + 'End Q12' + 20 * '-')
# =================================================================
# Class_Ex13:
# Read the tweets json file.
# Process the texts and print the entities
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q13' + 20 * '-')

print(20 * '-' + 'End Q13' + 20 * '-')
# =================================================================
# Class_Ex14:
# Use just spacy tokenization. for the following text
# "Burger King is an American fast food restaurant chain"
# make sure other pipes are disabled and not used.
# Disable parser and tagger and process the text. Print the tokens
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q14' + 20 * '-')

print(20 * '-' + 'End Q14' + 20 * '-')

# =================================================================
