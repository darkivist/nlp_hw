#Paul Kelly
#DATS 6312
#Homework 3
#9-25-23

#E1

import nltk
from nltk import word_tokenize
from nltk.corpus import wordnet

#load moby.txt
f = open('moby.txt')
raw = f.read()

#tokenize text
tokens = word_tokenize(raw)
print("here we go!")
#print total tokens
print("total tokens:", (len(tokens)))
#print unique tokens
#could we use hapaxes here? ASK TA tomorrow
print("total unique tokens:", (len(set(tokens))))
#tag verb tokens with part of speech
wnl = nltk.WordNetLemmatizer()

#source: https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/

def pos_tagger(nltk_tag):
    if nltk_tag.startswith('V'):
        return wordnet.VERB
    else:
        return None

tagged = nltk.pos_tag(tokens)
wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), tagged))

wnl = nltk.WordNetLemmatizer()

lemmatized_corpus = []
for word, tag in wordnet_tagged:
    if tag is None:
        # if there is no available tag, append the token as is
        lemmatized_corpus.append(word)
    else:
        # else use the tag to lemmatize the token
        lemmatized_corpus.append(wnl.lemmatize(word, tag))
lemmatized_corpus = " ".join(lemmatized_corpus)

print("total unique tokens after verb lemmatization:", (len(set(lemmatized_corpus))))

#lexical diversity
moby_diversity = float(len(set(raw))) / float(len(raw))
print("The lexical diversity is: ", moby_diversity * 100, "%")

#what percentage of corpus is "whale"?

print("Whale and whale are", (tokens.count("Whale")+tokens.count("whale"))/(len(tokens))*100, "% of tokens")

#20 most frequently occurring tokens plus their frequency

from nltk import FreqDist
Freq_Dist = FreqDist(tokens)

print("20 most frequently occurring tokens plus their frequencies are:", Freq_Dist.most_common(20))

#tokens with length of greater than 6 and frequency of more than 160

#long_words = [words for words in tokens if len(words) > 6]
#word_freq = nltk.FreqDist(long_words); print(word_freq.most_common(5))
#print(list(nltk.bigrams(long_words)))