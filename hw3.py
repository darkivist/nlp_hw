# Paul Kelly
# DATS 6312
# Homework 3
# 9-25-23

# E1

import nltk
from nltk import word_tokenize
from nltk.corpus import wordnet

# load moby.txt
f = open('moby.txt')
raw = f.read()

# tokenize text
tokens = word_tokenize(raw)
print("here we go!")
# print total tokens
print("total tokens:", (len(tokens)))
# print unique tokens
# could we use hapaxes here? ASK TA tomorrow
print("total unique tokens:", (len(set(tokens))))
# tag verb tokens with part of speech
wnl = nltk.WordNetLemmatizer()


# source: https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/

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

# lexical diversity
moby_diversity = float(len(set(raw))) / float(len(raw))
print("The lexical diversity is: ", moby_diversity * 100, "%")

# what percentage of corpus is "whale"?

print("Whale and whale are", (tokens.count("Whale") + tokens.count("whale")) / (len(tokens)) * 100, "% of tokens")

# 20 most frequently occurring tokens plus their frequency

from nltk import FreqDist

Freq_Dist = FreqDist(tokens)

print("20 most frequently occurring tokens plus their frequencies are:", Freq_Dist.most_common(20))
# tokens with length of greater than 6 and frequency of more than 160
#source: https://stackoverflow.com/questions/45730590/filtering-tokens-from-a-list-by-multiple-conditions
from collections import Counter

c = Counter(tokens)
selected_tokens = [x for x in set(tokens) if len(x) > 6 and c[x] > 160]
print("Tokens with length of > 6 and frequency of > 160:",selected_tokens)

# longest word in the text and its length

#find length of longest token
longest_len = max(len(s) for s in tokens); print(longest_len)
print("length of longest token is:",longest_len,"characters")
longest_token = [x for x in set(tokens) if len(x) >= longest_len]
print("the longest word in the text is:", longest_token)

#What unique words have a frequency of more than 2000? What is their frequency?
#COME BACK TO THIS

#more_tokens = [x for x in set(tokens) if c[x] > 2000]
#print(more_tokens)
Freq_Dist_1 = FreqDist(x for x in set(tokens) if c[x] > 2000)

print(c)
#wordz = Freq_Dist_1.hapaxes() ; print(wordz)

#Freq_Dist.hapaxes()
#Once_happend= Freq_Dist.hapaxes() ; print(Once_happend)

#average tokens per sentence
#source: https://stackoverflow.com/questions/42144071/counting-avg-number-of-words-per-sentence
sentences = [[]]
ends = set(".?!")
for token in tokens:
    if token in ends: sentences.append([])
    else: sentences[-1].append(token)

if sentences[0]:
    if not sentences[-1]: sentences.pop()
    print("average tokens per sentence:", sum(len(s) for s in sentences)/len(sentences))

#5 most frequent parts of speech in corpus

#combine pos with freq_dist

pos_list = nltk.pos_tag(tokens)
pos_counts = nltk.FreqDist(tag for (word, tag) in pos_list)
print("the six most common parts of speech tags in the corpus are", pos_counts.most_common(6))
print("the five most common parts of speech in the text are:")
print(nltk.help.upenn_tagset('NN'))
print(nltk.help.upenn_tagset('IN'))
print(nltk.help.upenn_tagset('DT'))
print(nltk.help.upenn_tagset('JJ'))
print(nltk.help.upenn_tagset('RB'))



