# Paul Kelly
# DATS 6312
# Homework 3
# 9-25-23

# E1

import nltk
from nltk import word_tokenize, sent_tokenize
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

# E2
from urllib import request
from bs4 import BeautifulSoup

#grab and decode html
url = "https://en.wikipedia.org/wiki/Benjamin_Franklin"
html = request.urlopen(url).read().decode('utf8')
#print(html[:60])
#print(html)
#clean html
raw = BeautifulSoup(html, 'html.parser').get_text()
#tokenize html
website_tokens = word_tokenize(raw)

#define function to check if words in a corpus do not appear in Words Corpus and makes a list of them

def unknown(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    novel = text_vocab - english_vocab
    return sorted(novel)

#applying function to scraped, tokenized html
#print("words from website that don't appear in Words Corpus are:", unknown(website_tokens))
novel_words = unknown(website_tokens)
#Use porter stemmer to stem all items in novel_words, saving result as novel-stems
porter = nltk.PorterStemmer()
novel_stems = ([porter.stem(n) for n in novel_words])

#Find as many proper names from novel-stems as possible, saving the result as proper_names

text_vocab = set(w.lower() for w in novel_stems if w.isalpha())
names_vocab = set(w.lower() for w in nltk.corpus.names.words())
proper_names = (sorted([w for w in names_vocab if w in text_vocab]))

print("proper names found in novel stems are:", proper_names)

# E3
#load  data and view  first few sentences.
textfile = open('twitter.txt', "r")
print(textfile.readlines()[0:5])
textfile.seek(0)

#split data into sentences with '\n' as delimiter
data = str(textfile.read().split('\n'))

#tokenize sentences
tokenized_tweets = sent_tokenize(data)
#^^^ still need to convert to lowercase

#print(tokenized_tweets)

#now tokenize words
tokenized_words = word_tokenize(data)
#print(tokenized_words)


#count how many times each word appears
from collections import defaultdict

count = defaultdict(int)
for word in tokenized_words:
    count[word] += 1
print(count)

# =================================================================
# Class_Ex1:
# Use NLTK Book fnd which the related Sense and Sensibility.
# Produce a dispersion plot of the four main protagonists in Sense and Sensibility:
# Elinor, Marianne, Edward, and Willoughby. What can you observe about the different
# roles played by the males and females in this novel? Can you identify the couples?
# Explain the result of plot in a couple of sentences.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q1' + 20 * '-')
from nltk.corpus import gutenberg
import nltk
import matplotlib.pyplot as plt

#import text
sense_text = nltk.Text(gutenberg.words('austen-sense.txt'))

#dispersion plot
sense_text.dispersion_plot(["Elinor", "Marianne", "Edward", "Willoughby"])
plt.show()

#Explain this

print(20 * '-' + 'End Q1' + 20 * '-')

# =================================================================
# Class_Ex2:
# What is the difference between the following two lines of code? Explain in details why?
# Make up and example base don your explanation.
# Which one will give a larger value? Will this be the case for other texts?
# 1- sorted(set(w.lower() for w in text1))
# 2- sorted(w.lower() for w in set(text1))
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q2' + 20 * '-')


print(20 * '-' + 'End Q2' + 20 * '-')
# =================================================================
# Class_Ex3:
# Find all the four-letter words in the Chat Corpus (text5).
# With the help of a frequency distribution (FreqDist), show these words in decreasing order of frequency.
#
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q3' + 20 * '-')
from nltk.book import text5
from nltk import FreqDist

#set up frequency distribution
Freq_Dist = FreqDist(text5)
#get number of samples
print(Freq_Dist)
#print freqdists in descening order of frequency with most_common
print(Freq_Dist.most_common(6066))

print(20 * '-' + 'End Q3' + 20 * '-')
# =================================================================
# Class_Ex4:
# Write expressions for finding all words in text6 that meet the conditions listed below.
# The result should be in the form of a list of words: ['word1', 'word2', ...].
# a. Ending in ise
# b. Containing the letter z
# c. Containing the sequence of letters pt
# d. Having all lowercase letters except for an initial capital (i.e., titlecase)
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q4' + 20 * '-')
from nltk.book import text5

#endswith

ing=[]
for word in text5:
    if word.endswith("ing"):
        ing.append(word)

print(ing)

#contains
z=[]
for word in text5:
    if 'z' in word:
        z.append(word)
print(z)

pt=[]
for word in text5:
    if 'pt' in word:
        pt.append(word)

print(pt)

titlecase=[]
for word in text5:
    if word.istitle():
        titlecase.append(word)

print(titlecase)

print(20 * '-' + 'End Q4' + 20 * '-')
# =================================================================
# Class_Ex5:
#  Read in the texts of the State of the Union addresses, using the state_union corpus reader.
#  Count occurrences of men, women, and people in each document.
#  What has happened to the usage of these words over time?
# Since there would be a lot of document use every couple of years.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q5' + 20 * '-')
from nltk.corpus import inaugural

#Count occurrences of men, women, and people in each document.
cfd = nltk.ConditionalFreqDist(
    (target, fileid[:4])
    for fileid in inaugural.fileids()
    for w in inaugural.words(fileid)
    for target in ['men', 'women', 'people']
    if w.lower().startswith(target))
cfd.tabulate(conditions=['men', 'women', 'people'])

#  What has happened to the usage of these words over time?
cfd.plot()
#usage of these words has reduced over time, but now women and men are mentioned at mostly similar levels,
# but people is still higher

print(20 * '-' + 'End Q5' + 20 * '-')

# =================================================================
# Class_Ex6:
# The CMU Pronouncing Dictionary contains multiple pronunciations for certain words.
# How many distinct words does it contain? What fraction of words in this dictionary have more than one possible pronunciation?
#
#
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q6' + 20 * '-')

entries = nltk.corpus.cmudict.dict()
#find distinct entries
print(len(entries))

#find words with more than one pronounciation
#this is not correct
p3 = []
for word in entries.items():
    if word[-1] == ')' and word[-3] == '(' and word[-2].isdigit():
        p3.append(word)
print(p3)

print(20 * '-' + 'End Q6' + 20 * '-')
# =================================================================
# Class_Ex7:
# What percentage of noun synsets have no hyponyms?
# You can get all noun synsets using wn.all_synsets('n')
#
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q7' + 20 * '-')

from nltk.corpus import wordnet as wn

nouns = wn.all_synsets('n')

nouns_without_hyponyms = []
for noun in nouns:
    if not noun.hyponyms():
        nouns_without_hyponyms.append(noun)

#print(nouns_without_hyponyms)


print(20 * '-' + 'End Q7' + 20 * '-')
# =================================================================
# Class_Ex8:
# Write a program to find all words that occur at least three times in the Brown Corpus.
# USe at least 2 different method.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q8' + 20 * '-')

from nltk.corpus import brown

brown_words = brown.words()
Freq_Dist = FreqDist(brown_words)

#source: https://stackoverflow.com/questions/14179543/how-to-write-a-python-program-that-returns-all-words-that-occur-at-least-5-times
list_3 = [w for w in Freq_Dist.keys() if Freq_Dist[w] >= 3]

print(list_3)

print(20 * '-' + 'End Q8' + 20 * '-')
# =================================================================
# Class_Ex9:
# Write a function that finds the 50 most frequently occurring words of a text that are not stopwords.
# Test it on Brown corpus (humor), Gutenberg (whitman-leaves.txt).
# Did you find any strange word in the list? If yes investigate the cause?
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q9' + 20 * '-')

def remove_stopwords(text):
    stopwords = nltk.corpus.stopwords.words('english')
    vocab = [w for w in text if w.lower() not in stopwords]
    Freq_Dist = FreqDist(vocab)
    return Freq_Dist.most_common(50)

from nltk.corpus import brown
humor_text = brown.words(categories='humor')
print(remove_stopwords(humor_text))
whitman = nltk.Text(gutenberg.words('whitman-leaves.txt'))
print(remove_stopwords(whitman))

print(20 * '-' + 'End Q9' + 20 * '-')
# =================================================================
# Class_Ex10:
# Write a program to create a table of word frequencies by genre, like the one given in 1 for modals.
# Choose your own words and try to find words whose presence (or absence) is typical of a genre. Discuss your findings.

# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q10' + 20 * '-')

cfd = nltk.ConditionalFreqDist((genre, word)
            for genre in brown.categories()
            for word in brown.words(categories=genre))
genres = ['adventure', 'humor', 'lore', 'science_fiction', 'romance', 'learned']
common_words = ['planet', 'blood', 'love', 'idol', 'silly', 'repeat']
print(); print(cfd.tabulate(conditions=genres, samples=common_words))

#add thoughts here

print(20 * '-' + 'End Q10' + 20 * '-')
# =================================================================
# Class_Ex11:
#  Write a utility function that takes a URL as its argument, and returns the contents of the URL,
#  with all HTML markup removed. Use from urllib import request and
#  then request.urlopen('http://nltk.org/').read().decode('utf8') to access the contents of the URL.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q11' + 20 * '-')

print(20 * '-' + 'End Q11' + 20 * '-')
# =================================================================
# Class_Ex12:
# Read in some text from a corpus, tokenize it, and print the list of all
# wh-word types that occur. (wh-words in English are used in questions,
# relative clauses and exclamations: who, which, what, and so on.)
# Print them in order. Are any words duplicated in this list,
# because of the presence of case distinctions or punctuation?
# Note Use: Gutenberg('bryant-stories.txt')
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q12' + 20 * '-')

print(20 * '-' + 'End Q12' + 20 * '-')
# =================================================================
# Class_Ex13:
# Write code to access a  webpage and extract some text from it.
# For example, access a weather site and extract  a feels like temprature..
# Note use the following site https://darksky.net/forecast/40.7127,-74.0059/us12/en
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q13' + 20 * '-')

print(20 * '-' + 'End Q13' + 20 * '-')
# =================================================================
# Class_Ex14:
# Use the brown tagged sentences corpus news.
# make a test and train sentences and then  use bi-gram tagger to train it.
# Then evaluate the trained model.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q14' + 20 * '-')

print(20 * '-' + 'End Q14' + 20 * '-')

# =================================================================
# Class_Ex15:
# Use sorted() and set() to get a sorted list of tags used in the Brown corpus, removing duplicates.
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q15' + 20 * '-')

print(20 * '-' + 'End Q15' + 20 * '-')

# =================================================================
# Class_Ex16:
# Write programs to process the Brown Corpus and find answers to the following questions:
# 1- Which nouns are more common in their plural form, rather than their singular form? (Only consider regular plurals, formed with the -s suffix.)
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q16' + 20 * '-')

print(20 * '-' + 'End Q16' + 20 * '-')
