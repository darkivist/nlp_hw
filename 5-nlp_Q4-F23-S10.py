#sources: nltk lecture notes, homework 5 vectorization section, and code proofed in chatgpt

from nltk.corpus import nps_chat
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize


nps_chat_docs = nps_chat.xml_posts()[:1000]

#clean

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

cleaned_docs = []
for doc in nps_chat_docs:
    doc_text = doc.text  # extract text from Element
    lowered = doc_text.lower()
    #punct_removed = lowered.translate(str.maketrans('','',punctuation))
    stops_removed = [w for w in lowered.split() if w not in stop_words]
    lemmatized = [lemmatizer.lemmatize(w) for w in stops_removed]
    cleaned_docs.append(lemmatized)

#summarize

summaries = []
for doc in cleaned_docs:
    sentences = sent_tokenize(' '.join(doc))
    summary = ' '.join(sentences[:3])
    summaries.append(summary)

#print number of words and unique words in docs

for summary in summaries:
    words = summary.split()
    word_count = len(words)
    unique_words = len(set(words))

    print("Summary:")
    print(summary)
    print()

    print("Number of words:", word_count)
    print("Number of unique words:", unique_words)

#numeric representation

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
vectorized = vectorizer.fit_transform(summaries)

print(vectorized)