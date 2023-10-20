#sources: DATS NLP lecture code: lessons 2, 4, and 5. Code proofed/debugged in chatgpt.

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

#read in data

data_train = pd.read_csv('Train.csv')
print(data_train)

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
data_train['Text'] = data_train['Text'].apply(preprocess)

#encode labels
label_encoder = LabelEncoder()
data_train['Target'] = label_encoder.fit_transform(data_train['Target'])

print(data_train)

#test/train split
X = data_train['Text']
y = data_train['Target']
X_train, X_test, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#vectorize to convert each response into meaningful numbers
vectorizer = TfidfVectorizer()  # You can use other vectorizers as well
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

#apply naive bayes classifier
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_vectorized, y_train)

#evaluate model on test
y_pred = naive_bayes.predict(X_test_vectorized)
accuracy = accuracy_score(y_val, y_pred)
#classification_rep = classification_report(y_val, y_pred, target_names=label_encoder.classes_)

#print accuracy
print(f"Accuracy: {accuracy:.2f}")


#calculate f1 score
f1 = f1_score(y_val, y_pred, average='weighted')

#print f1 score
print(f"F1 Score: {f1:.2f}")


#save model to pickle
#REMEMBER - PICKLE SAVES TO AWS, NOT LOCAL. YOU HAVE TO DOWNLOAD IT.
with open('naive_bayes_model.pkl', 'wb') as model_file:
    pickle.dump(naive_bayes, model_file)

#load test
data_test = pd.read_csv('Test_submission.csv')

#preprocess test data
data_test['Text'] = data_test['Text'].apply(preprocess)

#vectorize test data
X_test_submission = vectorizer.transform(data_test['Text'])

#predict targets
y_test_pred = naive_bayes.predict(X_test_submission)

#create submission dataframe
submission_df = pd.DataFrame({'Text': data_test['Text'], 'Target': y_test_pred})

#create csv
submission_df.to_csv('Test_submission_pkelly.csv', index=False)
