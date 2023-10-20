#sources: DATS NLP lecture code: lessons 2, 4, and 5. Code proofed/debugged in chatgpt.


#**********************************





#**********************************
#==================================================================================================================================================================
# Q1:
"""
For this question you need to download a text data from the NLTK movie_reviews.
Use you knowledge that you learned in the class and clean the text appropriately.
After Cleaning is done, please find the numerical representation of text by any methods that you learned.
You need to find a creative way to label the sentiment of the sentences.
This dataset already has positive and negative labels.
Labeling sentences as 'positive' or 'negative' based on sentiment scores and named then predicted sentiments.
Create a Pandas dataframe with sentences, true sentiment labels and predicted sentiment labels.
Calculate the accuracy of your predicted sentiment and true sentiments.
"""
#==================================================================================================================================================================

print(20*'-' + 'Begin Q1' + 20*'-')

import nltk
from nltk.corpus import movie_reviews
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import pandas as pd

# Import dataset
nltk.download('movie_reviews')

# Load dataset
positive_docs = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids('pos')]
negative_docs = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids('neg')]
reviews = positive_docs + negative_docs

# Preprocess reviews text (keep the same preprocessing function)
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(lemmatized_words)

# Call preprocessing function on reviews text
reviews = [preprocess(review) for review in reviews]

# Test/train split
X_train, X_test = train_test_split(reviews, test_size=0.2, random_state=42)

# Create text classification pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# Fit the pipeline on the entire dataset
pipeline.fit(reviews, [1] * len(positive_docs) + [0] * len(negative_docs))

# ... (previous code)

# Predict sentiment labels for test data using a rule-based method (exclude "neutral")
# ... (previous code)

# Predict sentiment labels for test data using a rule-based method (exclude "neutral")
sentences_without_neutral = []
y_pred = []
y_true = []

for sentence, true_label in zip(X_test, [1] * len(X_test) + [0] * len(X_test)):
    # Define your rules and heuristics here to predict sentiment (exclude "neutral")
    if "excellent" in sentence or "amazing" in sentence or "wonderful" in sentence or "good" in sentence:
        sentiment_label = 1  # Positive
    elif "terrible" in sentence or "awful" in sentence or "horrible" in sentence or "wonderful" in sentence:
        sentiment_label = 0  # Negative
    else:
        continue  # Exclude "neutral" sentiment

    sentences_without_neutral.append(sentence)
    y_pred.append(sentiment_label)
    y_true.append(true_label)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

# Create a DataFrame to store the results
results_df = pd.DataFrame({'Sentence': sentences_without_neutral, 'Rule-Based Sentiment': y_pred, 'True Sentiment': y_true})

# Print accuracy
print(f"Accuracy: {accuracy * 100:.2f}%")

# Display the DataFrame
print("\nDataFrame with Sentences, Rule-Based Sentiment, and True Sentiment Labels:")
print(results_df)
results_df.to_csv("sentiment_results.csv", index=False)

print(20*'-' + 'End Q1' + 20*'-')
