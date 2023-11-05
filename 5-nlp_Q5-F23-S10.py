#sources: Word2vec.py from DATS 6312 lecture code:
#https://github.com/amir-jafari/NLP/blob/master/Lecture_07/Lecture%20Code/Word2vec.py
#code proofed/debugged in chatgpt

# =================================================================
# Class_Ex7:
#
# The objective of this exercise to show the inner workings of Word2Vec in python using numpy.
# Do not be using any other libraries for that.
# We are not looking at efficient implementation, the purpose here is to understand the mechanism
# behind it. You can find the official paper here. https://arxiv.org/pdf/1301.3781.pdf
# The main component of your code should be the followings:
# Set your hyper-parameters
# Data Preparation (Read text file)
# Generate training data (indexing to an integer and the onehot encoding )
# Forward and backward steps of the autoencoder network
# Calculate the error
# look at error at by varying hidden dimensions and window size
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q7' + 20 * '-')

import numpy as np

#define hyperparameters
LR = 1e-2
N_EPOCHS = 2000
HIDDEN_DIM = 2
WINDOW_SIZE = 2
PRINT_LOSS_EVERY = 100

#read in text
corpus = ['king is a strong man',
          'queen is a wise woman',
          'boy is a young man',
          'girl is a young woman',
          'prince is a young king',
          'princess is a young queen',
          'man is strong',
          'woman is pretty',
          'prince is a boy will be king',
          'princess is a girl will be queen']

#helper function to remove stop words
def remove_stop_words(corpus):
    stop_words = ['is', 'a', 'will', 'be']
    results = []
    for text in corpus:
        tmp = text.split(' ')
        for stop_word in stop_words:
            if stop_word in tmp:
                tmp.remove(stop_word)
        results.append(" ".join(tmp))
    return results

corpus = remove_stop_words(corpus)

#tokenize the corpus
words = []
for text in corpus:
    for word in text.split(' '):
        words.append(word)

words = list(set(words))
word2int = {word: i for i, word in enumerate(words)}

#generate training data
data = []
for sentence in corpus:
    sentence = sentence.split()
    for idx, word in enumerate(sentence):
        for neighbor in sentence[max(idx - WINDOW_SIZE, 0): min(idx + WINDOW_SIZE, len(sentence)) + 1]:
            if neighbor != word:
                data.append([word, neighbor])

#create one-hot encoding for words
ONE_HOT_DIM = len(words)

def to_one_hot_encoding(data_point_index):
    one_hot_encoding = np.zeros(ONE_HOT_DIM)
    one_hot_encoding[data_point_index] = 1
    return one_hot_encoding

X = []
Y = []
for x, y in data:
    X.append(to_one_hot_encoding(word2int[x]))
    Y.append(to_one_hot_encoding(word2int[y]))

X_train = np.array(X)
Y_train = np.array(Y)

#word2vec training
np.random.seed(0)
W1 = np.random.randn(ONE_HOT_DIM, HIDDEN_DIM)
W2 = np.random.randn(HIDDEN_DIM, ONE_HOT_DIM)

#training loop
for epoch in range(N_EPOCHS):
    #forward pass
    h = X_train.dot(W1)
    u = h.dot(W2)
    y_pred = 1 / (1 + np.exp(-u))

    #calculate error
    mse = np.mean((Y_train - y_pred) ** 2)

    #backward pass
    E = Y_train - y_pred
    dW2 = h.T.dot(E)
    dW1 = X_train.T.dot(E.dot(W2.T))

    #update weights
    W1 += LR * dW1
    W2 += LR * dW2

    if epoch % PRINT_LOSS_EVERY == 0:
        print("Epoch {} | Mean Squared Error: {:.5f}".format(epoch, mse))

#extract word vectors
vectors = W1

#print word vectors
print(vectors)

print(20 * '-' + 'End Q7' + 20 * '-')

# =================================================================