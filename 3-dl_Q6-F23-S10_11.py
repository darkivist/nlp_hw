# =================================================================
# Class_Ex2:
# Download the swedish name dataset.
# classifying common swedish names into gender categories.
# Use Char level embedding.
# Then classify each name
# ----------------------------------------------------------------
print(20 * '-' + 'Begin Q2' + 20 * '-')
#sources: lecture notes; code proofed and debugged with chatgpt

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model

#load data
data = pd.read_csv('Name_Data_set.csv')

#cncoding target labels
label_encoder = LabelEncoder()
data['gender'] = label_encoder.fit_transform(data['gender'])

#test/train split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

#convert names into character sequences
def create_char_sequences(names):
    char_sequences = [[c for c in name] for name in names]
    return char_sequences

train_char_sequences = create_char_sequences(train_data['namn'])
test_char_sequences = create_char_sequences(test_data['namn'])

#get unique characters
all_chars = set([char for name in train_char_sequences for char in name])
char_to_index = {char: idx + 1 for idx, char in enumerate(sorted(all_chars))}

max_length = max(len(name) for name in train_char_sequences)  # Maximum sequence length

#pad sequences
train_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    [[char_to_index.get(char, 0) for char in name] for name in train_char_sequences],
    padding='post',
    maxlen=max_length
)

test_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    [[char_to_index.get(char, 0) for char in name] for name in test_char_sequences],
    padding='post',
    maxlen=max_length
)

#character-level LSTM model
input_seq = Input(shape=(max_length,))
embedding_layer = Embedding(input_dim=len(char_to_index) + 1, output_dim=128)(input_seq)
lstm_layer = LSTM(64)(embedding_layer)  # LSTM layer with 64 units
output = Dense(1, activation='sigmoid')(lstm_layer)

model = Model(inputs=input_seq, outputs=output)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)  # Set the learning rate here (0.001 as an example)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

#train
model.fit(train_sequences, train_data['gender'], epochs=100, batch_size=64, validation_split=0.1)

#check accuracy
predictions = model.predict(test_sequences)
predictions = np.round(predictions).astype(int)
accuracy = accuracy_score(test_data['gender'], predictions)
print(f"Accuracy: {accuracy}")

print(20 * '-' + 'End Q2' + 20 * '-')
