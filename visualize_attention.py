# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Layer, Input, Embedding, LSTM
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
import itertools
import os
import nltk
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from utils import ManDist
from data.data_processing import DataPreprocessor
nltk.download('stopwords')

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, LSTM, Input, Dense, GlobalAveragePooling1D, Layer
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Classe SelfAttention
class SelfAttention(Layer):
    def __init__(self, units, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W_q = self.add_weight(shape=(input_shape[-1], self.units),
                                   initializer='random_normal',
                                   trainable=True)
        self.W_k = self.add_weight(shape=(input_shape[-1], self.units),
                                   initializer='random_normal',
                                   trainable=True)
        self.W_v = self.add_weight(shape=(input_shape[-1], self.units),
                                   initializer='random_normal',
                                   trainable=True)
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs, return_attention_weights=False):
        query = tf.tensordot(inputs, self.W_q, axes=1)
        key = tf.tensordot(inputs, self.W_k, axes=1)
        value = tf.tensordot(inputs, self.W_v, axes=1)

        scores = tf.matmul(query, key, transpose_b=True)
        scores /= tf.math.sqrt(tf.cast(tf.shape(key)[-1], tf.float32))
        attention_weights = tf.nn.softmax(scores, axis=-1)
        context_vector = tf.matmul(attention_weights, value)

        # if return_attention_weights:
        return context_vector, attention_weights
        # return context_vector

# Fonction de calcul de la distance de Manhattan
class ManDist(Layer):
    def __init__(self, **kwargs):
        super(ManDist, self).__init__(**kwargs)

    def call(self, inputs):
        left, right = inputs
        return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))

# Classe pour construire le modèle Siamese LSTM avec Self-Attention
class AttenSiameseLSTM:
    def __init__(self, embeddings, embedding_dim=300, max_seq_length=20, attention_units=64):
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.attention_units = attention_units

    def build_model(self):
        left_input = Input(shape=(self.max_seq_length,), dtype='int32')
        right_input = Input(shape=(self.max_seq_length,), dtype='int32')

        shared_model = Sequential()
        shared_model.add(Embedding(input_dim=len(self.embeddings), output_dim=self.embedding_dim,
                                   weights=[self.embeddings], input_length=self.max_seq_length, trainable=False))
        shared_model.add(LSTM(50, return_sequences=True))
        attention_layer = SelfAttention(self.attention_units)
        encoded_left, attention_weights_left = attention_layer(shared_model(left_input))
        encoded_right, attention_weights_right = attention_layer(shared_model(right_input))

        encoded_left = tf.keras.layers.GlobalAveragePooling1D()(encoded_left)
        encoded_right = tf.keras.layers.GlobalAveragePooling1D()(encoded_right)

        malstm_distance = ManDist()([encoded_left, encoded_right])

        model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])
        attention_model = Model(inputs=[left_input, right_input], outputs=[attention_weights_left, attention_weights_right])

        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

        return model, attention_model

train_csv = os.path.join('C:/Users/DELL/Desktop/Akm/test_technique_efrei/CompAIre/data', 'questions.csv')
# Initialize preprocessor
embedding_path = os.path.join('C:/Users/DELL/Desktop/Akm/test_technique_efrei/CompAIre/data', 'GoogleNews-vectors-negative300.bin.gz')

preprocessor = DataPreprocessor(embedding_path, 10000)

# Load and process data
X_train, X_validation, Y_train, Y_validation,X_test, Y_test, embeddings = preprocessor.load_and_process_data(train_csv, 20)

# Construire le modèle
siamese_model = AttenSiameseLSTM(embeddings, embedding_dim=300, max_seq_length=20)
model, attention_model = siamese_model.build_model()

# Entraîner le modèle
history = model.fit([X_train['left'], X_train['right']], Y_train,
                    batch_size=2048, epochs=50,
                    validation_data=([X_validation['left'], X_validation['right']], Y_validation))

# Prédire et visualiser les heatmaps d'attention
def visualize_attention(attention_model, X_left, X_right, sample_idx):
    attention_weights_left, attention_weights_right = attention_model.predict([X_left[sample_idx:sample_idx+1], X_right[sample_idx:sample_idx+1]])

    # Heatmap pour la séquence de gauche
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights_left[0], cmap='viridis')
    plt.title("Attention Heatmap - Left Sequence")
    plt.xlabel("Attention Heads")
    plt.ylabel("Sequence Position")
    plt.savefig('checkpoints/heatmaps/lstm/hml.png')

    # Heatmap pour la séquence de droite
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights_right[0], cmap='viridis')
    plt.title("Attention Heatmap - Right Sequence")
    plt.xlabel("Attention Heads")
    plt.ylabel("Sequence Position")
    plt.savefig('checkpoints/heatmaps/lstm/hmr.png')

# Visualiser les heatmaps pour un échantillon de l'ensemble de test
# visualize_attention(attention_model, X_test_left, X_test_right, sample_idx=0)
for sample_id in range(10):
    visualize_attention(attention_model, X_test['left'], X_test['right'], sample_id)
