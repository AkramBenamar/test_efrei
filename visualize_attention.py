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

nltk.download('stopwords')
class ManDist(Layer):
    def __init__(self, **kwargs):
        super(ManDist, self).__init__(**kwargs)

    def call(self, inputs):
        left, right = inputs
        return tf.exp(-tf.reduce_sum(tf.abs(left - right), axis=1, keepdims=True))
class DataPreprocessor:
    def __init__(self, embedding_path, sample_size=10000):
        self.embedding_path = embedding_path
        self.sample_size = sample_size
        self.stop_words = set(stopwords.words('english'))
        self.word_to_index = None

    def text_to_word_list(self, text):
        # Pre-process and convert texts to a list of words
        text = str(text)
        text = text.lower()

        # Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)

        text = text.split()
        return text

    def make_w2v_embeddings(self, df, embedding_dim=300, empty_w2v=False):
        vocabs = {}
        vocabs_cnt = 0

        vocabs_not_w2v = {}
        vocabs_not_w2v_cnt = 0

        # Stopwords
        stops = set(stopwords.words('english'))

        # Load word2vec
        print("Loading word2vec model (this may take 2-3 mins) ...")
        word2vec = KeyedVectors.load_word2vec_format(self.embedding_path, binary=True)

        for index, row in df.iterrows():
            # Print the number of embedded sentences
            if index != 0 and index % 1000 == 0:
                print("{:,} sentences embedded.".format(index), flush=True)

            # Iterate through the text of both questions of the row
            for question in ['question1', 'question2']:
                q2n = []  # q2n -> question numbers representation
                for word in self.text_to_word_list(row[question]):
                    # Check for unwanted words
                    if word in stops:
                        continue

                    # If a word is missing from word2vec model
                    if word not in word2vec.key_to_index:
                        if word not in vocabs_not_w2v:
                            vocabs_not_w2v_cnt += 1
                            vocabs_not_w2v[word] = 1

                    # If you have never seen a word, append it to vocab dictionary
                    if word not in vocabs:
                        vocabs_cnt += 1
                        vocabs[word] = vocabs_cnt
                        q2n.append(vocabs_cnt)
                    else:
                        q2n.append(vocabs[word])

                # Append question as number representation
                df.at[index, question + '_n'] = q2n

        embeddings = 1 * np.random.randn(len(vocabs) + 1, embedding_dim)  
        embeddings[0] = 0  

        # Build the embedding matrix
        for word, index in vocabs.items():
            if word in word2vec.key_to_index:
                embeddings[index] = word2vec.get_vector(word)
        del word2vec

        return df, embeddings

    def split_and_zero_padding(self, df, max_seq_length):
        # Split to dicts
        X = {'left': df['question1_n'], 'right': df['question2_n']}

        # Zero padding
        for dataset, side in itertools.product([X], ['left', 'right']):
            dataset[side] = pad_sequences(dataset[side], padding='pre', truncating='post', maxlen=max_seq_length)

        return X

    def load_and_process_data(self, train_csv, max_seq_length=20):
        # Load training set
        df = pd.read_csv(train_csv)
        # df=df.head(1000)
        for q in ['question1', 'question2']:
            df[q + '_n'] = df[q]

        # Make word2vec embeddings
        embedding_dim = 300
        use_w2v = True

        df, embeddings = self.make_w2v_embeddings(df, embedding_dim=embedding_dim, empty_w2v=not use_w2v)

        # Split to train-validation-test
        train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=0.1 / 0.8, random_state=42)

        # Split and pad sequences
        X_train = self.split_and_zero_padding(train_df[['question1_n', 'question2_n']], max_seq_length)
        Y_train = train_df['is_duplicate'].values

        X_val = self.split_and_zero_padding(val_df[['question1_n', 'question2_n']], max_seq_length)
        Y_val = val_df['is_duplicate'].values

        X_test = self.split_and_zero_padding(test_df[['question1_n', 'question2_n']], max_seq_length)
        Y_test = test_df['is_duplicate'].values

        # Verify alignment
        assert X_train['left'].shape[0] == len(Y_train)
        assert X_train['right'].shape[0] == len(Y_train)
        assert X_val['left'].shape[0] == len(Y_val)
        assert X_val['right'].shape[0] == len(Y_val)
        assert X_test['left'].shape[0] == len(Y_test)
        assert X_test['right'].shape[0] == len(Y_test)

        return X_train, X_val, Y_train, Y_val, X_test, Y_test, embeddings

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
                    batch_size=64, epochs=5,
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
    plt.savefig('hml.png')

    # Heatmap pour la séquence de droite
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights_right[0], cmap='viridis')
    plt.title("Attention Heatmap - Right Sequence")
    plt.xlabel("Attention Heads")
    plt.ylabel("Sequence Position")
    plt.savefig('hmr.png')

# Visualiser les heatmaps pour un échantillon de l'ensemble de test
# visualize_attention(attention_model, X_test_left, X_test_right, sample_idx=0)
visualize_attention(attention_model, X_test['left'], X_test['right'], sample_idx=0)
