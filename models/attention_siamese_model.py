import tensorflow as tf
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Embedding, LSTM, Input, Lambda, Layer, Dense
from tensorflow.keras import backend as K

import tensorflow as tf
import numpy as np


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

    def call(self, inputs):

        query = tf.tensordot(inputs, self.W_q, axes=1)
        key = tf.tensordot(inputs, self.W_k, axes=1)
        value = tf.tensordot(inputs, self.W_v, axes=1)


        scores = tf.matmul(query, key, transpose_b=True)
        scores /= tf.math.sqrt(tf.cast(tf.shape(key)[-1], tf.float32))


        attention_weights = tf.nn.softmax(scores, axis=-1)


        context_vector = tf.matmul(attention_weights, value)

        return context_vector
class AttenSiameseLSTM:
    def __init__(self, embeddings, embedding_dim=300, max_seq_length=20,attention_units=64):
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.attention_units=attention_units

    def build_model(self):
        # Define the model
        left_input = Input(shape=(self.max_seq_length,), dtype='int32')
        right_input = Input(shape=(self.max_seq_length,), dtype='int32')

        shared_embedding = Embedding(input_dim=self.embeddings.shape[0], 
                                     output_dim=self.embedding_dim,
                                     weights=[self.embeddings], 
                                     input_length=self.max_seq_length, 
                                     trainable=False)
        
        shared_lstm = LSTM(50)
        shared_attention=SelfAttention(self.attention_units)
        left_embedding = shared_embedding(left_input)
        right_embedding = shared_embedding(right_input)
        left_lstm = shared_lstm(left_embedding)
        right_lstm = shared_lstm(right_embedding)
        left_output = shared_attention(left_lstm)
        right_output = shared_attention(right_lstm)

        # Manhattan Distance Layer
        def manhattan_distance(x):
            return K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True)

        distance = Lambda(manhattan_distance)([left_output, right_output])

        model = Model(inputs=[left_input, right_input], outputs=distance)
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        return model
