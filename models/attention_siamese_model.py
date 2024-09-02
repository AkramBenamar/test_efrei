import tensorflow as tf
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Embedding, LSTM, Input, Lambda, Layer, Dense,Bidirectional
from tensorflow.keras import backend as K

import tensorflow as tf
import numpy as np
from utils import ManDist

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
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
                left_input = Input(shape=(self.max_seq_length,), dtype='int32')
                right_input = Input(shape=(self.max_seq_length,), dtype='int32')

                # Shared LSTM model with self-attention
                shared_model = Sequential()
                shared_model.add(Embedding(input_dim=len(self.embeddings), output_dim=self.embedding_dim,
                                        weights=[self.embeddings], input_length=self.max_seq_length, trainable=False))
                shared_model.add(Bidirectional(LSTM(50, return_sequences=True)))
                shared_model.add(SelfAttention(self.attention_units))
                shared_model.add(tf.keras.layers.GlobalAveragePooling1D())

                # Compute the Manhattan distance between the two LSTM outputs
                malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])

                model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])
                model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

                return model
                    
