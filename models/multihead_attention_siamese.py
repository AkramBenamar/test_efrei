from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
import tensorflow as tf
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Embedding, LSTM, Input, Lambda, Layer, Dense
from tensorflow.keras import backend as K

import tensorflow as tf
import numpy as np
from utils import ManDist

class MHAttenSiameseLSTM:

    def __init__(self, embeddings, embedding_dim=300, max_seq_length=20, attention_units=64, num_heads=8):
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.attention_units = attention_units
        self.num_heads = num_heads

    def build_model(self):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            left_input = Input(shape=(self.max_seq_length,), dtype='int32')
            right_input = Input(shape=(self.max_seq_length,), dtype='int32')

            # Shared LSTM model with multi-head attention
            shared_model = Sequential()
            shared_model.add(Embedding(input_dim=len(self.embeddings), output_dim=self.embedding_dim,
                                        weights=[self.embeddings], input_length=self.max_seq_length, trainable=False))
            shared_model.add(LSTM(50, return_sequences=True))
            
            # Define multi-head attention and layer normalization
            self.multi_head_attention = MultiHeadAttention(
                num_heads=self.num_heads, 
                key_dim=self.attention_units
            )
            self.layer_norm = LayerNormalization()

            def apply_attention(x):
                # Apply multi-head attention
                attn_output = self.multi_head_attention(x, x)  # self-attention
                # Apply layer normalization
                attn_output = self.layer_norm(attn_output + x)  # Add & Norm
                return attn_output

            # Use Lambda to apply attention
            shared_model.add(Lambda(lambda x: apply_attention(x)))
            shared_model.add(tf.keras.layers.GlobalAveragePooling1D())

            # Compute the Manhattan distance between the two LSTM outputs
            malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])

            model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])
            model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

            return model
