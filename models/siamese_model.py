import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Input, Lambda
from tensorflow.keras import backend as K
from utils import ManDist

"""
Siamese LSTM Based
"""
class SiameseLSTM:
    def __init__(self, embeddings, embedding_dim=300, max_seq_length=20):
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length

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

        left_embedding = shared_embedding(left_input)
        right_embedding = shared_embedding(right_input)

        left_output = shared_lstm(left_embedding)
        right_output = shared_lstm(right_embedding)

        
        malstm_distance = ManDist()([left_output, right_output])


        model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
        return model
