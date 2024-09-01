from tensorflow.keras.layers import Layer
import tensorflow as tf

class ManDist(Layer):
    def __init__(self, **kwargs):
        super(ManDist, self).__init__(**kwargs)

    def call(self, inputs):
        left, right = inputs
        return tf.exp(-tf.reduce_sum(tf.abs(left - right), axis=1, keepdims=True))