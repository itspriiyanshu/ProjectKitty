from tensorflow.keras.layers import Layer, Softmax, Multiply
import tensorflow.keras.backend as K
import tensorflow as tf

class Attention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1), initializer="random_normal", name="att_weight")
        self.b = self.add_weight(shape=(input_shape[1], 1), initializer="zeros", name="att_bias")
        super().build(input_shape)

    def call(self, inputs):
        e = tf.tanh(K.dot(inputs, self.W) + self.b)  # (batch_size, timesteps, 1)
        alpha = Softmax(axis=1)(e)                   # attention weights over timesteps
        context = Multiply()([inputs, alpha])
        context_sum = K.sum(context, axis=1)         # weighted sum over timesteps
        return context_sum

    def compute_mask(self, inputs, mask=None):
        # No masking needed
        return None

    def get_config(self):
        base_config = super().get_config()
        return base_config
