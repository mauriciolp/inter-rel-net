import keras
import keras.backend as K
import tensorflow as tf
import tensorflow.nn as nn
import numpy as np

class IRNAttention(keras.layers.Layer):
    def __init__(self, num_head=1, projection_size=None, return_attention=False, **kwargs):
        self.return_attention = return_attention
        self.projection_size = projection_size
        self.num_head = num_head
        super(IRNAttention, self).__init__(**kwargs)

    def build(self, input_size):
        if self.projection_size is None:
            self.projection_size = input_size[0][1] # size of joint object
        
        self.layer_1 = keras.layers.Dense(self.projection_size, name="Att1", use_bias=False, activation='tanh')
        self.layer_2 = keras.layers.Dense(self.num_head, name="Att2", use_bias=False, activation=None)
        self.attention = self.add_weight(name="attention", shape=(len(input_size), self.num_head), trainable=False, initializer='zeros')
        # self.attention = tf.Variable(initial_value = tf.zeros(len(input_size), self.num_head,), trainable=False)
        # self.attention = K.variable(tf.zeros([len(input_size), self.num_head]), name="att_weights")
        super(IRNAttention, self).build(input_size)
    
    def exp_normalize(self, x):
        # Subtracting a constant in the exponent doesn't change the probability distribution
        # of the softmax. e**-b gets canceled out in the numerator and denominator.
        b = K.max(x)
        y = tf.exp(x - b)

        # Adding a small constant to the normalizing factor in case everything
        # is zero
        return y / (K.sum(K.expand_dims(y, axis=1), axis=2) + 1.2e-38)  # 1.175494e-38  is smallest float32 value
        
    def call(self, inputs):
        """
        N: Batch Size
        W: number of joints
        I: size of representation (last layer of g)
        H: num_head

        Parameters
        ----------

        x: torch.tensor
            (N, W, I) batch_first

        Returns
        -------
        Tuple[torch.tensor, torch.tensor]

            First item is the representation aggregated from a weighted
            average of the relational reasoning module

            Second item is the attention used for the weighted average
        """
        # Convert to format (N, W, I)
        inputs = tf.transpose(tf.stack(inputs), perm=[1,0,2])
        # (N, W, I) -> (N, W, self.Projection_size)
        out_1 = self.layer_1(inputs)
        # (N, W, self.Projection_size) -> (N, W, H)
        out_2 = self.layer_2(out_1)
        # Second dimension, W, will sum to one
        
        # att_out = tf.exp(out_2)
        
        # Adding a small constant to the normalizing factor in case everything
        # is zero
        # (N, W, H)/(N, 1, W, H).sum(dim=2) -> (N, W, H)
        
        attention = self.exp_normalize(out_2)

        # (N, W, I) -> (N, I, W)
        inputs = tf.transpose(inputs, perm=[0,2,1])

        # Weighted average of the input vectors
        # (N, I, W)*(N, W, H) = (N, I, H)
        sentence = keras.layers.dot([inputs, attention], axes=(2,1))

        # (N, I, H) -> (N, I*H, 1)
        sentence = keras.layers.Reshape((sentence.shape[1]*sentence.shape[2],))(sentence)
        
        if self.return_attention:
            return [sentence, attention]

        return (
            sentence
        )

    def get_config(self):
        config = super(IRNAttention, self).get_config()
        config.update({"num_head": self.num_head,
                       "projection_size": self.projection_size,
                       "return_attention": self.return_attention})
        return config