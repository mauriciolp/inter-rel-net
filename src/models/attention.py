import keras
import keras.backend as K
import tensorflow as tf
import tensorflow.nn as nn
import numpy as np

class IRNAttention(keras.layers.Layer):
    def __init__(self, num_head=1, **kwargs):
        self.num_head = num_head
        super(IRNAttention, self).__init__(**kwargs)

    def build(self, input_size, projection_size=None):
        if projection_size is None:
            projection_size = input_size[0][1] # size of joint object

        self.layer_1 = keras.layers.Dense(projection_size, use_bias=False, activation='tanh')
        self.layer_2 = keras.layers.Dense(self.num_head, use_bias=False, activation=None)
        super(IRNAttention, self).build(input_size)
    
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
        # (N, W, I) -> (N, W, Projection_size)
        out_1 = self.layer_1(inputs)
        # (N, W, Projection_size) -> (N, W, H)
        out_2 = self.layer_2(out_1)
        # Second dimension, W, will sum to one
        attention = tf.exp(out_2)
        # Adding a small constant to the normalizing factor in case everything
        # is zero
        # (N, W, H)/(N, 1, W, H).sum(dim=2) -> (N, W, H)
        attention = attention / (K.sum(K.expand_dims(attention, axis=1), axis=2) + 0.0000001)

        # (N, W, I) -> (N, I, W)
        inputs = tf.transpose(inputs, perm=[0,2,1])

        # Weighted average of the input vectors
        # (N, I, W)*(N, W, H) = (N, I, H)
        sentence = keras.layers.dot([inputs, attention], axes=(2,1))

        # (N, I, H) -> (N, I*H, 1)
        sentence = keras.layers.Reshape((sentence.shape[1]*sentence.shape[2],))(sentence)
        return (
            sentence
            # attention.view(N, -1, self.num_head, self.num_outputs),
        )