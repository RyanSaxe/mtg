import tensorflow as tf
import numpy as np

class Dense(tf.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        name = None,
        initializer = tf.initializers.GlorotNormal(),
        activation = tf.nn.relu,
    ):
        super().__init__(name=name)

        self.activation = activation

        self.w = tf.Variable(
            initializer([in_dim, out_dim]),
            dtype = tf.float32,
            name = 'w',
        )

        self.b = tf.Variable(
            tf.zeros([out_dim], name='b')
        )

    @tf.function
    def __call__(self, x, training=None):
        y = tf.matmul(x, self.w) + self.b
        return self.activation(y)

@tf.function
def sawtooth(x):
    N = 100
    oscillation = 0
    for i in range(1,N):
        sign = 1 if i % 2 == 0 else -1
        num = tf.math.sin(x * np.pi * 2 * i)/i
        oscillation += sign * num
    return x + ((1.0/np.pi) * oscillation)