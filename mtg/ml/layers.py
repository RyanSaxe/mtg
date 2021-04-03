import tensorflow as tf

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

    def __call__(self, x, training=None):
        y = tf.matmul(x, self.w) + self.b
        return self.activation(y)