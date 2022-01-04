import tensorflow as tf
from mtg.ml.layers import Dense


class MLP(tf.Module):
    def __init__(
        self,
        in_dim,
        start_dim,
        out_dim,
        n_h_layers,
        dropout=0.0,
        noise=0.0,
        start_act=tf.nn.relu,
        middle_act=tf.nn.relu,
        out_act=tf.nn.relu,
        style="bottleneck",
        name=None,
    ):
        assert style in ["bottleneck", "flat", "reverse_bottleneck"]
        super().__init__(name=name)
        self.noise = noise
        self.dropout = dropout
        self.layers = [
            Dense(in_dim, start_dim, activation=start_act, name=self.name + "_0")
        ]
        last_dim = start_dim
        for i in range(n_h_layers):
            if style == "bottleneck":
                dim = last_dim // 2
            elif style == "reverse_bottleneck":
                dim = last_dim * 2
            else:
                dim = last_dim
            self.layers.append(
                Dense(
                    last_dim,
                    dim,
                    activation=middle_act,
                    name=self.name + "_" + str(i + 1),
                )
            )
            last_dim = dim
        self.layers.append(
            Dense(last_dim, out_dim, activation=out_act, name=self.name + "_out")
        )

    # @tf.function
    def __call__(self, x, training=None):
        if self.noise > 0.0 and training:
            x = tf.nn.dropout(x, rate=self.noise)
        for layer in self.layers:
            x = layer(x)
            if self.dropout > 0.0 and training:
                x = tf.nn.dropout(x, rate=self.dropout)
        return x
