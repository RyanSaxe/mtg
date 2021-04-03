import tensorflow as tf
from mtg.ml.layers import Dense, Dropout

class MLP(tf.Module):
    def __init__(
        self,
        start_dim,
        out_dim,
        n_h_layers,
        dropout=None,
        name=None,
        noisy=False,
        start_act=tf.nn.linear,
        middle_act=tf.nn.relu,
        out_act=tf.nn.relu,
        style="bottleneck"
    ):
        assert style in ['bottleneck','flat','reverse_bottleneck']
        super().__init__(name=name)
        self.layers = [Dense(start_dim,activation=start_act)]
        last_dim = start_dim
        if dropout is not None:
            self.drop = Dropout(dropout)
            if noisy:
                self.layers = [self.drop] + self.layers
        for _ in range(n_h_layers):
            if style == "bottleneck":
                dim = last_dim // 2
            elif style == "reverse_bottleneck":
                dim = last_dim * 2
            else:
                dim = last_dim
            self.layers.append(
                Dense(dim, activation=middle_act)
            )
            self.layers.append(self.drop)
            last_dim = dim
        self.layers.append(Dense(out_dim, activation=out_act))

    @tf.function
    def __call__(self, x, training=None):
        for layer in self.layers:
            x = layer(x)
        return x