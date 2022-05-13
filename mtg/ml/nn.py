import tensorflow as tf
from mtg.ml.layers import Dense, LayerNormalization, MultiHeadAttention


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


class ConcatEmbedding(tf.Module):
    """
    Lets say you want an embedding that is a concatenation of the abstract object and data about the object

    so we learn a normal one hot embedding, and then have an MLP process the data about the object and concatenate the two.
    """

    def __init__(
        self,
        num_items,
        emb_dim,
        item_data,
        dropout=0.0,
        n_h_layers=1,
        initializer=tf.initializers.GlorotNormal(),
        name=None,
        activation=None,
        start_act=None,
        middle_act=None,
        out_act=None,
    ):
        super().__init__(name=name)
        assert item_data.shape[0] == num_items
        self.item_data = item_data
        self.item_MLP = MLP(
            in_dim=item_data.shape[1],
            start_dim=item_data.shape[1] // 2,
            out_dim=emb_dim // 2,
            n_h_layers=n_h_layers,
            dropout=dropout,
            name="item_data_mlp",
            start_act=start_act,
            middle_act=middle_act,
            out_act=out_act,
            style="bottleneck",
        )
        self.embedding = tf.Variable(
            initializer(shape=(num_items, emb_dim // 2)),
            dtype=tf.float32,
            name=self.name + "_embedding",
        )
        self.activation = activation

    # @tf.function
    def __call__(self, x, training=None):
        item_embeddings = tf.gather(self.embedding, x)
        data_embeddings = tf.gather(
            self.item_MLP(self.item_data, training=training),
            x,
        )
        embeddings = tf.concat([item_embeddings, data_embeddings], axis=-1)
        if self.activation is not None:
            embeddings = self.activation(embeddings)
        return embeddings


class TransformerBlock(tf.Module):
    """
    Transformer Block implementation. Rather than having a separate class for the encoder
        block and decoder block, instead there is a `decode` flag to determine if the extra
        processing step is necessary
    """

    def __init__(
        self,
        emb_dim,
        num_heads,
        pointwise_ffn_width,
        dropout=0.0,
        decode=False,
        name=None,
    ):
        super().__init__(name=name)
        self.dropout = dropout
        self.decode = decode
        # kdim and dmodel are the same because the embedding dimension of the non-attended
        # embeddings are the same as the attention embeddings.
        self.attention = MultiHeadAttention(
            emb_dim, emb_dim, num_heads, name=self.name + "_attention"
        )
        self.expand_attention = Dense(
            emb_dim,
            pointwise_ffn_width,
            activation=tf.nn.relu,
            name=self.name + "_pointwise_in",
        )
        self.compress_expansion = Dense(
            pointwise_ffn_width,
            emb_dim,
            activation=None,
            name=self.name + "_pointwise_out",
        )
        self.final_layer_norm = LayerNormalization(
            emb_dim, name=self.name + "_out_norm"
        )
        self.attention_layer_norm = LayerNormalization(
            emb_dim, name=self.name + "_attention_norm"
        )
        if self.decode:
            self.decode_attention = MultiHeadAttention(
                emb_dim, emb_dim, num_heads, name=self.name + "_decode_attention"
            )
            self.decode_layer_norm = LayerNormalization(
                emb_dim, name=self.name + "_decode_norm"
            )

    def pointwise_fnn(self, x, training=None):
        x = self.expand_attention(x, training=training)
        return self.compress_expansion(x, training=training)

    # @tf.function
    def __call__(self, x, mask, encoder_output=None, training=None):
        attention_emb, attention_weights = self.attention(
            x, x, x, mask, training=training
        )
        if training and self.dropout > 0:
            attention_emb = tf.nn.dropout(attention_emb, rate=self.dropout)
        residual_emb_w_memory = self.attention_layer_norm(
            x + attention_emb, training=training
        )
        if self.decode:
            assert encoder_output is not None
            decode_attention_emb, decode_attention_weights = self.decode_attention(
                encoder_output,
                encoder_output,
                residual_emb_w_memory,
                mask,
                training=training,
            )
            if training and self.dropout > 0:
                decode_attention_emb = tf.nn.dropout(
                    decode_attention_emb, rate=self.dropout
                )
            residual_emb_w_memory = self.decode_layer_norm(
                residual_emb_w_memory + decode_attention_emb, training=training
            )
            attention_weights = (attention_weights, decode_attention_weights)
        process_emb = self.pointwise_fnn(residual_emb_w_memory, training=training)
        if training and self.dropout > 0:
            process_emb = tf.nn.dropout(process_emb, rate=self.dropout)
        return (
            self.final_layer_norm(
                residual_emb_w_memory + process_emb, training=training
            ),
            attention_weights,
        )
