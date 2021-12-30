import tensorflow as tf
from mtg.utils.display import build_decks
from tensorflow.python.keras.engine.base_layer import Layer
from mtg.ml import nn
from mtg.ml.layers import MultiHeadAttention, Dense, LayerNormalization, Embedding
import numpy as np
import pandas as pd
import pdb
import pathlib
import os
import pickle


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=1000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


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
        self.item_MLP = nn.MLP(
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

    @tf.function
    def __call__(self, x, training=None):
        # x is a card id
        item_embeddings = tf.gather(self.embedding, x)
        data_embeddings = tf.gather(
            self.item_MLP(self.item_data, training=training), x,
        )
        embeddings = tf.concat([item_embeddings, data_embeddings], axis=-1)
        if self.activation is not None:
            embeddings = self.activation(embeddings)
        return embeddings


class DraftBot(tf.Module):
    def __init__(
        self,
        cards,
        emb_dim,
        t,
        num_heads,
        num_memory_layers,
        card_data=None,
        emb_dropout=0.0,
        memory_dropout=0.0,
        out_dropout=0.0,
        use_deckbuilder=False,
        output_MLP=False,
        name=None,
    ):
        super().__init__(name=name)
        self.idx_to_name = cards.set_index("idx")["name"].to_dict()
        self.card_data = cards
        self.n_cards = len(self.idx_to_name)
        self.t = t
        self.emb_dim = tf.Variable(
            emb_dim, dtype=tf.float32, trainable=False, name="emb_dim"
        )
        self.dropout = emb_dropout
        self.positional_embedding = Embedding(t, emb_dim, name="positional_embedding")
        self.positional_mask = 1 - tf.linalg.band_part(tf.ones((t, t)), -1, 0)
        self.encoder_layers = [
            TransformerBlock(
                self.n_cards,
                emb_dim,
                num_heads,
                dropout=memory_dropout,
                name=f"memory_encoder_{i}",
            )
            for i in range(num_memory_layers)
        ]
        # extra embedding as representation of bias before the draft starts. This is grabbed as the
        # representation for the "previous pick" that goes into the decoder for P1P1
        self.card_data = card_data
        if self.card_data is None:
            self.card_embedding = Embedding(
                self.n_cards + 1, emb_dim, name="card_embedding", activation=None
            )
        else:
            self.card_data = tf.convert_to_tensor(self.card_data, dtype=tf.float32)
            self.card_embedding = ConcatEmbedding(
                self.n_cards + 1,
                emb_dim,
                self.card_data,
                name="card_embedding",
                activation=None,
            )
        self.decoder_layers = [
            TransformerBlock(
                self.n_cards,
                emb_dim,
                num_heads,
                dropout=memory_dropout,
                name=f"memory_decoder_{i}",
                decode=True,
            )
            for i in range(num_memory_layers)
        ]
        self.output_MLP = output_MLP
        if self.output_MLP:
            self.output_decoder = nn.MLP(
                in_dim=emb_dim,
                start_dim=emb_dim * 2,
                out_dim=self.n_cards,
                n_h_layers=1,
                dropout=out_dropout,
                name="output_decoder",
                start_act=tf.nn.selu,
                middle_act=tf.nn.selu,
                out_act=None,
                style="reverse_bottleneck",
            )
        else:
            self.output_decoder = Dense(
                emb_dim, emb_dim, name="output_decoder", activation=None
            )
        if use_deckbuilder:
            self.deckbuilder = DeckBuilder()
        else:
            self.deckbuilder = None
        # initializer=tf.initializers.GlorotNormal()
        # self.initial_card_bias = tf.Variable(
        #     initializer(shape=(1, emb_dim)),
        #     dtype=tf.float32,
        #     name=self.name + "_initial_card_bias",
        # )

    @tf.function
    def __call__(
        self, features, training=None, return_attention=False, return_build=True
    ):
        if self.deckbuilder is not None and return_build:
            packs, picks, positions, final_pools = features
        else:
            packs, picks, positions = features
        # pools = draft_info[:, :, self.n_cards:]
        # draft_info is of shape (batch_size, t, n_cards * 2)
        positional_masks = tf.gather(self.positional_mask, positions)
        positional_embeddings = self.positional_embedding(positions, training=training)
        # old way: pack embedding = mean of card embeddings for only cards in the pack
        # batch_size x t x n_cards x emb_dim
        all_card_embeddings = self.card_embedding(
            tf.range(self.n_cards), training=training
        )
        if self.deckbuilder is not None and return_build:
            final_pool_embeddings = (
                final_pools[:, :, None] * all_card_embeddings[None, :, :]
            )
            built_decks = self.deckbuilder(final_pool_embeddings, training=training)
        pack_card_embeddings = (
            packs[:, :, :, None] * all_card_embeddings[None, None, :, :]
        )
        n_options = tf.reduce_sum(packs, axis=-1, keepdims=True)
        pack_embeddings = tf.reduce_sum(pack_card_embeddings, axis=2) / n_options
        embs = pack_embeddings * tf.math.sqrt(self.emb_dim) + positional_embeddings
        # insert an embedding to represent bias towards cards/archetypes/concepts you have before the draft starts
        # --> this could range from "generic pick order of all cards" to "blue is the best color", etc etc
        # batch_bias = tf.tile(tf.expand_dims(self.initial_card_bias,0), [embs.shape[0],1,1])
        # batch_mask = tf.tile(tf.expand_dims(self.positional_mask,0), [embs.shape[0],1,1])
        # embs = tf.concat([
        #     batch_bias,
        #     embs,
        # ], axis=1)
        if training and self.dropout > 0.0:
            embs = tf.nn.dropout(embs, rate=self.dropout)
        for memory_layer in self.encoder_layers:
            embs, attention_weights_pack = memory_layer(
                embs, positional_masks, training=training
            )  # (batch_size, t, emb_dim)
        dec_embs = self.card_embedding(picks, training=training)
        for memory_layer in self.decoder_layers:
            dec_embs, attention_weights_pick = memory_layer(
                dec_embs, positional_masks, encoder_output=embs, training=training
            )  # (batch_size, t, emb_dim)
        # dec_embs = dec_embs
        # get rid of output with respect to initial bias vector, as that is not part of prediction
        # embs = embs[:,1:,:]
        mask_for_softmax = 1e9 * (1 - packs)
        if self.output_MLP:
            card_rankings = (
                self.output_decoder(dec_embs, training=training) * packs
                - mask_for_softmax
            )  # (batch_size, t, n_cards)
            emb_dists = (
                tf.sqrt(
                    tf.reduce_sum(
                        tf.square(pack_card_embeddings - dec_embs[:, :, None, :]), -1
                    )
                )
                * packs
                + mask_for_softmax
            )
        else:
            approx_pick_embs = self.output_decoder(dec_embs, training=training)
            emb_dists = (
                tf.sqrt(
                    tf.reduce_sum(
                        tf.square(
                            pack_card_embeddings - approx_pick_embs[:, :, None, :]
                        ),
                        -1,
                    )
                )
                * packs
                + mask_for_softmax
            )
            card_rankings = -emb_dists
        output = tf.nn.softmax(card_rankings)
        # zero out the rankings for cards not in the pack
        # note1: this only works because no foils on arena means packs can never have 2x of a card
        #       if this changes, modify to clip packs at 1
        # note2: this zeros out the gradients for the cards not in the pack in order to not negatively
        #        affect backprop on cards that would techncally be taken if they were in the pack. However,
        #        if it turns out that there is a reason why these gradients shouldn't be zero, this
        #        multiplication could be done only during inference (when training is not True)

        # add epsilon for cards in the pack to ensure they are non-zero (handles edge cases)
        # card_rankings = card_rankings * packs + 1e-9 * packs
        # after zeroing out cards not in packs, we readjust the output to maintain that it sums to one
        # note: currently this sums to one so we do from_logits=True in Categorical Cross Entropy,
        #       possible softmax is better than relu, regardless this does have numerical instability issues
        #       so that is something to look out for. But from_logits=False had terrible performance
        if self.deckbuilder is not None and return_build:
            output = (output, built_decks)
        if return_attention:
            return output, (attention_weights_pack, attention_weights_pick)
        return output, emb_dists

    def compile(
        self,
        optimizer=None,
        learning_rate=0.001,
        margin=0.1,
        emb_lambda=1.0,
        pred_lambda=1.0,
        bad_behavior_lambda=1.0,
        rare_lambda=1.0,
        cmc_lambda=1.0,
        cmc_margin=1.0,
        card_data=None,
        metric_names=["top1", "top2", "top3"],
    ):
        if optimizer is None:
            if isinstance(learning_rate, dict):
                learning_rate = CustomSchedule(self.emb_dim, **learning_rate)
            else:
                learning_rate = learning_rate

            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
            )
        else:
            self.optimizer = optimizer
        self.loss_f = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM
        )
        self.margin = margin
        self.emb_lambda = emb_lambda
        self.pred_lambda = pred_lambda
        self.bad_behavior_lambda = bad_behavior_lambda
        self.rare_lambda = rare_lambda
        self.cmc_lambda = cmc_lambda
        self.cmc_margin = cmc_margin
        if card_data is not None:
            self.set_card_params(card_data)
        self.metric_names = metric_names

    def set_card_params(self, card_data):
        self.rare_flag = (card_data["mythic"] + card_data["rare"]).values[None, None, :]
        self.cmc = card_data["cmc"].values[None, None, :]

    def loss(self, true, pred, sample_weight=None, training=None, **kwargs):
        pred, emb_dists = pred
        # if isinstance(pred, tuple):
        #     pred, built_decks_pred = pred
        #     true, built_decks_true = true
        # else:
        #     self.deck_loss = 0
        self.prediction_loss = self.loss_f(true, pred, sample_weight=sample_weight)

        correct_one_hot = tf.one_hot(true, self.n_cards)
        dist_of_not_correct = emb_dists * (1 - correct_one_hot)
        dist_of_correct = tf.reduce_sum(
            emb_dists * correct_one_hot, axis=-1, keepdims=True
        )
        dist_loss = dist_of_correct - dist_of_not_correct
        sample_weight = 1 if sample_weight is None else sample_weight
        self.embedding_loss = tf.reduce_sum(
            tf.reduce_sum(tf.maximum(dist_loss + self.margin, 0.0), axis=-1)
            * sample_weight
        )
        self.bad_behavior_loss = self.determine_bad_behavior(
            true, pred, sample_weight=sample_weight
        )

        return (
            self.pred_lambda * self.prediction_loss
            + self.emb_lambda * self.embedding_loss
            + self.bad_behavior_lambda * self.bad_behavior_loss
        )

    def determine_bad_behavior(self, true, pred, sample_weight=None):
        true_one_hot = tf.one_hot(true, self.n_cards)
        # penalize for taking more expensive cards than what the human took
        #    basically, if you're going to make a mistake, bias to low cmc cards
        true_cmc = tf.reduce_sum(true_one_hot * self.cmc, axis=-1)
        pred_cmc = tf.reduce_sum(pred * self.cmc, axis=-1)
        cmc_loss = (
            tf.maximum(pred_cmc - true_cmc + self.cmc_margin, 0.0) * self.cmc_lambda
        )
        self.cmc_loss = tf.reduce_sum(cmc_loss * sample_weight)
        # penalize taking rares when the human doesn't. This helps not learn "take rares" to
        # explain raredrafting.
        human_took_rare = tf.reduce_sum(true_one_hot * self.rare_flag, axis=-1)
        pred_rare_val = tf.reduce_sum(pred * self.rare_flag, axis=-1)
        rare_loss = (1 - human_took_rare) * pred_rare_val * self.rare_lambda
        self.rare_loss = tf.reduce_sum(rare_loss * sample_weight)
        return self.cmc_loss + self.rare_loss

    def compute_metrics(self, true, pred, sample_weight=None, **kwargs):
        if sample_weight is None:
            sample_weight = tf.ones_like(true.shape) / (true.shape[0] * true.shape[1])
        sample_weight = sample_weight.flatten()
        pred, _ = pred
        # if isinstance(pred, tuple):
        #     pred, built_decks = pred
        top1 = tf.reduce_sum(
            tf.keras.metrics.sparse_top_k_categorical_accuracy(true, pred, 1)
            * sample_weight
        )
        top2 = tf.reduce_sum(
            tf.keras.metrics.sparse_top_k_categorical_accuracy(true, pred, 2)
            * sample_weight
        )
        top3 = tf.reduce_sum(
            tf.keras.metrics.sparse_top_k_categorical_accuracy(true, pred, 3)
            * sample_weight
        )
        return {"top1": top1, "top2": top2, "top3": top3}

    def save(self, location):
        pathlib.Path(location).mkdir(parents=True, exist_ok=True)
        model_loc = os.path.join(location, "model")
        tf.saved_model.save(self, model_loc)
        data_loc = os.path.join(location, "attrs.pkl")
        with open(data_loc, "wb") as f:
            attrs = {
                "t": self.t,
                "idx_to_name": self.idx_to_name,
                "n_cards": self.n_cards,
                "embeddings": self.card_embedding(
                    tf.range(self.n_cards), training=False
                ),
            }
            pickle.dump(attrs, f)


class TransformerBlock(tf.Module):
    """
    self attention block for encorporating memory into the draft bot
    """

    def __init__(
        self, n_cards, emb_dim, num_heads, dropout=0.0, decode=False, name=None
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
            emb_dim * 4,
            activation=tf.nn.relu,
            name=self.name + "_pointwise_in",
        )
        self.compress_expansion = Dense(
            emb_dim * 4, emb_dim, activation=None, name=self.name + "_pointwise_out"
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

    @tf.function
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


class DeckBuilder(tf.Module):
    def __init__(
        self,
        n_cards,
        dropout=0.0,
        latent_dim=32,
        embeddings=None,
        embedding_agg="mean",
        name=None,
    ):
        super().__init__(name=name)
        self.n_cards = n_cards
        self.card_embeddings = embeddings
        if self.card_embeddings is not None:
            # if embeddings is an integer, learn embeddings of that dimension,
            # if embeddings is None, don't use embeddings
            # otherwise, assume embeddings are pretrained and use them
            if isinstance(embeddings, int):
                emb_trainable = True
                initializer = tf.initializers.glorot_normal()
                emb_init = initializer(shape=(self.n_cards, embeddings))
            else:
                emb_trainable = False
                emb_init = embeddings
            self.card_embeddings = tf.Variable(emb_init, trainable=emb_trainable)
            encoder_in_dim = self.card_embeddings.shape[1]
        else:
            encoder_in_dim = self.n_cards
        if self.card_embeddings is None:
            self.deck_encoder = nn.MLP(
                in_dim=encoder_in_dim,
                start_dim=encoder_in_dim,
                out_dim=latent_dim,
                n_h_layers=2,
                dropout=dropout,
                name="deck_encoder",
                noise=0.0,
                start_act=None,
                middle_act=None,
                out_act=None,
                style="bottleneck",
            )
            self.pool_encoder = nn.MLP(
                in_dim=encoder_in_dim,
                start_dim=encoder_in_dim,
                out_dim=latent_dim,
                n_h_layers=2,
                dropout=dropout,
                name="pool_encoder",
                noise=0.0,
                start_act=tf.nn.selu,
                middle_act=tf.nn.selu,
                out_act=None,
                style="bottleneck",
            )
            concat_dim = latent_dim * 2
        else:
            concat_dim = self.card_embeddings.shape[1] * 2

        self.decoder = nn.MLP(
            in_dim=latent_dim,
            start_dim=latent_dim * 2,
            out_dim=self.n_cards,
            n_h_layers=2,
            dropout=0.0,
            name="decoder",
            noise=0.0,
            start_act=tf.nn.selu,
            middle_act=tf.nn.selu,
            out_act=tf.nn.sigmoid,
            style="reverse_bottleneck",
        )
        # self.interactions = nn.Dense(self.n_cards, self.n_cards, activation=None)
        self.add_basics_to_deck = nn.Dense(
            latent_dim, 5, activation=tf.nn.softmax, name="add_basics_to_deck"
        )
        # only allow more than 18 lands when there are multiple nonbasic lands
        self.determine_n_non_basics = nn.Dense(
            latent_dim,
            1,
            activation=lambda x: tf.nn.relu(x) * +22.0,
            name="determine_n_non_basics",
        )
        self.merge_deck_and_pool = nn.Dense(
            concat_dim, latent_dim, activation=None, name="merge_deck_and_pool"
        )
        self.dropout = dropout

    # @tf.function
    # def __call__(self, pools, training=None):
    #     if self.card_embeddings is not None:
    #         card_embs = pools[:, :, None] * self.card_embeddings[None, :, :]
    #         card_interactions = self.interactions_for_weighted_avg(pools)
    #         masked_interactions = card_interactions - (
    #             1e9 * (1 - tf.clip_by_value(pools, 0, 1))
    #         )
    #         pool_card_weights = tf.nn.softmax(masked_interactions)[:, :, None]
    #         pool_embs = tf.reduce_sum(card_embs * pool_card_weights, axis=1)
    #         self.latent_rep = self.embedding_compressor_pool(
    #             pool_embs, training=training
    #         )
    #     else:
    #         self.latent_rep = self.pool_encoder(pools, training=training)
    #     reconstruction = self.decoder(self.latent_rep, training=training) * pools
    #     n_lands = self.determine_n_lands(self.latent_rep, training=training)
    #     n_basics = n_lands - tf.reduce_sum(
    #         reconstruction * self.land_mtx[None, 5:], axis=-1, keepdims=True
    #     )
    #     basics_to_add = (
    #         self.add_basics_to_deck(self.latent_rep, training=training) * n_basics
    #     )
    #     return basics_to_add, reconstruction, n_basics

    @tf.function
    def __call__(self, features, training=None):
        # batch x sample x n_cards
        pools, decks = features
        # store full pools to access in metrics

        if self.card_embeddings is not None:
            self.latent_rep_pool = tf.reduce_sum(
                pools[:, :, :, None] * self.card_embeddings[None, None, :, :], axis=2
            )
            self.latent_rep_deck = tf.reduce_sum(
                decks[:, :, :, None] * self.card_embeddings[None, None, :, :], axis=2
            )
        else:
            self.latent_rep_pool = self.pool_encoder(pools, training=training)
            self.latent_rep_deck = self.deck_encoder(decks, training=training)
        concat_emb = tf.concat([self.latent_rep_deck, self.latent_rep_pool], axis=-1)
        if self.dropout > 0.0 and training:
            concat_emb = tf.nn.dropout(concat_emb, self.dropout)
        self.latent_rep = self.merge_deck_and_pool(concat_emb, training=training)
        cards_to_add = self.decoder(self.latent_rep, training=training) * pools
        # fully_built_deck = cards_to_add + decks
        n_non_basics = self.determine_n_non_basics(self.latent_rep, training=training)
        n_basics = 40 - n_non_basics
        # n_basics = n_lands - tf.reduce_sum(
        #     fully_built_deck * self.land_mtx[None, 5:], axis=-1, keepdims=True
        # )
        basics_to_add = (
            self.add_basics_to_deck(self.latent_rep, training=training) * n_basics
        )
        return basics_to_add, cards_to_add, n_non_basics

    def compile(
        self,
        card_data=None,
        cards=None,
        learning_rate=0.001,
        basic_lambda=1.0,
        built_lambda=1.0,
        card_count_lambda=1.0,
        cmc_lambda=0.01,
        # interaction_lambda=0.01,
        optimizer=None,
        metric_names=["basics_off", "spells_off"],
    ):
        if optimizer is None:
            if isinstance(learning_rate, dict):
                learning_rate = CustomSchedule(500, **learning_rate)
            else:
                learning_rate = learning_rate

            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
            )
        else:
            self.optimizer = optimizer

        self.basic_lambda = basic_lambda
        self.built_lambda = built_lambda
        self.card_count_lambda = card_count_lambda

        # self.built_loss_f = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
        # self.basic_loss_f = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)

        self.cmc_lambda = cmc_lambda
        # self.interaction_lambda = interaction_lambda
        self.cards = cards
        if card_data is not None:
            self.set_card_params(card_data)
        self.metric_names = metric_names

    def set_card_params(self, card_data):
        self.cmc_map = card_data["cmc"].to_numpy(dtype=np.float32)
        colors = list("wubrg")
        self.produces_mtx = card_data[["produces " + c for c in colors]].to_numpy(
            dtype=np.float32
        )
        self.pip_mtx = card_data[[c + " pips" for c in colors]].to_numpy(
            dtype=np.float32
        )
        self.land_mtx = card_data["land"].to_numpy(dtype=np.float32)

    def loss(self, true, pred, sample_weight=None, **kwargs):
        true_basics, true_built = true
        pred_basics, pred_built, n_basics = pred
        # self.basic_loss = self.basic_loss_f(true_basics, pred_basics, sample_weight=sample_weight)
        # self.built_loss = self.built_loss_f(true_built, pred_built, sample_weight=sample_weight)
        # n_spells = tf.reduce_sum(pred_built, axis=-1)
        # self.card_count_loss = tf.reduce_sum(tf.math.square(40 - (n_spells + n_basics)) * sample_weight)
        self.basic_loss = tf.reduce_sum(
            tf.reduce_sum(tf.math.square(pred_basics - true_basics), axis=-1)
            * sample_weight
        )
        self.built_loss = tf.reduce_sum(
            tf.reduce_sum(tf.math.square(pred_built - true_built), axis=-1)
            * sample_weight
        )
        if self.cmc_lambda > 0:
            self.pred_curve_average = tf.reduce_mean(
                tf.multiply(pred_built, tf.expand_dims(self.cmc_map[5:], 0)), axis=-1
            )
            self.true_curve_average = tf.reduce_mean(
                tf.multiply(true_built, tf.expand_dims(self.cmc_map[5:], 0)), axis=-1
            )
            self.curve_incentive = tf.reduce_sum(
                abs(self.pred_curve_average - self.true_curve_average) * sample_weight
            )
        else:
            self.curve_incentive = 0.0
        # if self.interaction_lambda > 0:
        #     #push card level interactions in pool to zero
        #     self.interaction_reg = tf.norm(self.interactions.w,ord=1)
        # else:
        #     self.interaction_reg = 0.0
        return (
            self.basic_lambda * self.basic_loss
            + self.built_lambda * self.built_loss
            + self.cmc_lambda * self.curve_incentive
            # self.card_count_lambda * self.card_count_loss
            # self.interaction_lambda * self.interaction_reg
        )

    def compute_metrics(self, true, pred, sample_weight=None, training=None, **kwargs):
        pred_basics, pred_decks, n_basics = pred
        true_basics, true_decks = true
        if sample_weight is None:
            sample_weight = 1.0 / true_decks.shape[0]
        # if not training:
        #     # if not training, we can do numpy based argmaxes to built the deck so we can see validation
        #     # performance on how we actually build decks from the output
        #     pred_basics, pred_decks = build_decks(
        #         pred_basics.numpy(), pred_decks.numpy(), n_basics.numpy(), cards=None
        #     )
        basic_diff = tf.reduce_sum(
            tf.reduce_sum(tf.math.abs(pred_basics - true_basics), axis=-1)
            * sample_weight
        )
        deck_diff = tf.reduce_sum(
            tf.reduce_sum(tf.math.abs(pred_decks - true_decks), axis=-1) * sample_weight
        )
        return {"basics_off": basic_diff, "spells_off": deck_diff}

    def save(self, cards, location):
        pathlib.Path(location).mkdir(parents=True, exist_ok=True)
        model_loc = os.path.join(location, "model")
        data_loc = os.path.join(location, "cards.pkl")
        tf.saved_model.save(self, model_loc)
        with open(data_loc, "wb") as f:
            pickle.dump(cards, f)

