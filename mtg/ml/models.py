import tensorflow as tf
from mtg.ml import nn
from mtg.ml.layers import Embedding
import numpy as np
import pathlib
import os
import pickle
from mtg.ml.utils import CustomSchedule


class DraftBot(tf.Module):
    def __init__(
        self,
        expansion,
        emb_dim,
        num_encoder_heads,
        num_decoder_heads,
        num_encoder_layers,
        num_decoder_layers,
        pointwise_ffn_width,
        emb_dropout=0.0,
        memory_dropout=0.0,
        out_dropout=0.0,
        name=None,
    ):
        super().__init__(name=name)
        self.idx_to_name = expansion.get_mapping("idx", "name", include_basics=False)
        self.n_cards = len(self.idx_to_name)
        self.t = expansion.t
        self.card_data = expansion.card_data_for_ML[5:]
        self.emb_dim = tf.Variable(
            emb_dim, dtype=tf.float32, trainable=False, name="emb_dim"
        )
        self.dropout = emb_dropout
        self.positional_embedding = Embedding(
            self.t, emb_dim, name="positional_embedding"
        )
        self.positional_mask = 1 - tf.linalg.band_part(tf.ones((self.t, self.t)), -1, 0)
        self.encoder_layers = [
            nn.TransformerBlock(
                emb_dim,
                num_encoder_heads,
                pointwise_ffn_width,
                dropout=memory_dropout,
                name=f"memory_encoder_{i}",
            )
            for i in range(num_encoder_layers)
        ]
        # extra embedding as representation of bias before the draft starts. This is grabbed as the
        # representation for the "previous pick" that goes into the decoder for P1P1
        self.card_embedding = nn.ConcatEmbedding(
            self.n_cards + 1,
            emb_dim,
            tf.convert_to_tensor(self.card_data, dtype=tf.float32),
            name="card_embedding",
            activation=None,
        )
        self.decoder_layers = [
            nn.TransformerBlock(
                emb_dim,
                num_decoder_heads,
                pointwise_ffn_width,
                dropout=memory_dropout,
                name=f"memory_decoder_{i}",
                decode=True,
            )
            for i in range(num_decoder_layers)
        ]
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

    @tf.function
    def __call__(
        self, features, training=None, return_attention=False,
    ):
        packs, picks, positions = features
        positional_masks = tf.gather(self.positional_mask, positions)
        positional_embeddings = self.positional_embedding(positions, training=training)
        all_card_embeddings = self.card_embedding(
            tf.range(self.n_cards), training=training
        )
        pack_card_embeddings = (
            packs[:, :, :, None] * all_card_embeddings[None, None, :, :]
        )
        n_options = tf.reduce_sum(packs, axis=-1, keepdims=True)
        pack_embeddings = tf.reduce_sum(pack_card_embeddings, axis=2) / n_options
        embs = pack_embeddings * tf.math.sqrt(self.emb_dim) + positional_embeddings

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

        mask_for_softmax = 1e9 * (1 - packs)
        card_rankings = (
            self.output_decoder(dec_embs, training=training) * packs - mask_for_softmax
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
        output = tf.nn.softmax(card_rankings)

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
        self.set_card_params(self.card_data.iloc[:-1, :])
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


class DeckBuilder(tf.Module):
    def __init__(
        self, n_cards, dropout=0.0, latent_dim=32, embeddings=128, name=None,
    ):
        super().__init__(name=name)
        self.n_cards = n_cards
        # if embeddings is an integer, create a normal embedding of that size
        # otherwise, assume embeddings are pretrained
        if isinstance(embeddings, int):
            emb_trainable = True
            initializer = tf.initializers.glorot_normal()
            emb_init = initializer(shape=(self.n_cards, embeddings))
        else:
            emb_trainable = False
            emb_init = embeddings
        self.card_embeddings = tf.Variable(emb_init, trainable=emb_trainable)
        concat_dim = self.card_embeddings.shape[1] * 2

        self.card_decoder = nn.MLP(
            in_dim=latent_dim,
            start_dim=latent_dim * 2,
            out_dim=self.n_cards,
            n_h_layers=2,
            dropout=0.0,
            name="card_decoder",
            noise=0.0,
            start_act=tf.nn.selu,
            middle_act=tf.nn.selu,
            out_act=tf.nn.sigmoid,
            style="reverse_bottleneck",
        )
        self.basic_decoder = nn.MLP(
            in_dim=self.n_cards,
            start_dim=self.n_cards // 2,
            out_dim=5,
            n_h_layers=2,
            dropout=0.0,
            name="basic_decoder",
            noise=0.0,
            start_act=tf.nn.selu,
            middle_act=tf.nn.selu,
            out_act=tf.nn.softmax,
            style="reverse_bottleneck",
        )
        # only allow more than 18 lands when there are multiple nonbasic lands
        self.determine_n_non_basics = nn.Dense(
            self.n_cards,
            1,
            activation=lambda x: tf.nn.relu(x) * +22.0,
            name="determine_n_non_basics",
        )
        self.merge_deck_and_pool = nn.Dense(
            concat_dim, latent_dim, activation=None, name="merge_deck_and_pool"
        )
        self.dropout = dropout

    # TODO: change input data to not require relaxed shape, and change from
    #      vector of n_cards size to vector of size cards in pool for efficiency
    @tf.function(experimental_relax_shapes=True)
    def __call__(self, features, training=None):
        # batch x sample x n_cards
        pools, decks = features

        self.latent_rep_pool = tf.reduce_sum(
            pools[:, :, :, None] * self.card_embeddings[None, None, :, :], axis=2
        )
        self.latent_rep_deck = tf.reduce_sum(
            decks[:, :, :, None] * self.card_embeddings[None, None, :, :], axis=2
        )

        concat_emb = tf.concat([self.latent_rep_deck, self.latent_rep_pool], axis=-1)
        if self.dropout > 0.0 and training:
            concat_emb = tf.nn.dropout(concat_emb, self.dropout)
        self.latent_rep = self.merge_deck_and_pool(concat_emb, training=training)
        self.cards_to_add = (
            self.card_decoder(self.latent_rep, training=training) * pools
        )
        built_deck = self.cards_to_add + decks
        self.n_non_basics = self.determine_n_non_basics(built_deck, training=training)
        n_basics = 40 - self.n_non_basics

        self.basics_to_add = (
            self.basic_decoder(built_deck, training=training) * n_basics
        )

        return self.basics_to_add, self.cards_to_add, self.n_non_basics

    def compile(
        self,
        card_data,
        learning_rate=0.001,
        basic_lambda=1.0,
        built_lambda=1.0,
        cmc_lambda=0.01,
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

        self.cmc_lambda = cmc_lambda
        self.set_card_params(card_data)
        self.metric_names = metric_names

    def set_card_params(self, card_data):
        self.cmc_map = card_data["cmc"].to_numpy(dtype=np.float32)

    def loss(self, true, pred, sample_weight=None, **kwargs):
        true_basics, true_built = true
        pred_basics, pred_built, _ = pred

        self.basic_loss = tf.reduce_sum(
            tf.reduce_sum(tf.math.square(pred_basics - true_basics), axis=-1)
            * sample_weight
        )
        self.built_loss = tf.reduce_sum(
            tf.reduce_sum(tf.math.square(pred_built - true_built), axis=-1)
            * sample_weight
        )
        if self.cmc_lambda > 0:
            # TODO: test replacing this with KL-Divergence on the distribution of the curve
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

        return (
            self.basic_lambda * self.basic_loss
            + self.built_lambda * self.built_loss
            + self.cmc_lambda * self.curve_incentive
        )

    def compute_metrics(self, true, pred, sample_weight=None, training=None, **kwargs):
        pred_basics, pred_built, _ = pred
        true_basics, true_decks = true
        if sample_weight is None:
            sample_weight = 1.0 / true_decks.shape[0]
        basic_diff = tf.reduce_sum(
            tf.reduce_sum(tf.math.abs(pred_basics - true_basics), axis=-1)
            * sample_weight
        )
        deck_diff = tf.reduce_sum(
            tf.reduce_sum(tf.math.abs(pred_built - true_decks), axis=-1) * sample_weight
        )
        return {"basics_off": basic_diff, "spells_off": deck_diff}

    def save(self, cards, location):
        pathlib.Path(location).mkdir(parents=True, exist_ok=True)
        model_loc = os.path.join(location, "model")
        data_loc = os.path.join(location, "cards.pkl")
        tf.saved_model.save(self, model_loc)
        with open(data_loc, "wb") as f:
            pickle.dump(cards, f)
