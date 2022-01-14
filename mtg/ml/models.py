import tensorflow as tf
from mtg.ml import nn
from mtg.ml.layers import Embedding
import numpy as np
import pathlib
import os
import pickle
from mtg.ml.utils import CustomSchedule


class DraftBot(tf.Module):
    """
    Custom Tensorflow Model for Magic: the Gathering Draft AI

    This algorithm is a transformer that functions on draft data modified to
        work in a sequence-to-sequence manner. Given a sequence of packs and
        picks, as well as a contextual pack that determines your options, the
        goal is to select the card from the context that is best (best determined
        via "what a human did, with weighting towards more experienced humans)

    --------------------------------------------------------------------------

    expansion:              Expansion object instance from mtg/obj/expansion.py
    emb_dim:                The embedding dimension to use for card embeddings
    num_encoder_heads:      Number of heads in each encoder transformer block
    num_decoder_heads:      Number of heads in each decoder transformer block
    num_encoder_layers:     Number of transformer blocks in the encoder
    num_decoder_layers:     Number of transformer blocks in the decoder
    pointwise_ffn_width:    The width of the pointwise feedforward NN projection
                                in each transformer block
    emb_dropout:            Dropout rate to be applied to card embeddings
    memory_dropout:         Dropout rate to be applied to the transformer blocks
    out_dropout:            Dropout rate to be applied to the hidden layers in the
                                MLP that converts the output from the transformer
                                decoder to the prediction of what card to take
    """

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
        # get some information from the expansion object for storage later. This is
        #     because we don't want to store the expansion object (it's big), and in
        #     case we lose it, we need to be able to initialize a new one with the
        #     same exact card to id mappings for proper inference.
        self.idx_to_name = expansion.get_mapping("idx", "name", include_basics=False)
        self.n_cards = len(self.idx_to_name)
        # self.t is the number of picks in a draft
        self.t = expansion.t
        # the first five elements will be card data on basics, which is irrelevant
        #     for drafting, so we get rid of them
        self.card_data = expansion.card_data_for_ML[5:]
        self.emb_dim = tf.Variable(
            emb_dim, dtype=tf.float32, trainable=False, name="emb_dim"
        )
        self.dropout = emb_dropout
        # positional embedding allows deviation given temporal context
        self.positional_embedding = Embedding(
            self.t, emb_dim, name="positional_embedding"
        )
        # lookahead mask to prevent the algorithm from seeing information it isn't
        #     allowed to (e.g. at P1P5 you cannot look at P1P6-P3P14)
        self.positional_mask = 1 - tf.linalg.band_part(tf.ones((self.t, self.t)), -1, 0)
        # transformer encoder block for processing pack information
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
        # additionally, we use a "concatEmbedding", which means we do the following:
        #     1. project a one_hot_vector to an embedding of dimension emb_dim//2
        #     2. use an MLP on the data about each card (self.card_data) to yield an
        #        emb_dim//2 dimension embedding
        #     3. The embedding we use for cards is the concatenation of 1. and 2.
        self.card_embedding = nn.ConcatEmbedding(
            self.n_cards + 1,
            emb_dim,
            tf.convert_to_tensor(self.card_data, dtype=tf.float32),
            name="card_embedding",
            activation=None,
        )
        # transformer decoder block for processing the pool with respect to the pack
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
        # convert transformer decoder output to projection of what card to pick
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
        self,
        features,
        training=None,
        return_attention=False,
    ):
        packs, picks, positions = features
        # get the positional mask, which is a lookahead mask for autoregressive predictions.
        #    effectively, to make a decision a P1P5, we make sure the model can never see P1P6
        #    or later
        positional_masks = tf.gather(self.positional_mask, positions)
        # to make sure the model can differentiate context of a pool and pack at different time
        #    steps, we have positional embeddings
        #    (e.g. representation of card A at P1P1 is different than P1P8)
        positional_embeddings = self.positional_embedding(positions, training=training)
        all_card_embeddings = self.card_embedding(
            tf.range(self.n_cards), training=training
        )
        # TODO: represent packs as 15 indices for each card in the pack rather than a
        #       binary vector. It's more computationally efficient and doesn't require
        #       the step below
        pack_card_embeddings = (
            packs[:, :, :, None] * all_card_embeddings[None, None, :, :]
        )
        # get the number of cards in each pack
        n_options = tf.reduce_sum(packs, axis=-1, keepdims=True)
        # the pack_embedding is the average of the embeddings of the cards in the pack
        pack_embeddings = tf.reduce_sum(pack_card_embeddings, axis=2) / n_options
        # add the positional information to the card embeddings
        embs = pack_embeddings * tf.math.sqrt(self.emb_dim) + positional_embeddings

        if training and self.dropout > 0.0:
            embs = tf.nn.dropout(embs, rate=self.dropout)

        # we run the transformer encoder on the pack information. This is where the
        #     bot learns how to predict the wheel. Search for improvements on how
        #     this informs color distribution/expectation and pivots
        for memory_layer in self.encoder_layers:
            embs, attention_weights_pack = memory_layer(
                embs, positional_masks, training=training
            )  # (batch_size, t, emb_dim)

        # we run the transformer decoder on the pick information. So, at P1P5 decision,
        #     the transformer gets passed what the human took at P1P4. Attention with a
        #     lookahead mask lets the pick information represent the whole pool, because
        #     the algorithm attends to prior picks, so at P1P5 the decoder looks at
        #     P1P1-P1P4, which is the pool.
        #
        # NOTE: at P1P1, we represent the pick (since there's no prior info) with a
        #       vector representationt that is meant to describe the bias at the beginning
        #       of the draft.
        # TODO: explore adding positional information to the picks here. Should it be the
        #       same positional embedding, or a different one?
        dec_embs = self.card_embedding(picks, training=training)
        if training and self.dropout > 0.0:
            dec_embs = tf.nn.dropout(dec_embs, rate=self.dropout)

        for memory_layer in self.decoder_layers:
            dec_embs, attention_weights_pick = memory_layer(
                dec_embs, positional_masks, encoder_output=embs, training=training
            )  # (batch_size, t, emb_dim)

        # in order to remove all cards in the set not in the pack as options, we create a
        #     mask that will guarantee the values will be zero when applying softmax
        mask_for_softmax = 1e9 * (1 - packs)
        card_rankings = (
            self.output_decoder(dec_embs, training=training) * packs - mask_for_softmax
        )  # (batch_size, t, n_cards)
        # compute the euclidian distance between each card embedding from the pack and
        #     the output of the transformer decoder. This is used to regularize the network
        #     by saying "the embedding for the correct pick should be close to the output
        #     from the transformer, and far from the other cards in the pack". Conceptually
        #     taken from this paper: https://ieee-cog.org/2021/assets/papers/paper_75.pdf.
        # NOTE: I tested the direct implementation of this paper where, rather than using
        #       `self.output_decoder` to determine the card rankings, you just directly pick
        #       the card with the closest distance to the output of the context (transformer
        #       decoder). This consistently lagged behind using the decoder on validation
        #       performance. Still a lot to experiment with the embedding space.
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
        rare_lambda=10.0,
        cmc_lambda=1.0,
        cmc_margin=1.0,
        metric_names=["top1", "top2", "top3"],
    ):
        """
        After initializing the model, we want to compile it by setting parameters for training

        optimizer:              optimizer to use for minimizing the objective function (default is Adam)
        learning_rate:          learning rate for the optimizer. If passing {'lr_warmup':N}, it will use
                                    Adam with a scheduler that warms up the LR. This is recommended.
        margin:                 the minimal distance margin for triplet loss on the card embeddings
        emb_lambda:             the regularization coefficient for triplet loss on the card embeddings
        pred_lambda:            the coefficent for the main prediction task of the loss function
        bad_behavior_lambda:    the regularization coefficient to be applied to all penalties that
                                    are structured as expert priors to avoid learning unwanted behavior
                                    such as rare drafting
        rare_lambda:            the regularization coefficient for the penalty on taking rares when the
                                    bot shouldn't
        cmc_lambda:             the regularization coefficient for the penalty that asks the model to bias
                                    towards cheaper cards.
        cmc_margin:             the minimal distance margin for when to incur a penalty for taking expensive
                                    cards. For example: if cmc_margin = 1.5, the bot confidently wants to take
                                    a 5 drop, but the human takes a 3-drop, we incur a penalty of 0.5. If the
                                    bot takes a card with cmc four or less, the penalty would be zero.
        metric_names:            a list of which metrics to use to help debug.
        """
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
        # because our output is softmax, CategoricalCrossentropy is the proper loss function
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
        """
        Create attributes that allow us to do computations with card level data
        """
        # this flag allows us to incur extra regularization penalty when it appears like
        #     the model has improperly learned to rare draft
        self.rare_flag = (card_data["mythic"] + card_data["rare"]).values[None, None, :]
        # this lets us easily convert packs or pools to cmc representations to help the
        #     model bias towards cheaper cards in general
        self.cmc = card_data["cmc"].values[None, None, :]

    def loss(self, true, pred, sample_weight=None, training=None, **kwargs):
        """
        implementation of the loss function.
        """
        pred, emb_dists = pred
        # CategoricalCrossentropy loss applied to what the human took vs softmax output
        self.prediction_loss = self.loss_f(true, pred, sample_weight=sample_weight)
        # get the one hot representation of what the human picked
        correct_one_hot = tf.one_hot(true, self.n_cards)
        # get the distance between the incorrect picks and the transformer output
        dist_of_not_correct = emb_dists * (1 - correct_one_hot)
        # get the distance between the correct pick and the transformer output
        dist_of_correct = tf.reduce_sum(
            emb_dists * correct_one_hot, axis=-1, keepdims=True
        )
        # we want the distance from the correct pick to transformer output to be smaller
        #    than the incorrect picks to the transformer output. So we do the following
        #    subtraction because `dist_loss` will be negative in that case. This is where
        #    the `margin` parameter comes into play. We want this subtraction to yield
        #    *at least* -self.margin. Otherwise, we incure a penalty.
        dist_loss = dist_of_correct - dist_of_not_correct
        sample_weight = 1 if sample_weight is None else sample_weight
        self.embedding_loss = tf.reduce_sum(
            tf.reduce_sum(tf.maximum(dist_loss + self.margin, 0.0), axis=-1)
            * sample_weight
        )
        # compute loss from expert priors (e.g. no rare drafting, take cheaper cards)
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
        # penalize taking rares when the human doesn't. This helps for generalization. Think
        #    about it like this: people *love* taking rares. This means, when they choose not
        #    to take a rare, that pick is likely important and full of information we want to
        #    learn. Hence, we incur a *massive* penalty (this is why default rare_lambda=10.0)
        #    to tell the model "when a person doesn't take a rare, you better pay attention".
        #    Additionally, this prevents the model from learning to rare draft!
        human_took_rare = tf.reduce_sum(true_one_hot * self.rare_flag, axis=-1)
        pred_rare_val = tf.reduce_sum(pred * self.rare_flag, axis=-1)
        rare_loss = (1 - human_took_rare) * pred_rare_val * self.rare_lambda
        self.rare_loss = tf.reduce_sum(rare_loss * sample_weight)
        return self.cmc_loss + self.rare_loss

    def compute_metrics(self, true, pred, sample_weight=None, **kwargs):
        """
        compute top1, top2, top3 accuracy to display as metrics during training when verbose=True
        """
        if sample_weight is None:
            sample_weight = tf.ones_like(true.shape) / (true.shape[0] * true.shape[1])
        sample_weight = sample_weight.flatten()
        pred, _ = pred
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
        """
        store the trained model and important attributes from the model to a file
        """
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
    """
    Custom Tensorflow Model for Magic: the Gathering DeckBuilder AI

    This algorithm is an Denoising AutoEncoder:
        Deckbuilding in Limited is about taking a card pool, and yielding a
        subset of that pool as a deck, which is effectively "denoising" the pool
        by removing the sideboard cards from it.

    However, just the Denoising AutoEncoder has a few problems.
        1. It doesn't address basics
        2. It has difficulties during inference because inference is a discrete
                problem, and training is continuous

    Addressing basics:
        Observe that adding basics to a deck is a function of the final deck, and
        not the direct card pool. So, let DeckBuilder(pool) -> deck_projection. Then,
        we want to learn an additional function F(deck_projection) -> basics.

    Addressing inference:
        If we just iteratively take the argmax of the output from DeckBuilder(pool),
        then multiples of cards are treated poorly. If a pool contains two copies of
        Card A, and one copy of Card B, how should you determine what to add to the
        deck when the model says "add 1.7 copies of card A and 0.75 copies of card B"?
        If you only have a few slots left, should you add two copies of A and 0 of B?
        1 and 1? It's unclear, and often yields issues (empirically it did at least).

        So, we modify the problem such that an iterative argmax makes sense. Rather than
        having the input just be a pool, we pass the model the available pool, and the
        current deck, where the current deck can be of any size and simply represents
        "cards in the pool that MUST be added to the deck at the end". This way, at
        inference, we can do the following:

            1. Pass a full pool and an empty deck
            2. Allocate the argmax of the output to the deck, and subtract it from the pool
            3. Run the model again, and repeat until the model says to stop adding cards
            4. Take the final allocation, and pass it to F mentioned in Addressing basics
                    section to yield the basics corresponding to the pool.

        In order to accomplish this, we generate deck data as follows:
            1. Sample a data point, which contains a deck and a sideboard
            2. Sample N cards from the deck, set that to your target
            3. Remove those N points from the deck, and add them to the sideboard
            4. Now the sideboard is the options, and the deck is currently allocated cards!
            5. If you'd like to view the code, look at DeckGenerator in mtg/ml.generator.py

    -------------------------------------------------------------------------------------------

    n_cards:    number of cards in the set, EXCLUDING basics
    dropout:    Dropout rate for the encoders for the pool and partial deck
    latent_dim: The input (pool, partial deck) pair gets projected to a latent
                    space of this dimension
    embeddings: The dimension for card embeddings. If a matrix is passed, it is
                    treated as pretrained card embeddings and frozen.
    """

    def __init__(
        self,
        n_cards,
        dropout=0.0,
        latent_dim=32,
        embeddings=128,
        name=None,
    ):
        super().__init__(name=name)
        self.n_cards = n_cards
        if isinstance(embeddings, int):
            emb_trainable = True
            initializer = tf.initializers.glorot_normal()
            emb_init = initializer(shape=(self.n_cards, embeddings))
        else:
            emb_trainable = False
            emb_init = embeddings
        self.card_embeddings = tf.Variable(emb_init, trainable=emb_trainable)
        # we use card embeddings to project the pool and partial deck to a vector
        #    space, we concatenate them and then project to `latent_dim`, so concat
        #    dim is always the embedding dimension * 2
        concat_dim = self.card_embeddings.shape[1] * 2
        # MLP that takes the latent representation and decodes to the projection of
        # what cards to add to the deck.
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
        # MLP that takes the projection of final deck and adds the basics
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
        # We learn to determine the number of non basics from a fully built deck to
        #    properly allocate the number of basics, as well as terminate the iterative
        #    process during inference.
        # TODO: experiment with using a sum along the last axis rather than a Dense layer,
        #       although I expect this to have issues because the output is not discrete
        # NOTE: activation of relu + 22 -> a minimum of 22 non-basics. This does mean that it
        #       is impossible to play more than 18 lands without cards like evolving wilds, and
        #       may improperly bias away from playing 18 lands when necessary.
        self.determine_n_non_basics = nn.Dense(
            self.n_cards,
            1,
            activation=lambda x: tf.nn.relu(x) + 22.0,
            name="determine_n_non_basics",
        )
        # Dense layer that takes the concatenated pool and partial deck embeddings and
        #     projects it to the latent representation of the deck
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
        # project pool and partial deck to their respective latent space as sums of
        #     card embeddings
        self.latent_rep_pool = tf.reduce_sum(
            pools[:, :, :, None] * self.card_embeddings[None, None, :, :], axis=2
        )
        self.latent_rep_deck = tf.reduce_sum(
            decks[:, :, :, None] * self.card_embeddings[None, None, :, :], axis=2
        )
        # concatenate representation of pool and partial deck
        concat_emb = tf.concat([self.latent_rep_deck, self.latent_rep_pool], axis=-1)
        if self.dropout > 0.0 and training:
            concat_emb = tf.nn.dropout(concat_emb, self.dropout)
        # yield final latent representation of deck
        self.latent_rep = self.merge_deck_and_pool(concat_emb, training=training)
        # compute the cards to add from the available pool
        self.cards_to_add = (
            self.card_decoder(self.latent_rep, training=training) * pools
        )
        # the final built deck is equal to the cards we want to allocate from the pool
        #     added to the partial deck input of cards already allocated to the deck
        built_deck = self.cards_to_add + decks
        # given the deck, determine how many non-basics, and hence basics, we want
        self.n_non_basics = self.determine_n_non_basics(built_deck, training=training)
        n_basics = 40 - self.n_non_basics
        # finally, add the basics to the deck!
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
        """
        After initializing the model, we want to compile it by setting parameters for training

        optimizer:              optimizer to use for minimizing the objective function (default is Adam)
        learning_rate:          learning rate for the optimizer. If passing {'lr_warmup':N}, it will use
                                    Adam with a scheduler that warms up the LR. This is recommended.
        basic_lambda:           the coefficient for matching the basics the human chose to add
        built_lambda:           the coefficent for matching the non-basics the human chose to add
        cmc_lambda:             the regularization coefficient for the penalty that asks the model to bias
                                    towards building decks with similar curves to humans (needs improvement)
        metric_names:            a list of which metrics to use to help debug.
        """
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
        """
        Create attributes that allow us to do computations with card level data
        """
        # this lets us easily compute the distribution of cmc in any deck to help
        #    the model yield similar curves to how humans approach deckbuilding
        self.cmc_map = card_data["cmc"].to_numpy(dtype=np.float32)

    def loss(self, true, pred, sample_weight=None, **kwargs):
        """
        implementation of the loss function. It is currently using MSE instead of MAE. MAE is
            intuitively a better fit because it would yield more sparse predictions. However,
            I empirically found this yielded problematic generalization results because it pushed
            the value of all cards in consideration to 1, which meant taking the argmax was often
            insufficient for building decks during inference.
        """
        true_basics, true_built = true
        pred_basics, pred_built, _ = pred
        # penalize the model for improperly allocating basic lands
        self.basic_loss = tf.reduce_sum(
            tf.reduce_sum(tf.math.square(pred_basics - true_basics), axis=-1)
            * sample_weight
        )
        # penalize the model for impoperly allocating non-basic-lands and spells
        self.built_loss = tf.reduce_sum(
            tf.reduce_sum(tf.math.square(pred_built - true_built), axis=-1)
            * sample_weight
        )
        # penalize the model for deviating from the average curve of the deck a person built
        if self.cmc_lambda > 0:
            # TODO: test replacing this with KL-Divergence on the distribution of the curve
            #       hopefully this helps the model play worse two-drops when needed, which it
            #       is currently not great at (it definitely does it, but not enough)
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
        # compute the average number of basics off the model is from human builds
        basic_diff = tf.reduce_sum(
            tf.reduce_sum(tf.math.abs(pred_basics - true_basics), axis=-1)
            * sample_weight
        )
        # compute the average number of non-basics and spells off the model is from human builds
        deck_diff = tf.reduce_sum(
            tf.reduce_sum(tf.math.abs(pred_built - true_decks), axis=-1) * sample_weight
        )
        return {"basics_off": basic_diff, "spells_off": deck_diff}

    def save(self, cards, location):
        """
        store the trained model and card object to a file
        """
        pathlib.Path(location).mkdir(parents=True, exist_ok=True)
        model_loc = os.path.join(location, "model")
        data_loc = os.path.join(location, "cards.pkl")
        tf.saved_model.save(self, model_loc)
        with open(data_loc, "wb") as f:
            pickle.dump(cards, f)
