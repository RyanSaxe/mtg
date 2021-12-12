import tensorflow as tf
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

class DraftBot(tf.Module):
    def __init__(
        self,
        cards,
        emb_dim,
        t,
        num_heads,
        num_memory_layers,
        emb_dropout=0.0,
        memory_dropout=0.0,
        out_dropout=0.0,
        attention_decoder=True,
        name=None
    ):
        super().__init__(name=name)
        self.idx_to_name = cards.set_index('idx')['name'].to_dict()
        self.n_cards = len(self.idx_to_name)
        self.t = t
        self.emb_dim = tf.Variable(emb_dim, dtype=tf.float32, trainable=False, name="emb_dim")
        self.dropout = emb_dropout
        self.positional_embedding = Embedding(t, emb_dim, name="positional_embedding")
        self.positional_mask = 1 - tf.linalg.band_part(tf.ones((t, t)), -1, 0)
        #MLP where the first hidden layer is of
        # the same size of the input layer to conceptually
        # cover all card x card interactions
        self.pool_pack_embedding = nn.MLP(
            in_dim=self.n_cards * 2,
            start_dim=self.n_cards,
            out_dim=emb_dim,
            n_h_layers=1,
            name="pack_embedding",
            start_act=None,
            middle_act=None,
            out_act=None,
            style="bottleneck",
        )
        self.encoder_layers = [
            MemoryEmbedding(
                self.n_cards,
                emb_dim,
                num_heads,
                dropout=memory_dropout,
                name=f"memory_encoder_{i}"
            )
            for i in range(num_memory_layers)
        ]
        self.attention_decoder = attention_decoder
        if self.attention_decoder:
            self.card_embeddings = Embedding(self.n_cards, emb_dim, name="card_embeddings")
            self.decoder_layers = [
                MemoryEmbedding(
                    self.n_cards,
                    emb_dim,
                    num_heads,
                    dropout=memory_dropout,
                    name=f"memory_decoder_{i}",
                    decode=True,
                    first_decoder_flag=i==0,
                )
                for i in range(num_memory_layers)
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
            out_act=tf.nn.relu,
            style="reverse_bottleneck",
        )


    @tf.function
    def __call__(self, features, training=None, return_attention=False):
        draft_info, picks, positions = features
        packs = draft_info[:, :, :self.n_cards]
        # pools = draft_info[:, :, self.n_cards:]
        # draft_info is of shape (batch_size, t, n_cards * 2)
        positional_masks = tf.gather(self.positional_mask, positions)
        positional_embeddings = self.positional_embedding(positions, training=training)
        #old way: pack embedding = mean of card embeddings for only cards in the pack
        #pack_embeddings = tf.reduce_sum(packs[:,:,:,None] * self.card_embeddings[None,None,:,:], axis=2)/tf.reduce_sum(packs, axis=-1, keepdims=True)
        pack_embeddings = self.pool_pack_embedding(draft_info)
        embs = pack_embeddings * tf.math.sqrt(self.emb_dim) + positional_embeddings
        if training and self.dropout > 0.0:
            embs = tf.nn.dropout(embs, rate=self.dropout)
        for memory_layer in self.encoder_layers:
            embs, attention_weights = memory_layer(embs, positional_masks, training=training) # (batch_size, t, emb_dim)
        if self.attention_decoder:
            dec_embs = tf.gather(self.card_embeddings, picks)
            for memory_layer in self.decoder_layers:
                dec_embs, attention_weights = memory_layer(dec_embs, positional_masks, encoder_output=embs, training=training) # (batch_size, t, emb_dim)
            embs = dec_embs
        card_rankings = self.output_decoder(embs, training=training) # (batch_size, t, n_cards)
        # zero out the rankings for cards not in the pack
        # note1: this only works because no foils on arena means packs can never have 2x of a card
        #       if this changes, modify to clip packs at 1
        # note2: this zeros out the gradients for the cards not in the pack in order to not negatively
        #        affect backprop on cards that would techncally be taken if they were in the pack. However,
        #        if it turns out that there is a reason why these gradients shouldn't be zero, this
        #        multiplication could be done only during inference (when training is not True)

        # add epsilon for cards in the pack to ensure they are non-zero (handles edge cases)
        card_rankings = card_rankings * packs + 1e-9 * packs
        # after zeroing out cards not in packs, we readjust the output to maintain that it sums to one
        # note: currently this sums to one so we do from_logits=True in Categorical Cross Entropy,
        #       possible softmax is better than relu, regardless this does have numerical instability issues
        #       so that is something to look out for. But from_logits=False had terrible performance
        output = card_rankings/tf.reduce_sum(card_rankings, axis=-1, keepdims=True)
        if return_attention:
            return output, attention_weights
        return output

    def compile(
        self,
        optimizer=None,
        learning_rate=0.001
    ):
        if optimizer is None:
            if isinstance(learning_rate, dict):
                learning_rate = CustomSchedule(self.emb_dim, **learning_rate)
            else:
                learning_rate = learning_rate

            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98,epsilon=1e-9)
        else:
            self.optimizer = optimizer
        self.loss_f = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    def loss(self, true, pred, sample_weight=None, store=True):
        return self.loss_f(true, pred, sample_weight=sample_weight)

    def compute_metrics(self, true, pred, sample_weight=None):
        top1 = tf.reduce_mean(tf.keras.metrics.sparse_top_k_categorical_accuracy(true, pred, 1))
        top2 = tf.reduce_mean(tf.keras.metrics.sparse_top_k_categorical_accuracy(true, pred, 2))
        top3 = tf.reduce_mean(tf.keras.metrics.sparse_top_k_categorical_accuracy(true, pred, 3))
        return top1, top2, top3

    def save(self, location):
        pathlib.Path(location).mkdir(parents=True, exist_ok=True)
        model_loc = os.path.join(location,"model")
        tf.saved_model.save(self,model_loc)
        data_loc = os.path.join(location,"attrs.pkl")
        with open(data_loc,'wb') as f:
            attrs = {
                't': self.t,
                'idx_to_name': self.idx_to_name,
                'n_cards': self.n_cards
            }
            pickle.dump(attrs,f) 

class MemoryEmbedding(tf.Module):
    """
    self attention block for encorporating memory into the draft bot
    """
    def __init__(self, n_cards, emb_dim, num_heads, dropout=0.0, decode=False, first_decoder_flag=False, name=None):
        super().__init__(name=name)
        self.dropout = dropout
        self.decode = decode
        self.first_decoder_flag = first_decoder_flag
        #kdim and dmodel are the same because the embedding dimension of the non-attended
        # embeddings are the same as the attention embeddings.
        self.attention = MultiHeadAttention(emb_dim, emb_dim, num_heads, name=self.name + "_attention")
        self.expand_attention = Dense(emb_dim, n_cards, activation=tf.nn.relu, name=self.name + "_pointwise_in")
        self.compress_expansion = Dense(n_cards, emb_dim, activation=None, name=self.name + "_pointwise_out")
        if not self.first_decoder_flag:           
            self.attention_layer_norm = LayerNormalization(emb_dim, name=self.name + "_attention_norm")
        self.final_layer_norm = LayerNormalization(emb_dim, name=self.name + "_out_norm")
        if self.decode:
            self.decode_attention = MultiHeadAttention(emb_dim, emb_dim, num_heads, name=self.name + "_decode_attention")
            self.decode_layer_norm = LayerNormalization(emb_dim, name=self.name + "_decode_norm")
    
    def pointwise_fnn(self, x, training=None):
        x = self.expand_attention(x, training=training)
        return self.compress_expansion(x, training=training)

    def __call__(self, x, mask, encoder_output=None, training=None):
        if self.first_decoder_flag:
            decoder_mask = mask + tf.eye(mask.shape[1], batch_shape=[mask.shape[0]])
            # x is the pick here, which means we are not allowed to look at it in order to make the prediction
            #     normally, we can look at the current time and everything before, but for the decoder we
            #     are only allowed to look before it, which is what subtracting tf.eye accomplishes
            attention_emb, attention_weights = self.attention(x, x, x, decoder_mask, training=training)
        else:
            attention_emb, attention_weights = self.attention(x, x, x, mask, training=training)
        if training and self.dropout > 0:
            attention_emb = tf.nn.dropout(attention_emb, rate=self.dropout)
        if self.first_decoder_flag:
            # x is the pick here, which means adding a residual connection to it is leakage
            residual_emb_w_memory = attention_emb
        else:
            residual_emb_w_memory = self.attention_layer_norm(x + attention_emb, training=training)
        if self.decode:
            assert encoder_output is not None
            decode_attention_emb, decode_attention_weights = self.decode_attention(
                encoder_output,
                encoder_output,
                residual_emb_w_memory,
                mask,
                training=training
            )
            if training and self.dropout > 0:
                decode_attention_emb = tf.nn.dropout(decode_attention_emb, rate=self.dropout)
            residual_emb_w_memory = self.decode_layer_norm(residual_emb_w_memory + decode_attention_emb, training=training)
            attention_weights = (attention_weights, decode_attention_weights)
        process_emb = self.pointwise_fnn(residual_emb_w_memory, training=training)
        if training and self.dropout > 0:
            process_emb = tf.nn.dropout(process_emb, rate=self.dropout)
        return self.final_layer_norm(residual_emb_w_memory + process_emb, training=training), attention_weights

class DeckBuilder(tf.Module):
    def __init__(self, n_cards, dropout=0.0, embeddings=None, name=None):
        super().__init__(name=name)
        self.n_cards = n_cards
        if embeddings is None:
            self.embeddings = None
            encoder_in_dim = self.n_cards
        else:
            #if embeddings is an integer, learn embeddings of that dimension,
            #if embeddings is None, don't use embeddings
            #otherwise, assume embeddings are pretrained and use them
            if isinstance(embeddings, int):
                emb_trainable = True
                initializer = tf.initializers.glorot_normal()
                emb_init = initializer(shape=(self.n_cards, embeddings))
            else:
                emb_trainable = False
                emb_init = embeddings
            self.embeddings = tf.Variable(emb_init, trainable=emb_trainable)
            encoder_in_dim = self.embeddings.shape[0] * (self.embeddings.shape[1])
        self.encoder = nn.MLP(
            in_dim=encoder_in_dim,
            start_dim=256,
            out_dim=32,
            n_h_layers=2,
            dropout=dropout,
            name="encoder",
            noise=0.0,
            start_act=tf.nn.selu,
            middle_act=tf.nn.selu,
            out_act=tf.nn.selu,
            style="bottleneck"
        )
        self.decoder = nn.MLP(
            in_dim=32,
            start_dim=64,
            out_dim=self.n_cards,
            n_h_layers=2,
            dropout=0.0,
            name="decoder",
            noise=0.0,
            start_act=tf.nn.selu,
            middle_act=tf.nn.selu,
            out_act=tf.nn.sigmoid,
            style="reverse_bottleneck"
        )
        #self.interactions = nn.Dense(self.n_cards, self.n_cards, activation=None)
        self.add_basics_to_deck = nn.Dense(32,5, activation=lambda x: tf.nn.sigmoid(x) * 18.0)

    def convert_pools_to_flattened_card_embeddings(self, pools):
        #expand dims of the pools so we can add dimension to use embeddings per card
        expanded_pools = tf.expand_dims(pools, axis=-1)
        #convert multiples into binary to use as multiplicative mask
        #in_pool_mask = tf.clip_by_value(expanded_pools,0,1)
        #pools = batch x n_cards 
        #embs = n_cards x emb_size
        # -> expand dims such that (batch x n_cards x 1) * (1 x n_cards x emb_size) = (batch x n_cards x emb_size)
        expanded_embs = tf.expand_dims(self.embeddings, axis=0)
        pool_embs = expanded_pools * expanded_embs
        # now, we want to add to the end of each embedding the number of the card in the pool
        # -> dimension now is (batch x n_cards x emb_size + 1)
        # pool_embs_w_card_count = tf.concat([
        #         pool_embs,
        #         expanded_pools
        #     ],
        #     axis=-1
        # )
        # by flattening out the last two dimensions, we have an input that is permutation
        # invariant. No matter the cards in the pool, the 4th feature of the embedding for
        # the 8th card will be located at the index 7 * (emb_size) + 4
        # shape = pool_embs_w_card_count.shape
        shape = pool_embs.shape
        return tf.reshape(pool_embs, [shape[0], shape[1] * shape[2]])

    @tf.function
    def __call__(self, decks, training=None):
        basics = decks[:,:5]
        pools = decks[:,5:] 
        if self.embeddings is not None:
            pool_embs = self.convert_pools_to_flattened_card_embeddings(pools)
        else:
            pool_embs = pools
        #self.pool_interactions = self.interactions(pools)
        # project the deck to a lower dimensional represnetation
        self.latent_rep = self.encoder(pool_embs, training=training)
        # project the latent representation to a potential output
        reconstruction = self.decoder(self.latent_rep, training=training)
        basics = self.add_basics_to_deck(self.latent_rep,  training=training)
        built_deck = tf.concat([basics, reconstruction * pools], axis=1)
        return built_deck

    def compile(
        self,
        cards=None,
        basic_lambda=1.0,
        built_lambda=1.0,
        cmc_lambda=0.01,
        # interaction_lambda=0.01,
        optimizer=None,
    ):
        self.optimizer = tf.optimizers.Adam() if optimizer is None else optimizer

        self.basic_lambda = basic_lambda
        self.built_lambda = built_lambda

        self.built_loss_f = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.basic_loss_f = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        self.cmc_lambda = cmc_lambda
        # self.interaction_lambda = interaction_lambda
        if cards is not None:
            self.set_card_params(cards)

    def set_card_params(self, cards):
        self.cmc_map = cards.sort_values(by='idx')['cmc'].to_numpy(dtype=np.float32)

    def loss(self, true, pred, sample_weight=None):
        true_basics,true_built = tf.split(true,[5,self.n_cards],1)
        pred_basics,pred_built = tf.split(pred,[5,self.n_cards],1)
        self.basic_loss = self.basic_loss_f(true_basics, pred_basics, sample_weight=sample_weight)
        self.built_loss = self.built_loss_f(true_built, pred_built, sample_weight=sample_weight)
        if self.cmc_lambda > 0:
            #pred_built instead of pred to avoid learning to add more basics
            #add a thing here to avoid all lands in general later
            self.curve_incentive = tf.reduce_mean(
                tf.multiply(pred_built,tf.expand_dims(self.cmc_map[5:],0)),
                axis=1
            )
        else:
            self.curve_incentive = 0.0
        # if self.interaction_lambda > 0:
        #     #push card level interactions in pool to zero
        #     self.interaction_reg = tf.norm(self.interactions.w,ord=1)
        # else:
        #     self.interaction_reg = 0.0
        return (
            self.basic_lambda * self.basic_loss + 
            self.built_lambda * self.built_loss +
            self.cmc_lambda * self.curve_incentive
            # self.interaction_lambda * self.interaction_reg
        )

    def save(self, cards, location):
        pathlib.Path(location).mkdir(parents=True, exist_ok=True)
        model_loc = os.path.join(location,"model")
        data_loc = os.path.join(location,"cards.pkl")
        tf.saved_model.save(self,model_loc)
        with open(data_loc,'wb') as f:
            pickle.dump(cards,f) 