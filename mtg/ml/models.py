import tensorflow as tf
from mtg.ml import nn
from mtg.ml.layers import harmonic_mean
import numpy as np
import pandas as pd
import pdb
import pathlib
import os
import pickle
from mtg.obj.cards import CardSet


# to add:
#    struggling to not play playables rares where they don't make sense
#    basic land count needs work --- lands in general --- figure out how to regularize

class DeckBuilder(tf.Module):
    def __init__(self, n_cards, dropout=0.0, name=None):
        super().__init__(name=name)
        self.n_cards = n_cards - 5
        #probability of random sampling a card similar to that of a SB slot
        self.encoder = nn.MLP(
            in_dim=self.n_cards,
            start_dim=256,
            out_dim=32,
            n_h_layers=2,
            dropout=dropout,
            name="encoder",
            noise=0.0,
            start_act=tf.nn.relu,
            middle_act=tf.nn.relu,
            out_act=tf.nn.relu,
            style="bottleneck"
        )
        self.decoder = nn.MLP(
            in_dim=32,
            start_dim=64,
            out_dim=self.n_cards,
            n_h_layers=2,
            dropout=dropout,
            name="decoder",
            noise=0.0,
            start_act=tf.nn.relu,
            middle_act=tf.nn.relu,
            out_act=tf.nn.sigmoid,
            style="reverse_bottleneck"
        )
        #self.interactions = nn.Dense(self.n_cards, self.n_cards, activation=None)
        self.add_basics_to_deck = nn.Dense(self.n_cards,5, activation=lambda x: tf.nn.sigmoid(x) * 18.0)

    @tf.function
    def __call__(self, decks, training=None):
        # noisy_decks is a temporary process until we get SB data
        basics = decks[:,:5]
        pools = decks[:,5:] 
        #self.pool_interactions = self.interactions(pools)
        # project the deck to a lower dimensional represnetation
        self.latent_rep = self.encoder(pools)
        # project the latent representation to a potential output
        reconstruction = self.decoder(self.latent_rep)
        basics = self.add_basics_to_deck(self.latent_rep)
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



