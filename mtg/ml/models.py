import tensorflow as tf
from mtg.ml import nn
from mtg.ml.layers import sawtooth
import numpy as np
import pandas as pd
import pdb
#todo for stream:
# separate out basics to build your own
# add more priors
# add more data via feature engineering and augmentation
# add sideboard probability relationship to dataset so that rares can change correct build

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
        self.interactions = nn.Dense(self.n_cards, self.n_cards, activation=tf.nn.relu)
        self.add_basics_to_deck = nn.Dense(self.n_cards,5, activation=lambda x: tf.nn.sigmoid(x) * 18.0)

    @tf.function
    def __call__(self, decks, training=None):
        # noisy_decks is a temporary process until we get SB data
        basics = decks[:,:5]
        pools = decks[:,5:] 
        interactions = self.interactions(pools)
        # project the deck to a lower dimensional represnetation
        self.latent_rep = self.encoder(interactions)
        # project the latent representation to a potential output
        reconstruction = self.decoder(self.latent_rep)
        # originally I had the basics go off the latent representation, but 
        # I got scenarios with heavy off-color sideboards influencing basics
        basics = self.add_basics_to_deck(reconstruction)
        if training is None:
            built_deck = tf.concat([basics, reconstruction * pools], axis=1)
        else:
            built_deck = tf.concat([basics, reconstruction], axis=1)
        return built_deck

    def compile(
        self,
        basic_lambda=1.0,
        built_lambda=1.0,
        optimizer=None
    ):
        self.optimizer = tf.optimizers.Adam(lr=0.001) if optimizer is None else optimizer
        self.basic_lambda = basic_lambda
        self.built_lambda = built_lambda
        self.built_loss = tf.keras.losses.BinaryCrossentropy()
        self.basic_loss = tf.keras.losses.MSE

    def loss(self, true, pred, sample_weight=None):
        true_basics,true_built = tf.split(true,[5,280],1)
        pred_basics,pred_built = tf.split(pred,[5,280],1)
        basic_loss = self.basic_loss(true_basics, pred_basics)
        built_loss = self.built_loss(true_built, pred_built, sample_weight=sample_weight)
        return self.basic_lambda * basic_loss + self.built_lambda * built_loss

    def save(self, location):
        tf.saved_model.save(self,location)



