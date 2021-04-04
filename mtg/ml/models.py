import tensorflow as tf
from mtg.ml import nn
import numpy as np
import pandas as pd
import pdb
#todo for stream:
# separate out basics to build your own
# add more priors
# add more data via feature engineering and augmentation
# add sideboard probability relationship to dataset so that rares can change correct build

class DeckBuilder(tf.Module):
    def __init__(self, n_cards, land_idxs, name=None):
        super().__init__(name=name)
        self.n_cards = n_cards - 5
        self.land_idxs = land_idxs
        #probability of random sampling a card similar to that of a SB slot
        self.encoder = nn.MLP(
            in_dim=self.n_cards,
            start_dim=256,
            out_dim=32,
            n_h_layers=2,
            dropout=0.0,
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
            dropout=0.0,
            name="decoder",
            noise=0.0,
            start_act=tf.nn.relu,
            middle_act=tf.nn.relu,
            out_act=tf.nn.sigmoid,
            style="reverse_bottleneck"
        )
        self.interactions = nn.Dense(self.n_cards, self.n_cards, activation=tf.nn.relu)
        self.add_basics_to_deck = nn.Dense(self.n_cards,5, activation=lambda x: tf.nn.sigmoid(x) * 18.0)

    def __call__(self, decks, training=None):
        # noisy_decks is a temporary process until we get SB data
        basics = decks[:,:5]
        pools = decks[:,5:] 
        self.noisy_decks = self.fake_sideboard(pools)
        # first layer is of same dim as number of cards so 100% of
        #       card by card interactions are plausibly modeled by it
        interactions = self.interactions(self.noisy_decks)
        # project the deck to a lower dimensional represnetation
        self.latent_rep = self.encoder(interactions)
        # project the latent representation to a potential output
        reconstruction = self.decoder(self.latent_rep)
        basics = self.add_basics_to_deck(reconstruction)
        return tf.concat([basics, reconstruction * self.noisy_decks], axis=1)

    def round_to_deck(self, reconstruction):
        # this is a little trick to return integer values by rounding
        #    however, treating the rounding operation as a constant 
        #    during differentiation. This lets the loss function target
        #    actual decks (with cards as integers) while still enabling backprop
        # the important part is this allows us to add priors to the loss function
        #    such as "has 15-18 lands" . . . maybe we can do this without rounding
        #    but I want to explore with rounding at some point so writing this function
        return reconstruction + tf.stop_gradient(
            tf.math.round(reconstruction) - reconstruction
        )

    def compare(self, deck, card_df, sort_by=['cmc','type_line']):
        deck = np.expand_dims(deck, 0)
        model_output = self.__call__(deck)
        built = tf.squeeze(self.round_to_deck(model_output))
        fake_sb = tf.concat([tf.zeros(5),tf.squeeze(self.noisy_decks - deck[:,5:])],axis=0)
        deck = tf.squeeze(deck)
        df = pd.DataFrame(columns=["name","real deck","fake sideboard","predicted deck"])
        card_df = card_df.sort_values(by=sort_by)
        for card_idx, card_name in card_df[['idx','name']].to_numpy():
            row_dict = {col:None for col in df.columns}
            row_dict['name'] = card_name
            sb_count = fake_sb[card_idx]
            n_found = 0
            if sb_count > 0:
                row_dict['fake sideboard'] = sb_count.numpy()
                n_found += 1
            deck_count = deck[card_idx]
            if deck_count > 0:
                row_dict['real deck'] = deck_count.numpy()
                n_found += 1
            pred_count = built[card_idx]
            if pred_count > 0:
                row_dict['predicted deck'] = pred_count.numpy()
                n_found += 1
            if n_found > 0:
                df = df.append(row_dict, ignore_index=True)  
        return df   

    def compile(
        self,
        **kwargs
    ):
        for k,v in kwargs.items():
            setattr(self, k, v)

    def loss(self, true, pred):
        return (
            self.reg_coef * self.loss_function(true, pred) + 
            self.constraint_coef * self.priors(true, pred)
        )

    def priors(self, true, pred):
        """
        regularization penalties according to deckbuilding priors:

            so far we have "40 card deck" and "match land count of what human built".
        """
        card_count = tf.math.square(40 - tf.reduce_sum(pred, axis=1))
        true_lands = tf.cast(tf.reduce_sum(tf.gather(true, self.land_idxs, axis=1), axis=1),dtype=tf.float32)
        pred_lands = tf.reduce_sum(tf.gather(pred, self.land_idxs, axis=1), axis=1)
        land_count = tf.math.square(true_lands - pred_lands)
        return card_count + land_count

    def fake_sideboard(self, decks):
        """
        inject noise into the decks by adding a fake sideboard randomly sampled
        """
        p = (45-23)/self.n_cards
        first_sample = tf.cast(
            tf.random.uniform(decks.shape) < (p/3),
            dtype=tf.float32
        )
        second_sample = tf.cast(
            tf.random.uniform(decks.shape) < (p/3),
            dtype=tf.float32
        )
        third_sample = tf.cast(
            tf.random.uniform(decks.shape) < (p/3),
            dtype=tf.float32
        )
        sideboard = first_sample + second_sample + third_sample
        return decks + sideboard



