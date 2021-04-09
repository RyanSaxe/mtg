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
        self.interactions = nn.Dense(self.n_cards, self.n_cards, activation=tf.nn.relu)
        self.add_basics_to_deck = nn.Dense(32,5, activation=lambda x: tf.nn.sigmoid(x) * 18.0)

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
        basics = self.add_basics_to_deck(self.latent_rep)
        if training is None:
            built_deck = tf.concat([basics, reconstruction * pools], axis=1)
        else:
            built_deck = tf.concat([basics, reconstruction], axis=1)
        return built_deck

    def compile(
        self,
        cards,
        basic_lambda=1.0,
        built_lambda=1.0,
        cmc_lambda=0.01,
        adv_mana_lambda=0.01,
        optimizer=None,
    ):
        self.optimizer = tf.optimizers.Adam(lr=0.001) if optimizer is None else optimizer

        self.basic_lambda = basic_lambda
        self.built_lambda = built_lambda
        self.built_loss_f = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.basic_loss_f = tf.keras.losses.MSE

        self.cmc_lambda = cmc_lambda
        self.adv_mana_lambda = adv_mana_lambda

        self.set_card_params(cards)


    def compute_total_pips(self, decks):
        return np.multiply(self.pips_mtx,np.expand_dims(decks,-1)).sum(axis=1)

    def compute_total_produces(self, decks):
        return np.multiply(self.produces_mtx,np.expand_dims(decks,-1)).sum(axis=1)

    def set_card_params(self, cards):
        self.cmc_map = cards.sort_values(by='idx')['cmc'].to_numpy(dtype=np.float32)
        self.produces_mtx = self.get_color_producers(cards)
        self.pips_mtx = self.get_color_pips(cards)

    def get_color_pips(self, cards):
        flip_to_mc_map = {
            'esika, god of the tree // the prismatic bridge':'{1}{G}{G}',
            'birgi, god of storytelling // harnfel, horn of bounty':'{2}{R}',
            'cosima, god of the voyage // the omenkeel':'{2}{U}',
            'barkchannel pathway // tidechannel pathway':None,
            "reidane, god of the worthy // valkmira, protector's shield":'{2}{W}',
            'darkbore pathway // slitherbore pathway':None,
            'hengegate pathway // mistgate pathway':None,
            'alrund, god of the cosmos // hakka, whispering raven':'{3}{U}{U}',
            'jorn, god of winter // kaldring, the rimestaff':'{G}{U}{B}',
            'blightstep pathway // searstep pathway':None,
            'valki, god of lies // tibalt, cosmic impostor':'{5}{B}{R}',
            "toralf, god of fury // toralf's hammer": '{2}{R}{R}',
            'kolvori, god of kinship // the ringhart crest':'{2}{G}{G}',
            'egon, god of death // throne of death':'{2}{B}',
            'halvar, god of battle // sword of the realms':'{2}{W}{W}',
            "tergrid, god of fright // tergrid's lantern":'{3}{B}{B}'
        }
        def mc_to_pips(mc):
            if mc is None:
                return np.zeros(5)
            return np.asarray(
                [mc.count(color) for color in list('WUBRG')]
            )
        return np.vstack(
            cards.sort_values(by='idx').apply(
                lambda x: flip_to_mc_map.get(x['name'],x['mana_cost']), axis=1
            ).apply(mc_to_pips).to_numpy()
        )
    def get_color_producers(self, cards):
        produces = CardSet(['set=khm','produces:any','is:booster'])
        color_to_land = {
            'W':0,
            'U':1,
            'B':2,
            'R':3,
            'G':4
        }
        bad_produces = [
            'goldspan dragon',
            'tundra fumarole',
            'the bloodsky massacre',
            'niko defies destiny',
            'open the omenpaths',
            'tyvar kell',
            'kolvori, god of kinship // the ringhart crest',
            'faceless haven',
            'valki, god of lies // tibalt, cosmic impostor',
            'karfell harbinger',
            'old-growth troll',
            'birgi, god of storytelling // harnfel, horn of bounty',
            'tyrite sanctum',
            'colossal plow',
            'smashing success',
            'arni slays the troll',
        ]
        good_produces = [x.name for x in produces.cards if x.name not in bad_produces]
        def make_mana_arr(mana_prod):
            arr = np.zeros(5)
            idxs = [color_to_land[c] for c in mana_prod]
            arr[idxs] = 1
            return arr
        return np.vstack(
            [make_mana_arr(x[1]) if x[0] in good_produces else np.zeros(5) for 
            x in cards.sort_values('idx')[['name','produced_mana']].to_numpy()]
        )
        
    def pip_vs_produce_penalty(self, true, pred):
        pred_pips = self.compute_total_pips(pred)
        true_pips = self.compute_total_pips(true)

        pred_produce = self.compute_total_produces(pred)
        true_produce = self.compute_total_produces(true)

        harmonic_pred = harmonic_mean(pred_pips, pred_produce)
        harmonic_true = harmonic_mean(true_pips, true_produce)
        #hypothesis, we want high produces for high pips
        #            and low produces for low pips
        #additional notes
        #  we want to be base 2 colors generally
        #  we want to learn monotonically increasing relationship to hypothesis

    def loss(self, true, pred, sample_weight=None):
        true_basics,true_built = tf.split(true,[5,280],1)
        pred_basics,pred_built = tf.split(pred,[5,280],1)
        self.basic_loss = self.basic_loss_f(true_basics, pred_basics)
        self.built_loss = self.built_loss_f(true_built, pred_built, sample_weight=sample_weight)
        self.lean_incentive = tf.reduce_sum(
            tf.multiply(pred,tf.expand_dims(self.cmc_map,0)),
            axis=1
        )
        self.mana_reg = self.pip_vs_produce_penalty(true, pred)
        return (
            self.basic_lambda * self.basic_loss + 
            self.built_lambda * self.built_loss +
            self.cmc_lambda * self.lean_incentive + 
            self.adv_mana_lambda * self.mana_reg
        )

    def save(self, cards, location):
        pathlib.Path(location).mkdir(parents=True, exist_ok=True)
        model_loc = os.path.join(location,"model")
        data_loc = os.path.join(location,"cards.pkl")
        tf.saved_model.save(self,model_loc)
        with open(data_loc,'wb') as f:
            pickle.dump(cards,f) 



