import tensorflow as tf
from mtg.ml import nn

class DeckBuilder(tf.Module):
    def __init__(self, n_cards, latent_dim, name=None):
        super().__init__(name=name)
        self.n_cards = n_cards
        #probability of random sampling a card similar to that of a SB slot
        self.encoder = nn.MLP(
            start_dim=128,
            out_dim=16,
            n_h_layers=2,
            dropout=0.2,
            name="encoder",
            noisy=False,
            start_act=tf.nn.relu,
            middle_act=tf.nn.relu,
            out_act=tf.nn.relu,
            style="bottleneck"
        )
        self.decoder = nn.MLP(
            start_dim=32,
            out_dim=n_cards,
            n_h_layers=2,
            dropout=0.2,
            name="decoder",
            noisy=False,
            start_act=tf.nn.relu,
            middle_act=tf.nn.relu,
            out_act=tf.nn.relu,
            style="reverse_bottleneck"
        )
        self.interactions = nn.Dense(n_cards, activation=tf.nn.relu)

    def __call__(self, decks, training=None):
        # noisy_decks is a temporary process until we get SB data
        decks = self.fake_sideboard(decks)
        # first layer is of same dim as number of cards so 100% of
        #       card by card interactions are plausibly modeled by it
        interactions = self.interactions(decks)
        # project the deck to a lower dimensional represnetation
        latent_rep = self.encoder(interactions)
        # project the latent representation to a potential output
        reconstruction = self.decoder(latent_rep)
        # this is a little trick to return integer values by rounding
        #    however, treating the rounding operation as a constant 
        #    during differentiation. This lets the loss function target
        #    actual decks (with cards as integers) while still enabling backprop
        return reconstruction + tf.stop_gradient(
            tf.math.round(reconstruction) - reconstruction
        )

    def priors(self):
        """
        reminder to build in regularization to enforce deckbuilding priors like

        1. matchin land counts
        2. enough creatures
        3. relatively good curve
        """
        pass

    def fake_sideboard(self, decks):
        """
        inject noise into the decks by adding a fake sideboard randomly sampled
        """
        p = 19/self.n_cards
        first_sample = tf.cast(
            tf.random.random(decks.shape) < (p/3),
            dtype=tf.float32
        )
        second_sample = tf.cast(
            tf.random.random(decks.shape) < (p/3),
            dtype=tf.float32
        )
        third_sample = tf.cast(
            tf.random.random(decks.shape) < (p/3),
            dtype=tf.float32
        )
        sideboard = first_sample + second_sample + third_sample
        return decks + sideboard



