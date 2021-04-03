import tensorflow as tf
from mtg.ml import nn

class DeckBuilder(tf.Module):
    def __init__(self, n_cards, latent_dim, name=None):
        super().__init__(name=name)
        self.n_cards = n_cards
        #probability of random sampling a card similar to that of a SB slot
        self.encoder = nn.MLP(
            in_dim=n_cards,
            start_dim=128,
            out_dim=16,
            n_h_layers=2,
            dropout=0.2,
            name="encoder",
            noise=0.0,
            start_act=tf.nn.relu,
            middle_act=tf.nn.relu,
            out_act=tf.nn.relu,
            style="bottleneck"
        )
        self.decoder = nn.MLP(
            in_dim=16,
            start_dim=32,
            out_dim=n_cards,
            n_h_layers=2,
            dropout=0.2,
            name="decoder",
            noise=0.0,
            start_act=tf.nn.relu,
            middle_act=tf.nn.relu,
            out_act=tf.nn.relu,
            style="reverse_bottleneck"
        )
        self.interactions = nn.Dense(n_cards, n_cards, activation=tf.nn.relu)

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
        return reconstruction

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
        reminder to build in regularization to enforce deckbuilding priors like

        1. matchin land counts
        2. enough creatures
        3. relatively good curve
        """
        return 0

    def fake_sideboard(self, decks):
        """
        inject noise into the decks by adding a fake sideboard randomly sampled
        """
        p = 19/self.n_cards
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



