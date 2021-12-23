from tensorflow.keras.utils import Sequence
import numpy as np
import pandas as pd
import tensorflow as tf
from mtg.ml.utils import importance_weighting
import gc

class MTGDataGenerator(Sequence):
    def __init__(
        self,
        data,
        cards,
        card_col_prefixes,
        batch_size=32,
        shuffle=True,
        to_fit=True,
        exclude_basics=True,
        store_basics=False,
    ):
        self.cards = cards.sort_values(by='idx', ascending=True)
        self.card_col_prefixes = card_col_prefixes
        self.exclude_basics = exclude_basics
        self.store_basics = store_basics
        if self.exclude_basics:
            self.cards = self.cards.iloc[5:,:]
            self.cards['idx'] = self.cards['idx'] - 5
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.n_cards = self.cards.shape[0]
        self.generate_global_data(data)
        self.size = data.shape[0]
        #generate initial indices for batching the data
        self.reset_indices()

    def __len__(self):
        """
        return: number of batches per epoch
        """
        return self.size // self.batch_size

    def reset_indices(self):
        self.indices = np.arange(self.size)
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def on_epoch_end(self):
        """
        Update indices after each epoch
        """
        self.reset_indices()
        gc.collect()

    def card_name_to_idx(self, card_name, exclude_basics=True):
        return self.cards[self.cards['name'] == card_name]['idx'].iloc[0]
    def card_idx_to_name(self, card_idx, exclude_basics=True):
        return self.cards[self.cards['idx'] == card_idx]['name'].iloc[0]

    def generate_global_data(self, data):
        self.all_cards = [col.split("_",1)[-1] for col in data.columns if col.startswith(self.card_col_prefixes[0])]
        basics = ["plains","island","swamp","mountain","forest"]
        if self.exclude_basics:
            exclude_cards = basics
        else:
            exclude_cards = []
        for prefix in self.card_col_prefixes:
            cols = [col for col in data.columns if col.startswith(prefix + "_") and not any([x in col for x in exclude_cards])]
            setattr(self, prefix, data[cols].values)
            if self.store_basics:
                basic_cols = [col for col in data.columns if any([prefix + "_" + x == col for x in basics])]
                setattr(self, prefix + "_basics", data[basic_cols].values)
        if 'ml_weights' in data.columns:
            self.weights = data['ml_weights'].values
        else:
            self.weights = None

    def __getitem__(self, batch_number):
        """
        Generates a data mini-batch
        param batch_number: which batch to generate  
        return: X and y when fitting. X only when predicting
        """
        indices = self.indices[
            batch_number * self.batch_size:(batch_number + 1) * self.batch_size
        ]
        X,y,weights = self.generate_data(indices)
        
        if self.to_fit:
            return X,y,weights
        else:
            return X

    def generate_data(self, indices):
        raise NotImplementedError

class DraftGenerator(MTGDataGenerator):
    def __init__(
        self,
        data,
        cards,
        batch_size=32,
        shuffle=True,
        to_fit=True,
        exclude_basics=True,
        store_basics=False,
    ):
        super().__init__(
            data,
            cards,
            card_col_prefixes=['pack_card'],
            batch_size=batch_size,
            shuffle=shuffle,
            to_fit=to_fit,
            exclude_basics=exclude_basics,
            store_basics=store_basics,
        )
        #overwrite the size to make sure we always sample full drafts
        self.size = len(self.draft_ids)
        self.reset_indices()

    def generate_global_data(self, data):
        self.draft_ids = data['draft_id'].unique()
        self.t = data['position'].max() + 1
        data = data.set_index(["draft_id","position"])
        self.all_cards = [col.split("_",1)[-1] for col in data.columns if col.startswith(self.card_col_prefixes[0])]
        basics = ["plains","island","swamp","mountain","forest"]
        if self.exclude_basics:
            exclude_cards = basics
        else:
            exclude_cards = []
        for prefix in self.card_col_prefixes:
            cols = [col for col in data.columns if col.startswith(prefix + "_") and not any([x in col for x in exclude_cards])]
            setattr(self, prefix, data[cols])
            if self.store_basics:
                basic_cols = [col for col in data.columns if any([prefix + "_" + x == col for x in basics])]
                setattr(self, prefix + "_basics", data[basic_cols])
        if 'ml_weights' in data.columns:
            self.weights = data['ml_weights']
        else:
            self.weights = None
        name_to_idx_mapping = {k.split("//")[0].strip().lower():v for k,v in self.cards.set_index('name')['idx'].to_dict().items()}
        self.pick = data['pick'].apply(lambda x: name_to_idx_mapping[x])
        self.shifted_pick = self.pick.groupby(level=0).shift(1).fillna(self.n_cards)
        self.position = data['pack_number'] * (data['pick_number'].max() + 1) + data['pick_number']

    def generate_data(self, indices):
        draft_ids = self.draft_ids[indices]
        packs = self.pack_card.loc[draft_ids].values.reshape(len(indices), self.t, len(self.pack_card.columns))
        #pools = self.pool.loc[draft_ids].values.reshape(len(indices), self.t, len(self.pack_card.columns))
        picks = self.pick.loc[draft_ids].values.reshape(len(indices), self.t)
        shifted_picks = self.shifted_pick.loc[draft_ids].values.reshape(len(indices), self.t)
        positions = self.position.loc[draft_ids].values.reshape(len(indices), self.t)
        #draft_info = np.concatenate([packs, pools], axis=-1)
        if self.weights is not None:
            #comment below is if weights sum to 1 for each draft rather than for each batch
            #weights = (self.weights.loc[draft_ids]/self.weights.loc[draft_ids].groupby(level=0).sum()).values.reshape(len(indices), self.t)
            weights = self.weights.loc[draft_ids]/self.weights.loc[draft_ids].sum()
        else:
            weights = None
        # convert to tensor needed for tf.function
        packs = tf.convert_to_tensor(packs.astype(np.float32), dtype=tf.float32)
        positions = tf.convert_to_tensor(positions.astype(np.int32), dtype=tf.int32)
        picks = tf.convert_to_tensor(picks.astype(np.float32), dtype=tf.int32)
        shifted_picks = tf.convert_to_tensor(shifted_picks.astype(np.float32), dtype=tf.int32)
        return (packs, shifted_picks, positions), picks, weights

class DeckGenerator(MTGDataGenerator):
    def __init__(
        self,
        data,
        cards,
        batch_size=32,
        shuffle=True,
        to_fit=True,
        exclude_basics=True,
        store_basics=False,
        pos_neg_sample=False,
    ):
        super().__init__(
            data,
            cards,
            card_col_prefixes=['deck','sideboard'],
            batch_size=batch_size,
            shuffle=shuffle,
            to_fit=to_fit,
            exclude_basics=exclude_basics,
            store_basics=store_basics,
        )
        self.pos_neg_sample = pos_neg_sample

    def generate_data(self, indices):
        decks = self.deck[indices,:]
        sideboards = self.sideboard[indices,:]
        basics = self.deck_basics[indices,:]
        masked_decks, masked_basics = self.create_masked_objects(decks, basics)
        cards_to_add = decks - masked_decks
        basics_to_add = basics - masked_basics
        modified_sideboards = sideboards + cards_to_add
        if self.weights is not None:
            weights = self.weights[indices]/self.weights[indices].sum()
        else:
            weights = None
        if self.pos_neg_sample:
            anchor, pos, neg = self.sample_card_pairs(decks, sideboards)
            return (modified_sideboards, masked_decks, anchor, pos, neg), (basics_to_add, cards_to_add), weights
        return (modified_sideboards, masked_decks), (basics_to_add, cards_to_add), weights

    def create_masked_objects(self, decks, basics):
        deck_idxs = np.where(decks > 0)
        basic_idxs = np.where(basics > 0)

    def get_vectorized_sample(self, mtx):
        probabilities = mtx/mtx.sum(1, keepdims=True)
        cumulative_dist = probabilities.cumsum(axis=1)
        random_bin = np.random.rand(len(cumulative_dist), 1)
        return (random_bin < cumulative_dist).argmax(axis=1)

    def sample_card_pairs(self, decks, sideboards):
        anchors = self.get_vectorized_sample(decks)

        # never sample the same card as the anchor as the positive or negative axample
        decks_without_anchors = decks.copy()
        decks_without_anchors[np.arange(decks.shape[0]),anchors] = 0
        sideboards_without_anchors = sideboards.copy()
        sideboards_without_anchors[np.arange(decks.shape[0]),anchors] = 0

        positive_samples = self.get_vectorized_sample(decks_without_anchors)
        negative_samples = self.get_vectorized_sample(sideboards_without_anchors)

        return anchors, positive_samples, negative_samples

def create_train_and_val_gens(
    data,
    cards,
    id_col=None,
    train_p=1.0, 
    weights=True, 
    train_batch_size=32, 
    shuffle=True, 
    to_fit=True, 
    exclude_basics=True,
    generator=MTGDataGenerator,
    include_val=True,
):
    if weights:
        data['ml_weights'] = importance_weighting(data)
    if train_p < 1.0:
        if id_col is None:
            idxs = np.arange(data.shape[0])
            train_idxs = np.random.choice(idxs,int(len(idxs) * train_p),replace=False)
            test_idxs = np.asarray(list(set(idxs.flatten()) - set(train_idxs.flatten())))
            train_data = data[train_idxs,:]
            test_data = data[test_idxs,:]
        else:
            idxs = data[id_col].unique()
            train_idxs = np.random.choice(idxs,int(len(idxs) * train_p),replace=False)
            train_data = data[data[id_col].isin(train_idxs)]
            test_data = data[~data[id_col].isin(train_idxs)]
        n_train = int(len(idxs) * train_p)
        n_test = len(idxs) - n_train
    else:
        train_data = data
        test_data = None
    train_gen = generator(
        train_data,
        cards.copy(),
        batch_size = train_batch_size,
        shuffle = shuffle,
        to_fit = to_fit,
        exclude_basics = exclude_basics
    )
    if test_data is not None and include_val:
        n_train_batches = len(train_gen)
        val_batch_size = n_test // n_train_batches
        val_gen = generator(
            test_data,
            cards.copy(),
            batch_size = val_batch_size,
            shuffle = shuffle,
            to_fit = to_fit,
            exclude_basics = exclude_basics
        )
    else:
        val_gen = None
    return train_gen, val_gen