from mtg.utils.display import print_deck
import numpy as np
import tensorflow as tf
import pickle
import pathlib
import os
def load_model(location):
    model_loc = os.path.join(location,"model")
    data_loc = os.path.join(location,"cards.pkl")
    model = tf.saved_model.load(model_loc)
    with open(data_loc,'rb') as f:
        cards = pickle.load(f)
    return (model,cards)
def text_to_arr(deck,cards):
    id_lookup = cards.set_index('name')
    deck_arr = np.zeros(cards['idx'].max() + 1,dtype=np.float32)
    lines = deck.split("\n")
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue
        amount = int(line[0])
        name = line[2:].lower()
        try:
            idx = id_lookup.loc[name]['idx']
        except:
            idx = cards[cards['name'].apply(lambda x: x.startswith(name))].iloc[0]['idx']
        deck_arr[idx] += amount
    return deck_arr

def predict_from_text(model, pool, cards):
    pool = text_to_arr(pool, cards)
    pred = tf.squeeze(model(np.expand_dims(pool,0)))
    return build_from_output(pred, pool, cards, show=True)

def build_from_output(pred, pool, cards, show=True):
    deck = pred.numpy()
    built_deck = np.zeros_like(deck)
    excess = np.zeros_like(deck)
    order_to_add = np.argsort(deck * pool)[::-1]
    deck_count = 0
    for card_idx in order_to_add:
        if deck_count >= 40:
            break
        allowed_to_add = 40 - deck_count
        val = pred[card_idx]
        n_to_add = np.round(val)
        # zero if rounded up, excess if rounded down
        excess[card_idx] = np.clip(val - n_to_add, a_min=0, a_max=1)
        n_to_add = np.clip(n_to_add, 0, allowed_to_add)
        built_deck[card_idx] = n_to_add
        deck_count += n_to_add      
    if deck_count > 40:
        print('too many cards: THIS SHOULDNT BE POSSIBLE')
    elif deck_count < 40:
        sorted_excess = np.argsort(excess)[::-1]
        for i in range(40 - int(deck_count)):
            card_idx = sorted_excess[i]
            if excess[card_idx] == 0:
                print('not enough cards')
                break
            else:
                built_deck[card_idx] += 1
    if show:
        print_deck(built_deck.astype(int), cards, sort_by=['cmc','type_line'])
        print('\nSIDEBOARD\n')
        # zero out basics so that they are excluded from sideboard
        pool[:5] = 0
        built_deck[:5] = 0
        print_deck((pool - built_deck).astype(int), cards)
    return built_deck