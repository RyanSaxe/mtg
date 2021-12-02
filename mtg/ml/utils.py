from mtg.utils.display import print_deck
import numpy as np
import tensorflow as tf
import pickle
import pathlib
import os

def importance_weighting(df,minim=0.1,maxim=1.0):
    rank_to_score = {
        'bronze':0.01,
        'silver':0.1,
        'gold':0.25,
        'platinum':0.5,
        'diamond':0.75,
        'mythic':1.0
    }
    #decrease exponentiation by larger amounts for higher
    # ranks such that rank and win-rate matter together
    rank_addition = df['rank'].apply(
        lambda x: rank_to_score.get(
            x,
            0.5
        )
    )
    scaled_win_rate = np.clip(
        df['user_win_rate_bucket'] ** (2 - rank_addition),
        a_min=minim,
        a_max=maxim,
    )
    
    last = df['date'].max()
    # increase importance factor for recent data points according to number of weeks from most recent data point
    n_weeks = df['date'].apply(lambda x: (last - x).days // 7)
    #lower the value of pxp11 + 
    if "position" in df.columns:
        n_picks_per_pack = (df['position'].max() + 1)/3
        position_scale = df['position'].apply(lambda x: 1.0 if x % n_picks_per_pack <= 9 else 0.5)
    else:
        position_scale = 1.0
    return position_scale * scaled_win_rate * np.clip(df['won'],a_min=0.5,a_max=1.0) * 0.9 ** n_weeks 

def get_decks_for_ml(df, train_p=0.9):
    #get each unique decks last build
    d = {
        column: 'last' for column in df.columns if column not in ["opp_colors"]
    }
    d.update({
            "won":"mean",
            "on_play":"mean",
            "num_mulligans":"mean",
            "opp_num_mulligans": "mean",
            "num_turns": "mean",
    })
    df = df.groupby('draft_id').agg(d)
    decks = df[[c for c in df.columns if c.startswith("deck_")]].to_numpy(dtype=np.float32)
    sideboards = df[[c for c in df.columns if c.startswith("sideboard_")]].to_numpy(dtype=np.float32)
    # note that pool has basics but shouldn't. Im choosing not
    # to zero them out here and instead do it on modeling side
    pools = decks + sideboards
    # #convert decks to be 0-1 for specifically non basics
    #  currently not used because we multiply output in 0-1 by the pool to address this
    # decks[:,5:] = np.divide(
    #     decks[:,5:],
    #     pools[:,5:],
    #     out=np.zeros_like(decks[:,5:]),
    #     where=pools[:,5:]!=0,
    # )
    weights = importance_weighting(df).to_numpy(dtype=np.float32)
    idxs = np.arange(len(df))
    if train_p < 1.0:
        train_idxs = np.random.choice(idxs,int(len(idxs) * train_p),replace=False)
        test_idxs = np.asarray(list(set(idxs.flatten()) - set(train_idxs.flatten())))
        train_data = (pools[train_idxs,:],decks[train_idxs,:], weights[train_idxs])
        test_data = (pools[test_idxs,:],decks[test_idxs,:], weights[test_idxs])
    else:
        train_idxs = idxs
        test_idxs = []
        train_data = (pools,decks, weights)
        test_data = None
        test_idxs = []
    return train_data, test_data, {'train':train_idxs, 'test':test_idxs, 'map':dict(zip(idxs,df.index.tolist()))}

def load_model(location, extra_pickle='attrs.pkl'):
    model_loc = os.path.join(location,"model")
    data_loc = os.path.join(location,extra_pickle)
    model = tf.saved_model.load(model_loc)
    try:
        with open(data_loc,'rb') as f:
            cards = pickle.load(f)
        return (model,cards)
    except:
        return model

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

def build_from_output(pred, pool, cards, show=True, verbose=False, calibrate_mana=True):
    deck = pred.numpy()
    built_deck = np.zeros_like(deck)
    excess = np.zeros_like(deck)
    # ensure we go through basics first otherwise we get very low lands
    # for good pools
    order_to_add = np.argsort(deck * pool)[::-1]
    non_basic_idx = np.where(order_to_add >= 5)
    order_to_add = order_to_add[non_basic_idx]
    order_to_add = np.concatenate(
        [np.arange(5),order_to_add]
    )
    deck_count = 0
    for card_idx in order_to_add:
        if deck_count >= 40:
            break
        allowed_to_add = 40 - deck_count
        val = pred[card_idx].numpy()
        n_to_add = np.round(val)
        remainder = np.clip(val - n_to_add, a_min=0, a_max=1)
        pool_count = pool[card_idx]
        n_to_add = np.clip(n_to_add, 0, allowed_to_add)
        # if pool_count > 0:
        #   print(cards[cards['idx'] == card_idx]['name'], val, pool_count)
        if pool_count - n_to_add > 0:
          single_card_val = val/pool_count
        else:
          single_card_val = 0
        # zero if rounded up, excess if rounded down
        if verbose and val != 0:
            print(
                cards[cards['idx'] == card_idx].iloc[0]['name'],
                val,
                single_card_val,
            )
        excess[card_idx] = single_card_val
        built_deck[card_idx] = n_to_add
        deck_count = built_deck.sum()      
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
                if card_idx in [0,1,2,3,4]:
                  n_left = 1
                else:
                  n_left = pool[card_idx] - built_deck[card_idx]
                allowed_to_add = 40 - built_deck.sum()
                n_to_add = np.clip(n_left,0,allowed_to_add)
                built_deck[card_idx] += n_to_add
                if verbose:
                    print(
                        cards[cards['idx'] == card_idx].iloc[0]['name'],
                        n_to_add,
                        excess[card_idx],
                    )
    #now we have a deck, but sometimes the rounding of basics gets odd, here is an example:
    #a UR deck splashing W has a basic breakdown of:
    #    Plains: 0.4
    #    Island: 7.5
    #    Swamp: 0
    #    Mountain: 7.5
    #    Forest: 0
    #    Evolving Wilds: 1.8
    # what happens is we get a manabase of 2 Evolving Wilds, 7/8 Island, and 7/8 Mountain, with zero Plains
    # so the following code checks for this case and then will add a plains over either a mountain or island
    # it will also remove that plains if there were no need for it too
    #eventually move this to all lands instead of basics
    color_to_idx = cards[cards['idx'] < 5].set_index('idx')['produced_mana'].apply(
        lambda x: x[0]
    ).reset_index().set_index('produced_mana').to_dict()['idx']

    if calibrate_mana:
        pip_count = {c:0 for c in list('WUBRG')}
        # don't count a green mana dork that produces G as a G source, but if it produces other colors, it can count as a source
        basic_adds_extra_sources = {c:0 for c in list('WUBRG')}
        # cards that dont produce mana, but search for basics should also affect the sources we are counting
        special_case_cards = cards[cards['basic_land_search']]['name'].tolist()
        splash_produces_count = {c:0 for c in list('WUBRG')}
        for card_idx,count in enumerate(built_deck):
            if count == 0:
                continue
            card = cards[cards['idx'] == card_idx]
            basic_special_case_flag = (card['name'].isin(special_case_cards)).iloc[0]
            mc = card['mana_cost'].iloc[0]
            splash_produce = list(set(card['produced_mana'].iloc[0]) - {'C'} - set(card['colors'].iloc[0])) if not card['produced_mana'].isna().iloc[0] else []
            for color in pip_count.keys():
                card_color_count = mc.count(color)
                pip_count[color] += count * card_color_count
                if basic_special_case_flag:
                    basic_count = built_deck[color_to_idx[color]]
                    if basic_count == 0:
                        basic_adds_extra_sources[color] += count
                    else:
                        #do not count a green ramp spell as a green source
                        if card_color_count == 0:
                            splash_produces_count[color] += count
                elif color in splash_produce:
                    splash_produces_count[color] += count

        min_produces_map = {
            0: 0,
            1: 3,
            2: 4,
            3: 4,
            4: 5,
            5: 5,
            6: 6,
            7: 6
        }

        
        add_basics_dict = {c:0 for c in list('WUBRG')}
        
        cut_basics_dict = {c:0 for c in list('WUBRG')}

        basic_cut_limit = {c:0 for c in list('WUBRG')}
        
        for color in list('WUBRG'):
            pips = pip_count[color]
            if pips == 0:
                #ensure we cut basics that dont do anything
                idx_for_basic = color_to_idx[color]
                basic_count_in_deck = built_deck[idx_for_basic]
                cut_basics_dict[color] += basic_count_in_deck
            if pips > 0 and basic_adds_extra_sources[color] > 0:
                min_add = 1
            else:
                min_add = 0
            mana_req = min_produces_map.get(pips, 7)
            produces = splash_produces_count[color]
            produces_diff = produces - mana_req
            if produces_diff < 0:
                add_basics_dict[color] += abs(produces_diff)
            else:
                basic_cut_limit[color] = max(produces_diff,0)
            if add_basics_dict[color] < min_add:
                add_basics_dict[color] = min_add

        #now ad_basics_dict is the number of basics per color that needs to be added
        # the following logic determines what basics need to be cut
        # get number of basics in the deck, but if that basic is required to be added, don't allow it to be cut
        basics_that_can_be_cut = {c:min(built_deck[color_to_idx[c]],basic_cut_limit[c]) if n == 0 else 0 for c,n in add_basics_dict.items()}
        total_basics_to_cut = sum([x for x in add_basics_dict.values()])
        if total_basics_to_cut > sum([x for x in basics_that_can_be_cut.values()]):
            if verbose:
                print('This manabase is not salvageable')
        cur_color_idx = 0
        colors_to_add = [c for c,n in add_basics_dict.items() if n > 0]
        check_bug = 0
        colors = list('WUBRG')
        while sum([x for x in add_basics_dict.values()]) > 0 or sum([x for x in cut_basics_dict.values()]) > 0:
            if len(colors_to_add) == 0:
                if sum([x for x in add_basics_dict.values()]) > 0:
                    colors_to_add = [c for c,n in add_basics_dict.items() if n > 0]
                else:
                    if sum([x for x in cut_basics_dict.values()]) <= 0:
                        #nothing to add or cut!
                        break
                    else:
                        colors_to_add = [c for c,n in basics_that_can_be_cut.items() if n > 0]
            if len(colors_to_add) == 0:
                if verbose:
                    print('Nothing else is allowed to be cut, bad manabase')
                break 
            c = colors[cur_color_idx % 5]
            #this is the actual idx in the deck built, not the fake one used to cycle through colors
            idx = color_to_idx[c]
            ad_c = colors_to_add[0]
            colors_to_add = colors_to_add[1:]
            ad_idx = color_to_idx[ad_c]
            if sum([x for x in cut_basics_dict.values()]) > 0:
                if cut_basics_dict[c] > 0:
                    built_deck[idx] -= 1
                    built_deck[ad_idx] += 1
                    basics_that_can_be_cut[c] -= 1
                    cut_basics_dict[c] -= 1
                    add_basics_dict[ad_c] -= 1
            else:
                if basics_that_can_be_cut[c] > 0:
                    built_deck[idx] -= 1
                    built_deck[ad_idx] += 1
                    basics_that_can_be_cut[c] -= 1
                    cut_basics_dict[c] -= 1
                    add_basics_dict[ad_c] -= 1

            cur_color_idx += 1
            check_bug += 1
            if check_bug > 100:
                print('BUG')
                break


    #for now cut evenly, where if there's a tie we cut from the biggest
    
    if show:
        print_deck(built_deck.astype(int), cards, sort_by=['cmc','type_line'])
        print('\nSIDEBOARD\n')
        # zero out basics so that they are excluded from sideboard
        pool[:5] = 0
        built_deck[:5] = 0
        print_deck((pool - built_deck).astype(int), cards)
    return built_deck