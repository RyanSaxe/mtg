

from mtg.obj.cards import CardSet
import pandas as pd
from mtg.preprocess.seventeenlands import clean_bo1_games, get_card_rating_data
from mtg.utils.dataloading_utils import load_data
import numpy as np
import re
import time
import random

class Expansion:
    def __init__(self, expansion, bo1=None, bo3=None, quick=None, draft=None, replay=None, ml_data=True, idx_to_name=None):
        self.expansion = expansion
        self.cards = CardSet([f'set={self.expansion}','is:booster']).to_dataframe()
        self.clean_card_df(idx_to_name)
        self.bo1 = self.process_data(bo1, name="bo1")
        self.bo3 = self.process_data(bo3, name="bo3")
        self.quick = self.process_data(quick, name="quick")
        self.draft = self.process_data(draft, name="draft")
        self.replay = self.process_data(replay, name="replay")
        if ml_data:
            self.card_data_for_ML = self.get_card_data_for_ML()
        else:
            self.card_data_for_ML = None

    @property
    def types(self):
        return ['instant','sorcery','creature','planeswalker','artifact','enchantment','land']
        
    def process_data(self, file_or_df, name=None):
        if isinstance(file_or_df,str):
            if name is None:
                df = pd.read_csv(file_or_df)
            else:
                df = load_data(file_or_df, self.cards.copy(), name=name)
        else:
            df = file_or_df
        return df
    
    def clean_card_df(self, idx_to_name=None):
        if idx_to_name is not None:
            if "plains" not in idx_to_name.keys():
                idx_to_name = {k + 5:v for k,v in idx_to_name.items()}
                basics = ["plains","island","swamp","mountain","forest"]
                for basic_idx, basic in basics:
                    idx_to_name[basic_idx] = basic
            name_to_idx = {v:k for k,v in idx_to_name.items()}
            self.cards['idx'] = self.cards['name'].apply(name_to_idx)
        #set it so ramp spells that search for basics are seen as rainbow producers
        # logic to subset by basic implemented where needed
        search_check = lambda x: 'search your library' in x['oracle_text']
        basic_check = lambda x: 'basic land' in x['oracle_text']
        self.cards['basic_land_search'] = self.cards.apply(
            lambda x: search_check(x) and basic_check(x),
            axis=1
        )
        self.cards['flip'] = self.cards['layout'].apply(lambda x: 0.0 if x == 'normal' else 1.0)

    def get_card_data_for_ML(self, return_df=True):
        ml_data = self.get_card_stats()
        colors = list('WUBRG')
        cards = self.cards.set_index('name').copy()
        cards = cards.replace(to_replace="*", value=-1)
        copy_from_scryfall = ['power', 'toughness', 'basic_land_search', 'flip', 'cmc']
        for column in copy_from_scryfall:
            ml_data[column] = cards[column].astype(float)
        keywords = list(set(cards['keywords'].sum()))
        keyword_df = pd.DataFrame(index=cards.index, columns=keywords).fillna(0)
        for card_idx,keys in cards['keywords'].to_dict().items():
            keyword_df.loc[card_idx, keys] = 1.0
        ml_data = pd.concat([ml_data, keyword_df], axis=1)
        for color in colors:
            ml_data[color + " pips"] = cards['mana_cost'].apply(lambda x: x.count(color))
            ml_data['produces ' + color] = cards['produced_mana'].apply(lambda x: 0.0 if not isinstance(x, list) else int(color in x))
        for cardtype in self.types:
            cardtype = cardtype.lower()
            ml_data[cardtype] = cards['type_line'].str.lower().apply(lambda x: 0.0 if not isinstance(x, str) else int(cardtype in x))
        rarities = cards['rarity'].unique()
        for rarity in rarities:
            ml_data[rarity] = cards['rarity'].apply(lambda x: int(x == rarity))
        ml_data['produces C'] = cards['produced_mana'].apply(lambda x: 0 if not isinstance(x, list) else int('C' in x))
        ml_data.columns = [x.lower() for x in ml_data.columns]
        count_cols = [x for x in ml_data.columns if '_count' in x]
        # 0-1 normalize data representing counts
        ml_data[count_cols] = ml_data[count_cols].apply(lambda x: x/x.max(), axis=0)
        ml_data['idx'] = cards['idx']
        # the way our embeddings work is we always have an embedding that represents the lack of a card. This helps the model
        # represent stuff like generic format information. Hence we make this a one-hot vector that gets used in Draft when
        # the pack is empty, but have that concept "on" for every single card so it can affect the learned representations
        ml_data.loc['bias',:] = 0.0
        ml_data.loc['bias', 'idx'] = cards['idx'].max() + 1
        ml_data['bias'] = 1.0
        ml_data = ml_data.fillna(0).sort_values('idx').reset_index(drop=True)
        ml_data = ml_data.drop('idx', axis=1)
        if return_df:
            return ml_data
        return ml_data.values
    
    def get_card_stats(self):
        all_colors = [
            None,
            'W','U','B','R','G',
            'WU', 'WB', 'WR', 'WG',
            'UB', 'UR', 'UG',
            'BR', 'BG',
            'RG',
            'WUB', 'WUR', 'WUG',
            'WBR', 'WBG',
            'WRG',
            'UBR', 'UBG',
            'URG',
            'BRG',
            'WUBR', 'WUBG', 'WURG',
            'WBRG',
            'UBRG',
            'WUBRG'
        ]
        card_df = pd.DataFrame()
        for colors in all_colors:
            time.sleep(1)
            card_data_df = get_card_rating_data("VOW", colors=colors)
            extension = "" if colors is None else "_" + colors
            card_data_df.columns = [col + extension for col in card_data_df.columns]
            card_df = pd.concat([card_df, card_data_df], axis=1).fillna(0.0)
        return card_df

    def get_bo1_decks(self):
        d = {
            column: 'last' for column in self.bo1.columns if column not in ["opp_colors"]
        }
        d.update({
                "won":"sum",
                "on_play":"mean",
                "num_mulligans":"mean",
                "opp_num_mulligans": "mean",
                "num_turns": "mean",
        })
        decks = self.bo1.groupby('draft_id').agg(d)
        deck_cols = [x for x in decks.columns if x.startswith("deck_")]
        print('num decks total:',decks.shape[0])
        decks = decks[decks[deck_cols].sum(1) == 40]
        print('num decks 40 cards:',decks.shape[0])
        return decks

    def generate_pack(self, exclude_basics=True):
        """
        generate random pack of MTG cards
        """
        cards = self.cards.copy()
        if exclude_basics:
            cards = cards[cards['idx'] >= 5].copy()
            cards['idx'] = cards['idx'] - 5
        p_r = 7/8
        p_m = 1/8
        if np.random.random() < 1/8:
            rare = random.sample(cards[cards['rarity'] == 'mythic']['idx'].tolist(),1)
        else:
            rare = random.sample(cards[cards['rarity'] == 'rare']['idx'].tolist(),1)
        uncommons = random.sample(cards[cards['rarity'] == 'uncommon']['idx'].tolist(),3)
        commons = random.sample(cards[cards['rarity'] == 'common']['idx'].tolist(),10)
        idxs = rare + uncommons + commons
        pack = np.zeros(len(cards))
        pack[idxs] = 1
        return pack

class MID(Expansion):
    def __init__(self, bo1=None, bo3=None, quick=None, draft=None, replay=None, ml_data=True):
        super().__init__(expansion='mid', bo1=bo1, bo3=bo3, quick=quick, draft=draft, replay=replay, ml_data=ml_data)

class VOW(Expansion):
    def __init__(self, bo1=None, bo3=None, quick=None, draft=None, replay=None, ml_data=True):
        super().__init__(expansion='vow', bo1=bo1, bo3=bo3, quick=quick, draft=draft, replay=replay, ml_data=ml_data)

    def generate_pack(self, exclude_basics=True, name_to_idx=None, return_names=False):
        """
        generate random pack of MTG cards
        """
        cards = self.cards.copy()
        if exclude_basics:
            cards = cards[cards['idx'] >= 5].copy()
            cards['idx'] = cards['idx'] - 5
        if name_to_idx is None:
            name_to_idx = cards.set_index('name')['idx'].to_dict()
        uncommon_or_rare_flip = random.sample(
            cards[
                (cards['rarity'].isin(['mythic','rare','uncommon'])) &
                (cards['flip'] == 1)
            ]['name'].tolist(),
            1
        )[0]
        common_flip = random.sample(
            cards[
                (cards['rarity'] == 'common') &
                (cards['flip'] == 1)
            ]['name'].tolist(),
            1
        )[0]
        upper_rarity = cards[cards['name'] == uncommon_or_rare_flip]['rarity'].values[0]
        if upper_rarity == 'uncommon':
            p_r = 7/8
            p_m = 1/8
            if np.random.random() < 1/8:
                rare = random.sample(cards[(cards['rarity'] == 'mythic') & (cards['flip'] == 0)]['name'].tolist(),1)
            else:
                rare = random.sample(cards[(cards['rarity'] == 'rare') & (cards['flip'] == 0)]['name'].tolist(),1)
            uncommons = random.sample(cards[(cards['rarity'] == 'uncommon') & (cards['flip'] == 0)]['name'].tolist(),2) + [uncommon_or_rare_flip]
        else:
            uncommons = random.sample(cards[(cards['rarity'] == 'uncommon') & (cards['flip'] == 0)]['name'].tolist(),3)
            rare = [uncommon_or_rare_flip]
        commons = [common_flip]
        # make sure at least one common of each color
        for color in list('WUBRG'):
            color_common = random.sample(cards[
                (cards['rarity'] == 'common')
                & (cards['flip'] == 0)
                & (cards['mana_cost'].str.contains(color))
                & (~cards['name'].isin(commons))
            ]['name'].tolist(),1)
            commons += color_common
        other_commons = random.sample(cards[((cards['rarity'] == 'common')) & (cards['flip'] == 0) & (~cards['name'].isin(commons))]['name'].tolist(),4)
        commons += other_commons
        names = rare + uncommons + commons
        if return_names:
            return names
        idxs = [name_to_idx[name] for name in names]
        pack = np.zeros(len(cards))
        pack[idxs] = 1
        return pack

    @property
    def types(self):
        types = super().types
        return types + ['human','zombie','wolf','werewolf', 'spirit', 'aura']