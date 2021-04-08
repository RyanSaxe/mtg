import numpy as np
import mtg.preprocess.seventeenlands as seventeenlands
from mtg.obj.cards import CardSet
import pandas as pd

class Games:
    def __init__(
        self,
        file_or_df,
        expansion,
    ):
        self.expansion = expansion
        # query for scryfall to get cards in the set that are in boosters
        self.set_q = [f'set={self.expansion}','is:booster']
        # self.cards = seventeenlands.get_card_rating_data(
        #     self.expansion, join=True
        # )
        self.cards = CardSet(self.set_q).to_dataframe()
        if isinstance(file_or_df, pd.DataFrame):
            df = file_or_df
        else:
            assert file_or_df.endswith('.csv')
            df = pd.read_csv(file_or_df)
        self.df = self._preprocess(df)

        basics = ["plains","island","swamp","mountain","forest"]
        self.card_names = [x for x in self.cards['name'].unique()]
        for basic in basics:
            self.card_names.remove(basic)
        self.card_names = basics + self.card_names
        self.id_to_name = {i:card_name for i,card_name in enumerate(self.card_names)}
        self.name_to_id = {name:idx for idx,name in self.id_to_name.items()}
        self.cards['idx'] = self.cards['name'].apply(lambda x: self.name_to_id[x])
        card_col_prefixes = ['deck','opening_hand','drawn','sideboard']
        #initialize columns to start with the non-card columns
        column_order = [c for c in self.df.columns if not any([c.startswith(prefix) for prefix in card_col_prefixes])]
        for prefix in card_col_prefixes:
            prefix_columns = [prefix + "_" + name for name in self.card_names]
            setattr(self, prefix + "_cols", prefix_columns)
            column_order += prefix_columns
        #reorder dataframe to abide by new column ordering
        #   this is just so self.df[self.deck_cols].to_numpy() 
        #   yields a comparable matrix to self.df[self.sideboard_cols].to_numpy() 
        self.df = self.df[column_order] 

    def importance_weighting(self,df=None,minim=0.1,maxim=1.0):
        if df is None:
            df = self.df
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
                x.split("-")[0].strip().lower(),
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
        return scaled_win_rate * np.clip(df['won'],a_min=0.5,a_max=1.0) * 0.9 ** n_weeks 

    def get_decks_for_ml(self, train_p=0.9):
        #get each unique decks last build
        d = {
            column: 'last' for column in self.df.columns if column not in ["opp_colors"]
        }
        d.update({
                "won":"mean",
                "on_play":"mean",
                "num_mulligans":"mean",
                "opp_num_mulligans": "mean",
                "num_turns": "mean",
        })
        df = self.df.groupby('draft_id').agg(d)
        decks = df[self.deck_cols].to_numpy(dtype=np.float32)
        sideboards = df[self.sideboard_cols].to_numpy(dtype=np.float32)
        # note that pool has basics but shouldn't. Im choosing not
        # to zero them out here and instead do it on modeling side
        pools = decks + sideboards
        #convert decks to be 0-1 for specifically non basics
        decks[:,5:] = np.divide(
            decks[:,5:],
            pools[:,5:],
            out=np.zeros_like(decks[:,5:]),
            where=pools[:,5:]!=0,
        )
        weights = self.importance_weighting(df).to_numpy(dtype=np.float32)
        idxs = np.arange(len(df))
        train_idxs = np.random.choice(idxs,int(len(idxs) * train_p),replace=False)
        test_idxs = np.asarray(list(set(idxs.flatten()) - set(train_idxs.flatten())))
        train_data = (pools[train_idxs,:],decks[train_idxs,:], weights[train_idxs])
        test_data = (pools[test_idxs,:],decks[test_idxs,:])
        return train_data, test_data

    def _preprocess(self, df):
        df = seventeenlands.clean_bo1_games(
            df,
            self.cards,
            drop_cols=['expansion','event_type','game_number'],
            rename_cols={'draft_time':'date'}
        )
        df['date'] = pd.to_datetime(df['date'])
        df = seventeenlands.add_archetypes(df)
        return df





