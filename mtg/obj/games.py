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
        self.set_q = f'set={self.expansion}'
        # self.cards = seventeenlands.get_card_rating_data(
        #     self.expansion, join=True
        # )
        self.cards = CardSet([self.set_q]).to_dataframe()
        if isinstance(file_or_df, pd.DataFrame):
            df = file_or_df
        else:
            assert file_or_df.endswith('.csv')
            df = pd.read_csv(file_or_df)
        self.df = self._preprocess(df)

        deck_cols = [x for x in self.df.columns if x.startswith("deck_")]
        basics = ["plains","island","swamp","mountain","forest"]
        self.card_names = [x.split("_",1)[-1] for x in deck_cols]
        for basic in basics:
            self.card_names.remove(basic)
        self.card_names = basics + self.card_names
        self.cards = self.cards[self.cards['name'].isin(self.card_names)]
        self.id_to_name = {i:card_name for i,card_name in enumerate(self.card_names)}
        self.name_to_id = {name:idx for idx,name in self.id_to_name.items()}
        self.cards['idx'] = self.cards['name'].apply(lambda x: self.name_to_id[x])

    def get_decks(self):
        df = seventeenlands.isolate_decks(self.df.copy())
        deck_cols = ["deck_" + x for x in self.card_names]
        return df[deck_cols].to_numpy()

    def get_decks_for_ml(self, train_p=0.8):
        df = seventeenlands.isolate_decks(self.df.copy())
        deck_cols = ["deck_" + x for x in self.card_names]
        decks = df[deck_cols].to_numpy()
        idxs = np.arange(len(df))
        train_idxs = np.random.choice(idxs,int(len(idxs) * train_p),replace=False)
        test_idxs = np.asarray(list(set(idxs.flatten()) - set(train_idxs.flatten())))
        train_data = decks[train_idxs,:]
        test_data = decks[test_idxs,:]
        return train_data, test_data

    def _preprocess(self, df):
        df = seventeenlands.clean_bo1_games(
            df,
            self.cards,
            drop_cols=['expansion','event_type','game_number'],
            rename_cols={'draft_time':'date'}
        )
        df = seventeenlands.add_archetypes(df)
        return df





