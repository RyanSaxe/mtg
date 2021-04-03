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
        self.cards = seventeenlands.get_card_rating_data(
            self.expansion, join=True
        )
        if isinstance(file_or_df, pd.DataFrame):
            df = file_or_df
        else:
            assert file_or_df.endswith('.csv')
            df = pd.read_csv(file_or_df)
        self.df = self._preprocess(df)

        deck_cols = [x for x in df.columns if x.startswith("deck_")]
        card_names = [x.split("_",1)[-1] for x in deck_cols]
        self.id_to_name = {i:card_name for i,card_name in enumerate(card_names)}
        self.name_to_id = {name:idx for idx,name in self.id_to_name.items()}

    def _preprocess(self, df):
        df = seventeenlands.clean_bo1_games(
            df,
            self.cards,
            drop_cols=['expansion','event_type','game_number']
        )
        df = seventeenlands.add_archetypes(df)
        return df





