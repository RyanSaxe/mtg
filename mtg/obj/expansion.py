

from mtg.obj.cards import CardSet
import pandas as pd
from mtg.preprocess.seventeenlands import clean_bo1_games

class Expansion:
    def __init__(self, expansion, bo1=None, bo3=None, quick=None, draft=None, replay=None):
        self.expansion = expansion
        self.cards = CardSet([f'set={self.expansion}','is:booster']).to_dataframe()
        self.clean_card_df()
        self.bo1 = self.process_bo1(bo1)
        self.bo3 = self.process_bo3(bo3)
        self.quick = self.process_quick(quick)
        self.draft = self.process_draft(draft)
        self.replay = self.process_replay(replay)

    def generic_process(self,file_or_df):
        if isinstance(file_or_df,str):
            df = pd.read_csv(file_or_df)
        else:
            df = file_or_df
        return df
    
    def process_bo1(self, file_or_df):
        if file_or_df is None:
            return None
        df = self.generic_process(file_or_df)
        clean_bo1_games(
            df,
            self.cards,
            drop_cols=['expansion','event_type','game_number'],
            rename_cols={'draft_time':'date'}
        )
        df['date'] = pd.to_datetime(df['date'])
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
        df = df[column_order]
        return df

    def process_bo3(self, file_or_df):
        if file_or_df is None:
            return None
        df = self.generic_process(file_or_df)
        return df

    def process_quick(self, file_or_df):
        if file_or_df is None:
            return None
        df = self.generic_process(file_or_df)
        return df

    def process_draft(self, file_or_df):
        if file_or_df is None:
            return None
        df = self.generic_process(file_or_df)
        return df

    def process_replay(self, file_or_df):
        if file_or_df is None:
            return None
        df = self.generic_process(file_or_df)
        return df
    
    def clean_card_df(self):
        raise NotImplementedError

class MID(Expansion):
    def __init__(self, bo1=None, bo3=None, quick=None, draft=None, replay=None):
        super().__init__(expansion='mid', bo1=bo1, bo3=bo3, quick=quick, draft=draft, replay=replay)

    def clean_card_df(self):
        #set it so ramp spells that search for basics are seen as rainbow producers
        # logic to subset by basic implemented where needed
        search_check = lambda x: 'search your library' in x['oracle_text']
        basic_check = lambda x: 'basic land' in x['oracle_text']
        self.cards['basic_land_search'] = self.cards.apply(
            lambda x: search_check(x) and basic_check(x),
            axis=1
        )