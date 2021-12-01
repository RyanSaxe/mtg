

from mtg.obj.cards import CardSet
import pandas as pd
from mtg.preprocess.seventeenlands import clean_bo1_games
from mtg.utils.dataloading_utils import load_data

class Expansion:
    def __init__(self, expansion, bo1=None, bo3=None, quick=None, draft=None, replay=None):
        self.expansion = expansion
        self.cards = CardSet([f'set={self.expansion}','is:booster']).to_dataframe()
        self.clean_card_df()
        self.bo1 = self.process_data(bo1, name="bo1")
        self.bo3 = self.process_data(bo3, name="bo3")
        self.quick = self.process_data(quick, name="quick")
        self.draft = self.process_data(draft, name="draft")
        self.replay = self.process_data(replay, name="replay")

    def process_data(self, file_or_df, name=None):
        if isinstance(file_or_df,str):
            if name is None:
                df = pd.read_csv(file_or_df)
            else:
                df = load_data(file_or_df, self.cards.copy(), name=name)
        else:
            df = file_or_df
        return df
    
    def clean_card_df(self):
        raise NotImplementedError

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
        return self.bo1.groupby('draft_id').agg(d)

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