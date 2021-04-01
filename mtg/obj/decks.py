import numpy as np
import mtg.preprocess.seventeenlands import as seventeenlands

class Decks:
    def __init__(
        self,
        filename,
        expansion,
        **preprocess_kwargs,
    ):
        self.expansion = expansion
        self.cards = seventeenlands.get_card_rating_data(
            self.expansion, join=True
        )
        df = pd.read_csv(filename)
        df = self._preprocess(df, **preprocess_kwargs)

        deck_cols = [x for x in df.columns if x.startswith("deck_")]
        card_names = [x.split("_",1)[-1] for x in deck_cols]
        self.id_to_name = {i:card_name for i,card_name in enumerate(card_names)}
        self.name_to_id = {name:idx for idx,name in self.id_to_name.items()}
        #step 1: figure out how to reduce duplicates
        #step 2: store deck metadata
        self.decks = df[deck_cols].to_numpy()

    def _preprocess(self, df, **preprocess_kwargs):
        for func_name, kwargs in preprocess_kwargs.items():
            func = getattr(seventeenlands, func_name)
            df = func(df, **kwargs)
        return df
        
    def search_meta(self,query):
        """
        return a subset of decks according to meta data
        """
        pass

    def contains(self, cards):
        """
        given a dictionary of cards, return decks containing those cards
        """



