import numpy as np

class Decks:
    def __init__(self, df, cards):
        deck_cols = [x for x in df.columns if x.startswith("deck_")]
        card_names = [x.split("_",1)[-1] for x in deck_cols]
        self.id_to_name = {i:card_name for i,card_name in enumerate(card_names)}
        self.name_to_id = {name:idx for idx,name in self.id_to_name.items()}
        self.cards = cards.to_pandas(mapping=self.name_to_id)
        #step 1: figure out how to reduce duplicates
        #step 2: store deck metadata
        self.decks = df[deck_cols].to_numpy()

    def search_meta(self,query):
        """
        return a subset of decks according to meta data
        """
        pass

    def contains(self, cards):
        """
        given a dictionary of cards, return decks containing those cards
        """



