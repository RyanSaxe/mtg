import numpy as np
import mtg.preprocess.seventeenlands import as seventeenlands
from mtg.obj.cards import CardSet
import pandas as pd

class Games:
    def __init__(
        self,
        file_or_df,
        expansion,
        **preprocess_kwargs,
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
        self.df = self._preprocess(df, **preprocess_kwargs)

        deck_cols = [x for x in df.columns if x.startswith("deck_")]
        card_names = [x.split("_",1)[-1] for x in deck_cols]
        self.id_to_name = {i:card_name for i,card_name in enumerate(card_names)}
        self.name_to_id = {name:idx for idx,name in self.id_to_name.items()}

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
    def search(self, queries, functions, conditions, styles, logic):
        """
        function that returns a subset of cames according to a flexible collection of conditions

        queries: a list of scryfall queries to get cards if cards=True, otherwise a list of columns
        functions: a list of functions to apply to the subset of the dataset relevant to the queries
        conditions: a list of functions that return True or False to apply to the transformed data
        styles: the type of query to segment the dataset, whether card in deck, hand, drawn or a normal 
                    set of columns in the dataframe
        logic: a function that creates a final mask by specifying how all the subquery masks interact

        example = {
        'n_equip':{
            'query':['cmc=2','t:equipment','rarity:c OR ratity:u'],
            'function': lambda x: x.sum(),
            'mask': lambda df: x
        }
    }
        """
        assert all([x in ["deck","hand","drawn","cols"] for x in styles])
        if isinstance(queries, str):
            assert all([
                not isinstance(functions, list)),
                not isinstance(conditions, list)),
                isinstance(style, str),
            ])
            qfcs = [[queries, functions, conditions, styles]]
        else:
            # if functions, conditions, styles are not lists
            #       then apply same function/condition/style to all
            #       queries.
            if not isinstance(functions, list):
                functions = [functions] * len(queries)
            if not isinstance(conditions, list):
                functions = [conditions] * len(queries)
            if not isinstance(styles, list):
                functions = [styles] * len(queries)
            assert len(queries) == len(functions) == len(conditions) == len(styles)
            qfcs = zip(queries, functions, conditions)
        masks = []
        for query, function, condition, style in qfcs:
            if style == "cols":
                subdf = self.df[query]
            else:
                query_cards = CardSet(query)
                subdf = self.df[card.colnames[col] for card in query_cards]             
            mask = subdf.apply(function, axis=1).apply(condition)
        final_mask = logic(mask)
        return df[final_mask]





