import requests
import urllib
import json as js
import pandas as pd
import mtg.obj.scryfall_utils as scry_utils


class CardSet:
    """
    Run a scryfall search for list of cards according to a query:
    
    parameters:
    
        query_args: a list of queries to hit the scryfall api with
        
    usage:
    
        KHM_expensive = Cards([
            "set=khm",
            "cmc>=4",
        ])
        #this gets you all kaldheim cards with cmc 4 or greater
        
    """

    def __init__(self, query_args, json_files=[]):
        self.search_q = "https://api.scryfall.com/cards/search?q="
        if isinstance(query_args, str):
            self.search_q += urllib.parse.quote(query_args)
        else:
            self.search_q += urllib.parse.quote(
                " & ".join([query for query in query_args])
            )
        response = requests.get(self.search_q)
        self._json = response.json()
        self.cards = set()
        self._build_card_list_query()
        self._build_card_list_json(json_files)

    def _build_card_list_query(self):
        """
        store cards from the query in self.cards
        """
        json = self._json
        while json.get("has_more", False):
            self.cards = self.cards.union({Card(card) for card in json.get("data", [])})
            if json.get("next_page", None) is not None:
                next_page = requests.get(json["next_page"])
            json = next_page.json()
        self.cards = self.cards.union({Card(card) for card in json.get("data", [])})

    def _build_card_list_json(self, json_files):
        """
        store cards in the json_files in self.cards
        """
        for json_f in json_files:
            json = json.load(json_f)
            self.cards = self.cards.union({Card(card) for card in json.get("data", [])})

    def to_dataframe(self):
        card_data = [card.__dict__ for card in self.cards]
        df = pd.DataFrame(card_data)
        df = self.scryfall_modifications(df)
        # modify so that basics have the first 5 idxs
        basics = ["plains", "island", "swamp", "mountain", "forest"]
        card_names = [x for x in df["name"].unique()]
        for basic in basics:
            card_names.remove(basic)
        card_names = basics + card_names
        id_to_name = {i: card_name for i, card_name in enumerate(card_names)}
        name_to_id = {name: idx for idx, name in id_to_name.items()}
        df["idx"] = df["name"].apply(lambda x: name_to_id[x])
        return df

    def scryfall_modifications(self, df):
        df = df.apply(scry_utils.merge_card_faces, axis=1)
        df["produces_for_splash"] = df.apply(scry_utils.produce_for_splash, axis=1)
        df["name"] = df["name"].apply(lambda x: x.split("//")[0].strip().lower())
        return df

    def union(self, cardset2):
        return self.cards | cardset2.cards

    def intersection(self, cardset2):
        return self.cards & cardset2.cards

    def difference(self, cardset2):
        return self.cards - cardset2.cards

    def simdiff(self, cardset2):
        return self.cards ^ cardset2.cards


class Card:
    def __init__(self, *args, **kwargs):
        for dictionary in args:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])
        if hasattr(self, "name"):
            self.name = self.name.lower()

        self.colnames = {
            "deck": "deck_" + self.name,
            "hand": "opening_hand_" + self.name,
            "drawn": "drawn_" + self.name,
            "sideboard": "sideboard_" + self.name,
        }

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, card2):
        return self.name == card2.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
