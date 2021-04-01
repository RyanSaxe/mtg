import requests
import urllib
import json as js
import pandas as pd

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
    def __init__(self, query_args, json_files = []):
        self.search_q = 'https://api.scryfall.com/cards/search?q='
        self.search_q += urllib.parse.quote(' & '.join([
            query for query in query_args
        ]))
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
        while json.get('has_more',False):
            self.cards = self.cards.union({Card(card) for card in json.get('data',[])})
            if json.get('next_page',None) is not None:
                next_page = requests.get(json['next_page'])
            json = next_page.json()
        self.cards = self.cards.union({Card(card) for card in json.get('data',[])})

    def _build_card_list_json(self, json_files):
        """
        store cards in the json_files in self.cards
        """
        for json_f in json_files:
            json = json.load(json_f)
            self.cards = self.cards.union({Card(card) for card in json.get('data',[])})

    def to_dataframe(self):
        card_data = [card.__dict__ for card in self.cards]
        return pd.DataFrame(card_data)

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
        if hasattr(self,"name"):
            self.name = self.name.lower()

        self.colnames = {
            'deck': 'deck_' + self.name,
            'hand': 'opening_hand_' + self.name,
            'drawn': 'drawn_' + self.name
        }

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, card2):
        return self.name == card2.name

    def __str__(self):
        return self.name
