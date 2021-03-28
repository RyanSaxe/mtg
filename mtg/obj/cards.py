import requests
import urllib
import json as js
import pandas as pd

class Cards:
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
        search_q = 'https://api.scryfall.com/cards/search?q='
        search_q += '&'.join([
            urllib.parse.quote(query) for query in query_args
        ])
        response = requests.get(search_q)
        self._json = response.json()
        self.cards = []
        self._build_card_list_query()
        self._build_card_list_json(json_files)
        self._clean_duplicates()
        
    def _build_card_list_query(self):
        """
        store cards from the query in self.cards
        """
        json = self._json
        while json['has_more']:
            self.cards += [Card(card) for card in json['data']]
            next_page = requests.get(json['next_page'])
            json = next_page.json()
        self.cards += [Card(card) for card in json['data']]

    def _build_card_list_json(self, json_files):
        """
        store cards in the json_files in self.cards
        """
        for json_f in json_files:
            json = json.load(json_f)
            self.cards += [Card(card) for card in json['data']]
    
    def _clean_duplicates(self):
        """
        remove duplicates from self.cards
        """
        temp_cards = []
        for card in self.cards:
            if card not in temp_cards:
                temp_cards.append(card)
        self.cards = temp_cards

    def to_dataframe(self):
        card_data = [card.__dict__ for card in self.cards]
        return pd.DataFrame(card_data)
        
class Card:
    def __init__(self, *args, **kwargs):
        for dictionary in args:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])
        if hasattr(self,"name"):
            self.name = self.name.lower()
    def __eq__(self, card2):
        return self.name == card2.name