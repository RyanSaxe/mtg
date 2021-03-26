class Cards:
    """
    Run a scryfall search for list of cards according to a query:
    
    parameters:
    
        query_args: a list of lists of the form [[key,condition,value], . . .]
        
    usage:
    
        KHM_expensive = Cards([
            ["set","=","khm"],
            ["cmc",">=","4"]
        ])
        #this gets you all kaldheim cards with cmc 4 or greater
        
    """
    def __init__(self, query_args):
        search_q = 'https://api.scryfall.com/cards/search?q='
        search_q += '&'.join([
            urllib.parse.quote(''.join(query))
            for query in query_args
        ])
        response = requests.get(search_q)
        self._json = response.json()
        self._build_card_list()
        
    def _build_card_list(self):
        json = self._json
        self.cards = []
        while json['has_more']:
            self.cards += [Card(card) for card in json['data']]
            next_page = requests.get(json['next_page'])
            json = next_page.json()
        self.cards += [Card(card) for card in json['data']]

    @property
    def n_rarity(self):
        init_rarity = {
            'common':0,
            'uncommon':0,
            'rare':0,
            'mythic':0
        }
        for card in self.cards:
            init_rarity[card.rarity] += 1
        return init_rarity
        
class Card:
    def __init__(self, *args, **kwargs):
        for dictionary in args:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])
        if hasattr(self,"name"):
            self.name = self.name.lower()