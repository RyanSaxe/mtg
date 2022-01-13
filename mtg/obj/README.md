# obj

This part of the project is responsible for card and data type objects. 

## CardSet

A `CardSet` object is meant to download card information from scryfall.com to easily integrate that information with other Magic data.

```python
from mtg.obj.cards import CardSet
#this object contains all cards in Crimson Vow with
#cmc 4 or greater, that are present in booster packs
VOW_expensive = CardSet([
    "set=vow",
    "cmc>=4",
    "is:booster",
])
```

Generally, to work with `CardSet` data, it is best to use a pandas DataFrame. So the CardSet object has a `to_dataframe` function for that conversion.

## Expansion

Different expansions have different custom rules and datasets. The `Expansion` object will automatically pull the proper statistical data from 17lands.com, and integrate that with information from scryfall.com using the `CardSet` object.

```python
from mtg.obj.expansion import VOW
#use_ml_data specifies to get the 17lands stat data
VOW_expansion = VOW(use_ml_data=True)
```

Additionally, you can pass `bo1` and `draft` arguments to any `Expansion` class to tell it to load 17lands bo1 game data or draft data. Currently I have not used the replay data or bo3 data, so there is no custom preprocessing for that.

When working with data from a new (or old set), create a child of the `Expansion` object accordingly.
