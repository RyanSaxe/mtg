# scripts

Scripts for preprocessing data, and training a `DraftBot` and `DeckBuilder`

## instructions

First, Download bo1 game data and bo1 draft data for an expansion from https://www.17lands.com/public_datasets

The following will preprocess the data and store it:

```
>>> python preprocess.py  --expansion VOW \
                          --game_data path/to/game/data.csv \
                          --draft_data path/to/draft/data.csv \
                          --expansion_fname path/to/expansion.pkl
```

Now, you can run the script to train the draft model using the preprocessed data.

```
>>> python train_drafter.py  --expansion_fname path/to/expansion.pkl \
                             --model_name path/to/draft_model
```

And the same for the deckbuilder model. Note, that it is advised to train the draft model first so that you can use the embeddings from it in the deckbuilder model:

```
>>> python train_builder.py  --expansion_fname path/to/expansion.pkl \
                             --draft_model path/to/draft_model \
                             --model_name path/to/build_model
```

If you want to train any of these models with different hyperparameters, please check the flags specified in the corresponding scripts.

## usage

Once you've trained your instances of these models, you can load them in python and see how they would build decks and make decisions given 17lands logs. If you don't currently have an account at 17lands.com, please make one, as you need an API token for this part. Below is an example of seeing how the model would make different deckbuilding decisions and draft decisions given a 17lands log:

```python
from mtg.ml.utils import load_model
from mtg.ml.display import draft_log_ai
import pickle

draft_model, attrs = load_model("path/to/draft_model")
build_model, cards = load_model("path/to/build_model", extra_pickle="cards.pkl")
expansion = pickle.load(open("path/to/expansion.pkl, "rb"))

log = 'https://www.17lands.com/draft/[draft_id]'
token = '[your API token]'
# log_url[0] will be a link to a 17lands draft log
# log_url[1] will be a link to a sealeddeck.tech deckbuild
log_url = draft_log_ai(
    log,
    draft_model,
    expansion=expansion,
    batch_size=1,
    token=token,
    build_model=build_model,
    basic_prior=True,
)
```
