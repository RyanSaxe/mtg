# ml

This repository is dedicated to Machine Learning implementations for Magic: the Gathering. Below is a quick description of each file, if you would like to see how to take all of this and train your own instances of the models in this folder, please refer to the `train_xxx.py` files in `mtg/scripts/`.

## layers.py

This file contains implementations of layers, such as `Dense`, `MultiHeadedAttention`, `LayerNormalization`, and more!

## nn.py

This file contains implementations of module blocks that models can use such as `MLP` and `TransformerBlock`. Here is an example of building an autoencoder using the `MLP` block.

```python
import tensorflow as tf
from mtg.ml.nn import MLP

class AutoEncoder(tf.Module):
  def __init__(self, in_dim, emb_dim, name=None):
    super().__init__(name=name)
    self.encoder = MLP(
      in_dim = in_dim,
      start_dim = in_dim // 2
      n_h_layers = 2,
      out_dim = emb_dim,
      style="bottleneck",
    )
    self.decoder = MLP(
      in_dim = emb_dim,
      start_dim = emb_dim * 2,
      n_h_layers = 2,
      out_dim = in_dim,
      style="reverse_bottleneck",
    )
    
  def __call__(self, x, training=None):
    embedding = self.encoder(x, training=training)
    return self.decoder(embedding, training=training)
 ```

## model.py

This file contains implementations of projects to apply Machine Learning to Magic: the Gathering. Currently it contains a model for drafting, `DraftBot` and a model for deckbuilding, `DeckBuilder`

## generator.py

This file contains data generator objects for batching 17lands data properly to feed into models in `models.py`.

## trainer.py

This file contains the custom training object used to train models from `models.py` using generators from `generator.py`.

## display.py

This file contains different ways to visualize and run pretrained models. Here is an example of a common use case for debugging:

```python
from mtg.ml.display import draft_sim

# assume draft_model and build_model are pretrained instances of those MTG models
# assume expansion is a loaded instance of the expansion object containing the 
#     data corresponding to draft_model and build_model
# then, draft_sim as ran below will spin up a table of 8 bots and run them through a draft.
#       what is returned is links to 8 corresponding 17land draft logs and sealeddeck.tech deck builds.

token = "abcdefghijk1234567890" #replace this with your 17lands API token
bot_table = draft_sim(expansion, draft_model, token=token, build_model=build_model)
```

## utils.py

This file contains utility functions needed for the models such as learning rate schedulers, and model loading functions.

## TODO:

- integrate the deckbuilder model to be a part of the drafting model.
- update MLP implementation such that `n_h_layers` actually corresponds to the number of hidden layers (at the moment, it's technically 1 more)
