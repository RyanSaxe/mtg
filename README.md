# mtg

State of the art Magic: the Gathering Draft and DeckBuilder AI.

## achievements

![mythicbot](https://user-images.githubusercontent.com/2286292/149001531-9c983259-4ac6-4ed3-b54a-b0705fb57124.PNG)

This repository contains an algorithm for automated drafting and building for Magic: the Gathering. I used this algorithm to achieve the highest rank (Mythic) on Magic Arena. I did so in 23 drafts, with a 66% win-rate, which is comparable to how I perform on my normal account in which I do not use any algorithms. The highest rank within Mythic I have hit so far is #27. As far as I know, this is the first time anybody has acheived results of this caliber using an AI in Magic: the Gathering.

## architecture

Below is a general description of the transformer architecture for the Draft AI in order to make it easier to grok than reading through code.

![transformer](https://user-images.githubusercontent.com/2286292/158926118-86d8301e-8c0e-43c2-a21b-cced4f785b97.jpg)

## installation

- Step 1: clone this repository, and cd into it.
- Step 2: create a virtual environment in whatever your favorite way to do that is (e.g. `conda create -n my_env` -> `conda activate my_env`).
- Step 3: `pip install .` will install this repo such that you can use `from mtg.xxx.yyy import zzz`.

**NOTE:** I am not currently providing a pretrained instance of the Draft AI or DeckBulder AI in this repository. That means you cannot simply install this codebase, launch Magic Arena, and use the bot like I do. If you would like to do that, you need to use this code to train it yourself following [these instructions](mtg/scripts). A non-cleaned version of the UI I use that interacts with Magic Arena can be found [here](https://github.com/RyanSaxe/MTGA_Draft_17Lands), and it will eventually be cleaned and added to this repository under mtg/app/.

## documentation

Find any documentation on usage of the different sections in the README of their corresponding folders.

## TODO

- Integrate deckbuilder and drafter in one end-to-end pipeline.
- Add mtg/viz/ as a folder for containing 17lands data visualizations, explorations, and useful insights.
- Add mtg/app/ as a folder to contain the application UI for running on live arena drafts.
