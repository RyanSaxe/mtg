# mtg

Collection of data science and machine learning projects for Magic: the Gathering, including a state of the art Draft and DeckBuilder AI.

## achievements

![mythicbot](https://user-images.githubusercontent.com/2286292/149001531-9c983259-4ac6-4ed3-b54a-b0705fb57124.PNG)

This repository contains an algorithm for automated drafting and building for Magic: the Gathering. I used this algorithm to achieve the highest rank (Mythic) on Magic Arena. I did so in 23 drafts, with a 66% win-rate, which is comparable to how I perform on my normal account in which I do not use any algorithms. The highest rank within Mythic I have hit so far is #27. As far as I know, this is the first time anybody has acheived results of this caliber using an AI in Magic: the Gathering.

## installation

- Step 1: clone this repository, and cd into it.
- Step 3: create a virtual environment in whatever your favorite way to do that is (e.g. `conda create -n my_env` -> `conda activate my_env`).
- Step 3: `pip install .` will install this repo such that you can use `from mtg.xxx.yyy import zzz`.

## documentation

Find any documentation on usage of the different sections in the README of their corresponding folders.

## TODO

- Integrate deckbuilder and drafter in one end-to-end pipeline
- Create project for 17lands data visualizations and explorations
