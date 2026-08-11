# mtg

State of the art Magic: the Gathering Draft and DeckBuilder AI.

> **Fork note (mtg-draft-buddy).** This fork modernises the 2022 TensorFlow
> code into a PyTorch/MPS multi-set drafter. New code lives in
> `src/draftbot/`; `mtg/` below is the original, kept for reference. See
> `CLAUDE.md`, `docs/PLAN.md`, and `docs/HUD_PLAN.md`.

## draft HUD

A small always-on-top window that ranks the live pack while you draft in
Arena, then proposes a 40-card build when the draft ends.

```bash
uv sync --extra dev
uv run python -m draftbot.hud --ui window
# replay a captured log instead of a live draft:
uv run python -m draftbot.hud --ui window \
    --replay tests/fixtures/real_draft_msh.log --replay-delay 0.4
```

**Arena must have Detailed Logs on** — *Settings → Account → ✓ Detailed Logs
(Plugin Support)*, then restart Arena. Without it Arena writes no draft events
to `~/Library/Logs/Wizards Of The Coast/MTGA/Player.log` and the HUD sits idle.
The HUD only ever reads that file; it never writes to it or uploads anything,
and it coexists with the 17lands uploader tailing the same log.

When the draft ends the window switches to the deck builder: the proposed 40
with per-card membership confidence, the arguable boundary between the last
cards in and the first cards out, the cut list, and the mana split. Click any
row to lock it in or out and hit **Rebuild**; **Copy list** yields an
Arena-importable decklist.

### a set that released today

17lands publishes nothing for weeks and Scryfall has no `arena_id` yet, so
neither the vocab nor the grpId bridge can come from the usual places. Build
them from the Arena client's own card database instead:

```bash
uv run python -m draftbot.data.day0 --set HOB
uv run python -m draftbot.hud --ui window --stats none
```

Stats come out all-NaN → zeroed with missing-masks on, which is the
stat-dropout condition the trunk was pretrained under. These artifacts are
provisional (a `.day0` marker flags them) and are rebuilt with new card ids
once 17lands publishes; see docs/JOURNAL.md 2026-08-11.

Defaults are the corpus-pretrained production trunks — `EXP-033` drafting
(`--models`), `EXP-126` building with `EXP-116` for lock-and-rebuild
(`--deck-models`, or `--no-deck` to skip the builder). They ship over the
marginally-better-on-MSH single-set models (EXP-013 / EXP-121) because they
also work day-1 on a set they have never seen; add `--stats none` for that
case. See docs/DECKBUILDER_HANDOFF.md "Final model selection".

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
