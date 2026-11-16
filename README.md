# mtg-draft-buddy

A Magic: the Gathering draft and deck-building assistant for Arena: a
multi-set pretrained transformer that ranks every pack as you draft, then
proposes a 40-card build — including for a set that released today and has no
play data yet.

It started as a fork of [RyanSaxe/mtg](https://github.com/RyanSaxe/mtg), Ryan
Saxe's 2022 transformer draft AI (the bot he took to Mythic #27). The models,
data pipeline, training and evaluation code here are a ground-up rewrite in
PyTorch under `src/draftbot/`; Ryan's TensorFlow implementation is kept
unmodified in `mtg/` as the reference it was ported from. See
[Lineage](#lineage) for what carried over and what did not.

## What it does

- **Draft advisor** — a causal transformer over the picks of a draft that
  scores the cards in the current pack. One trunk is pretrained on ~30 sets of
  [17lands](https://www.17lands.com/public_datasets) draft logs and reads
  cards through their *features* (Scryfall characteristics, 17lands
  statistics, and an embedding of the oracle text) rather than through
  per-card ids, so it can draft a set it has never seen.
- **Deck builder** — a set-attention model over the drafted pool that predicts
  maindeck membership, land count and basic split; a masked-infilling variant
  rebuilds around cards you lock in or out. Draft and sealed.
- **HUD** — a small always-on-top window that follows Arena's `Player.log`,
  ranks the live pack, and switches to the deck builder when the draft ends.
- **Set onboarding** — one command on release day (no play data), one command
  when 17lands publishes (fine-tune).

## Results

All numbers come from `python -m draftbot.eval`; the full table is
[docs/LEADERBOARD.md](docs/LEADERBOARD.md). "Expert" is agreement with the
picks of high-win-rate drafters, the selection metric throughout.

MSH (Marvel Super Heroes), held-out **test** split, top-1 agreement:

| model | top-1 | expert top-1 |
|---|---|---|
| best heuristic (ALSA-greedy) | .467 | .484 |
| `LegacyDraftBot` — PyTorch port of the 2022 architecture (EXP-001) | .574 | .598 |
| `ModernDraftBot`, MSH only (EXP-013) | .676 | .699 |
| pretrained trunk, MSH never seen, week-1 stats (EXP-030) | .649 | .670 |
| pretrained trunk, MSH never seen, no stats (EXP-030) | .524 | .517 |

HOB (The Hobbit), **val** split, expert top-1 — the set held out from all
pretraining, by how much is known about the set:

| trunk | day 0 (no stats) | week-1 stats | full stats |
|---|---|---|---|
| EXP-033, no card text | .539 | .680 | .686 |
| **EXP-056, with card text** | **.568** | **.687** | **.693** |
| EXP-058, EXP-056 fine-tuned on HOB | | | .713 |

The production trunk is now **EXP-061**: the EXP-056 recipe retrained with HOB
and FRA (Reality Fracture) in the corpus and TRK (Star Trek) held out. It has
no zero-shot number yet because 17lands has not published TRK. On FRA, which
EXP-056 never saw, EXP-056 scored .538 / .672 / .679 expert (day 0 / week 1 /
full) and fine-tuning reached .699.

Deck builder on HOB val (card-level F1 against the human build): .879
zero-shot, .906 fine-tuned.

## Quick start

```bash
uv sync --extra dev          # Python 3.12, PyTorch (MPS on Apple Silicon)
uv run pytest
```

Checkpoints and data are not in the repository. Either build them (below) or
unpack a bundle made with `python -m draftbot.bundle` on the training machine.

### Draft HUD

```bash
uv run python -m draftbot.hud --ui window
# replay a captured log instead of a live draft:
uv run python -m draftbot.hud --ui window \
    --replay tests/fixtures/real_draft_msh.log --replay-delay 0.4
```

**Arena must have Detailed Logs on** — *Settings → Account → ✓ Detailed Logs
(Plugin Support)*, then restart Arena. Without it Arena writes no draft events
to `~/Library/Logs/Wizards Of The Coast/MTGA/Player.log` and the HUD sits idle.
The HUD only reads that file; it never writes to it or uploads anything, and
it coexists with the 17lands uploader tailing the same log.

When the draft ends the window switches to the deck builder: the proposed 40
with per-card membership confidence, the arguable boundary between the last
cards in and the first cards out, the cut list, and the mana split. Click any
row to lock it in or out and hit **Rebuild**; **Copy list** yields an
Arena-importable decklist.

Defaults are the corpus-pretrained production models: `EXP-061` drafting
(`--models`; `EXP-033` is the no-text fallback), `EXP-126` building with
`EXP-116` for lock-and-rebuild (`--deck-models`, `--no-deck` to skip).
The HUD's `--stats none|week1|full` selects how much 17lands data the model is shown.

### A set that released today

17lands publishes nothing for about three weeks and Scryfall has no
`arena_id` yet, so the card list comes from the Arena client's own card
database (launch Arena first so it downloads the set):

```bash
uv sync --extra text                              # sentence encoder for card text
uv run python -m draftbot.data.day0 --set HOB
uv run python -m draftbot.hud --ui window --stats none
```

Statistics are all missing, which is a condition the trunk was trained under
(stat dropout). These artifacts are provisional (a `.day0` marker) and are
rebuilt with final card ids once 17lands publishes.

### When 17lands publishes the set

```bash
uv run python -m draftbot.onboard_set --set HOB --finetune-mode full
```

Downloads the draft and game files, builds the card table, features, stat
snapshots and splits, writes zero-shot scorecards for the pretrained trunk,
then fine-tunes it on the set (`lora`, `head` or `full`) and scores the result
under `docs/scorecards/onboard/<SET>/`.

### Training and evaluation

```bash
uv run python -m draftbot.train.pretrain --config configs/EXP-061.yaml   # multi-set trunk
uv run python -m draftbot.train --config configs/EXP-058.yaml            # single-set / fine-tune
uv run python -m draftbot.eval --model checkpoints/EXP-061 --set FRA --split val --stats-snapshot week1
```

Every run is a YAML under `configs/` with an experiment id and a seed; runs
checkpoint and resume (`--resume`).

## How it works

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) is the full description. In
short:

- **Cards are feature vectors.** Static Scryfall features, z-scored 17lands
  statistics with explicit missing-masks, and a frozen `bge-base` embedding of
  the oracle text. A shared MLP turns them into card embeddings, so the same
  weights serve every set.
- **Packs are sets.** Two set-attention blocks and attention pooling summarise
  the pack; a decoder-only causal transformer runs over the draft's picks.
- **A pointer head** scores each card in the pack by dot product with the
  decoder state — no fixed output vocabulary.
- **Pretraining is built for unseen sets.** Statistics are dropped for a
  quarter of batches, and Gaussian noise on the text block stops the model
  using an embedding as a card id. One set is always quarantined from the
  corpus to measure zero-shot drafting honestly (`tests/test_quarantine.py`).

## Repository layout

| path | what |
|---|---|
| `src/draftbot/data/` | 17lands download, CSV→Parquet, card tables, features, stat snapshots, splits, day-0 onboarding |
| `src/draftbot/models/` | `ModernDraftBot`, `DeckBuilder`, LoRA, and `LegacyDraftBot` (port of the 2022 architecture, the baseline) |
| `src/draftbot/train/`, `eval/` | training loops, pretraining, eval harness, heuristic baseline bots |
| `src/draftbot/hud/` | log follower, draft state, window/terminal UI, deck panel |
| `configs/` | one YAML per experiment; `configs/corpus/` are the pretraining manifests |
| `docs/` | [ARCHITECTURE](docs/ARCHITECTURE.md), [PLAN](docs/PLAN.md), [JOURNAL](docs/JOURNAL.md) (append-only lab notebook), [LEADERBOARD](docs/LEADERBOARD.md), per-set data cards and scorecards |
| `mtg/` | Ryan Saxe's original TensorFlow code, unmodified, reference only |

## Lineage

Ryan Saxe's `mtg` showed that a transformer trained on 17lands logs can draft
at a Mythic level, and this project began as an attempt to reproduce it on
2026 data. What survives from it:

- the problem framing — next-pick prediction over a draft, trained on 17lands
  public data, weighted toward stronger drafters;
- the idea of a card embedding that is half learned id, half feature MLP;
- `LegacyDraftBot`, a faithful PyTorch port, kept as the baseline every newer
  model is measured against.

Everything else is new: the architecture (decoder-only, set-attention packs,
pointer head), feature-only multi-set pretraining and zero-shot drafting,
oracle-text embeddings, stat snapshots with missing-masks, the quarantine
protocol and eval suite, the deck builder and sealed support, the HUD, and
day-0 onboarding. The original's README, results and training scripts are in
`mtg/` and at the upstream repository; the Mythic result is Ryan's, with his
model, and is not a claim about this one.

## License

Apache 2.0, as upstream (see `LICENSE`).
