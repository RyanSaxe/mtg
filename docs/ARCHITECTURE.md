# Architecture

How the system works as of the EXP-056 production trunk. For why each choice
was made, the experiment that settled it is named in brackets; the narrative
is in [JOURNAL.md](JOURNAL.md) and the numbers in
[LEADERBOARD.md](LEADERBOARD.md).

```
17lands draft/game CSVs ─┐                       ┌─ zero-shot eval on the held-out set
Scryfall card objects ───┼─► per-set artifacts ─►│  multi-set pretraining ─► trunk (EXP-056)
Arena card DB (day 0) ───┘   cards / features /  └─ per-set fine-tune ─────► set model
                             snapshots / splits                                  │
                                                     Player.log ─► HUD ◄─────────┘
                                                                    └─► deck builder
```

## 1. Data

Code: `src/draftbot/data/`. Everything under `data/` is generated and never
committed; `data/raw/` is immutable.

- **Draft logs.** 17lands public `draft_data` CSVs are streamed into a compact
  Parquet, one row per pick: the pack's card ids, the pick, the drafter's rank
  and win-rate bucket, the event record. The pool is not stored; it is the
  prefix of picks.
- **Card table.** Per set, `cards.parquet` is exactly the names that appear in
  the 17lands CSV header, resolved against Scryfall (the set plus its bonus
  sheets). Basics are real vocabulary entries.
- **Features**, three blocks per card:
  - *static* — mana value, colour pips, types, rarity, power/toughness,
    keywords, produced mana. Available before release.
  - *stats* — pick-order and win-rate statistics computed from the raw draft
    and game files over a date window, saved as immutable **snapshots**
    (`week1` = release to release+7 days, `full` = season). Continuous columns
    are z-scored with corpus statistics; a missing value becomes 0 plus a
    paired missing-mask column, never a silent "0% win rate".
  - *text* — one canonical string per card (name, cost, type line, oracle
    text) through a frozen `BAAI/bge-base-en-v1.5` sentence encoder, 768-d.
- **Stat modes.** Every model is evaluated, and the trunk trained, under
  `none` / `week1` / `full` — what you know on release day, a week in, and at
  season's end.
- **Splits** are by `draft_id`, written once to `data/splits/<SET>.json` and
  never regenerated. The *expert subset* (high win rate, enough games) is the
  selection metric.
- **Day 0.** Before 17lands publishes, `data/day0.py` builds a provisional
  card table from the Arena client's local card database plus Scryfall, with
  every statistic missing.

## 2. Draft model — `ModernDraftBot`

Code: `src/draftbot/models/modern.py`. Production size: embedding 256, 8
layers, 8 heads, 8.1M parameters.

**Card encoder.** A card embedding is the concatenation of two halves: a
shared MLP over the card's feature vector, and a learned per-card id vector.
For pretraining the id half is zero (`feature_only`), so a card is *only* its
features and the same weights apply to any set; a per-set id table is trained
only when fine-tuning. `set_context()` swaps the card universe per batch.

**Per-pick input.** For pick *t* the token is a linear fusion of

- a **pack summary** — two set-attention blocks over the cards in the pack,
  then pooling by multi-head attention (one learned seed query);
- the embedding of the **previous pick**;
- a **position** embedding.

A **set token**, an MLP of the mean feature vector of the whole set, is
prepended so the model can calibrate to the set it is drafting.

**Trunk.** A decoder-only causal transformer over the picks (pre-norm RMSNorm,
SwiGLU feed-forward). The pool is never fed explicitly; the decoder state at
*t* has attended to every earlier pack and pick.

**Pointer head.** `logit(card) = ⟨q(state_t), k(card embedding)⟩ / √d` over the
cards in the pack only. There is no output vocabulary, which is what lets the
head score cards it has never seen.

**Loss.** Cross-entropy on the human pick with label smoothing, weighted
toward stronger drafters. Soft weighting beat a hard skill filter for
single-set training [EXP-013 vs EXP-014]; pretraining uses a mild filter (win rate ≥ .54,
≥ 50 games). No hand-written priors: the rare-take rate is watched as an eval
metric instead.

### Pretraining for unseen sets

Code: `src/draftbot/train/pretrain.py`, manifests in `configs/corpus/`.

- **Corpus.** ~30 sets, each batch drawn from one set with its own card
  context; recent play-booster sets are weighted 2×.
- **Stat dropout.** A quarter of batches see the statistics block as entirely
  missing, so `--stats none` at inference is a trained condition rather than a
  degraded one.
- **Text noise.** Gaussian noise (σ = 2.0) is added to the text block during
  pretraining. Without it the model uses a 768-d embedding as a card id: a
  control with a fixed *random* vector per card fits the corpus equally well
  and loses about 3 points on an unseen set [EXP-054]. Noise removes the id
  and keeps the semantics [EXP-055…060].
- **Stat-reconstruction head.** An auxiliary head predicts a card's measured
  win-rate statistics from its stat-less features. It turned out redundant
  once noise was in [EXP-059] but is kept in the production config.
- **Quarantine.** One set — the newest — is excluded from every pretraining
  corpus, scaler fit and model-selection signal, and is what zero-shot
  numbers are reported on. `tests/test_quarantine.py` enforces it. When a new
  set arrives the old one is promoted into the next corpus manifest and the
  trunk is retrained from scratch (PLAN Appendix D); earlier manifests are
  frozen as records.

### Adapting to a set

`draftbot.onboard_set` fine-tunes the trunk on a set once its data exists:
`full`, `head`, or `lora` (low-rank adapters on attention and feed-forward
weights, with the per-set id table trained in full). LoRA lands within ~0.2
points of full fine-tuning at ~4% of the parameters.

## 3. Deck builder — `DeckBuilder`

Code: `src/draftbot/models/builder.py`, about 0.8M parameters.

The pool is a set of (card, copies) slots → feature MLP + copy-count embedding
→ three set-attention blocks → three heads:

- **membership** — per slot, the probability that copy is in the maindeck;
- **land count** — a distribution over 14–20 total lands (from the pooled
  state);
- **basics** — the colour split of the basic lands.

Decoding fills 40 cards greedily by membership, honouring the land count and
basic split; nonbasic lands compete for slots like any other card.

The **masked-infilling variant** (`diffusion=True`, EXP-116) adds a per-slot
deck-state embedding (decided count, or MASK), is trained with random masking
and decoded by iterative commitment. Locks from the HUD enter as already
decided slots, so "rebuild around these cards" is native rather than a
post-hoc patch. A deck-quality head trained on winner/loser pairs from the
same pool ranks candidate builds.

Like the drafter, the builder is feature-based and pretrained across sets
(EXP-126), with a format embedding for joint draft + sealed training
(EXP-135).

## 4. Evaluation

Code: `src/draftbot/eval/`. Leaderboard rows come only from
`python -m draftbot.eval`.

- **Scorecard** — top-1/2/3 overall and by pick, pack, rarity and expert
  subset; NLL; calibration (ECE); rare-take rate against the human rate; the
  number of colours a greedy self-draft ends in.
- **Baselines** — random, rarity-first, ALSA-greedy, GIH-greedy, GIH-in-lane,
  and `LegacyDraftBot`.
- **Behavioural fixtures** — hand-written pack/pool scenarios with expected
  picks (`tests/fixtures/scenarios/`), run in every stat mode.
- **Correctness tests** for any model: causality (future picks cannot change
  earlier logits), pack masking, an overfit canary, finite loss under missing
  stats.
- **Promotion rule** — one lever per experiment, selected on val expert
  agreement, +0.5 points to promote; test splits only at declared gates.

## 5. HUD

Code: `src/draftbot/hud/`. A follower tails Arena's `Player.log`, a state
machine reconstructs the draft (packs, picks, pool) from the log events, the
advisor scores the current pack with the production checkpoints, and a small
local web view (or terminal UI) shows the ranking. At the end of the draft the
deck panel shows the proposed build and handles lock-and-rebuild. It only
reads the log.

## 6. Relation to the 2022 model

`LegacyDraftBot` (`models/legacy.py`) is a PyTorch port of Ryan Saxe's
architecture and the baseline row on the leaderboard.

| | 2022 model (`mtg/`, ported as `LegacyDraftBot`) | this project |
|---|---|---|
| framework | TensorFlow / Keras 2 | PyTorch (MPS) |
| structure | encoder–decoder | decoder-only causal |
| pack representation | mean of card embeddings | set attention + attention pooling |
| output | MLP over the set's vocabulary, masked to the pack | pointer over in-pack card embeddings |
| card identity | per-set learned ids + feature MLP | features only in the trunk; ids only at fine-tune |
| card text | not used | frozen sentence-encoder embedding, noise-regularised |
| training data | one set | ~30 sets, one held out |
| auxiliary losses | triplet embedding loss, rare/cmc priors | none on the pick; stat reconstruction on features |
| missing stats | filled with 0 | missing-masks; stat dropout in training |
| new set | retrain from scratch | zero-shot on day 0, fine-tune when data exists |
| deck building | separate model, single set | set-attention builder, multi-set, lock-and-rebuild, sealed |
| MSH test top-1 / expert | .574 / .598 (EXP-001) | .676 / .699 (EXP-013) |
