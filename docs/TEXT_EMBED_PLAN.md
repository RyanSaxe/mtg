# Oracle-text card embeddings — P3.T5c proposal (2026-09-05)

Owner: *"Running an embedding training regime on the card text would be pretty
awesome — I think that would capture quite a bit of alpha for day-0 drafts
next set."* This is the P3.T5 "ceiling-raiser" lever, never run. HOB is the
honest yardstick for it (quarantined sandbag with val data since 2026-09-05).

## 1. Why — measured, not assumed

| HOB val, EXP-033 expert top-1 | none = day 0 | day 3 | week 1 | full |
|---|---|---|---|---|
| | .5387 | .6644 | .6801 | .6855 |

- **12.5 of the 15-pt day-0 gap closes by day 3.** Stats arrive fast; the only
  thing that can move day 0 itself is what the model knows about a card before
  anyone has played it — and today that is 58 static columns.
- **Rules text is invisible.** Keywords are a fixed list of 20; 147 of 296
  HOB cards had rules text and zero Scryfall keywords; HOB's own mechanic
  ("Storied") is not in Scryfall's keyword array. In feature space
  Thranduil's Decree ≡ Sound the Trumpets except cmc (journal 2026-08-11).
- **Skill-gap is negative at day 0** (HOB −.009, MSH −.011) and positive with
  stats: without card semantics the trunk cannot tell expert taste from
  average taste. Rare-take is 1.52× expert at day 0 (1.07× by week 1): with
  nothing else to go on it leans on rarity.
- Literature: arXiv 2607.04782 (Jul 2026) builds set-contextualised card
  embeddings from oracle text with a 768-d text encoder; the LLM route
  (UrzaGPT, arXiv 2508.08382: 66% fine-tuned, GPT-4o 43% zero-shot) lands
  below our supervised numbers. Frozen encoder + learned adapter is the
  route.

## 2. Design

### D1 — card text → frozen sentence embedding (`features.text.parquet`)
- **Text.** One string per card from fields cards.parquet already persists:
  `"{name}. {mana_cost} {type_line}. {power}/{toughness}. {rarity}.
  {oracle_text}"`, with the card's own name inside the oracle text replaced
  by "this card" (the CARDNAME convention — stops names doing the work);
  DFCs: both faces joined by " // ".
- **Encoder.** Frozen `BAAI/bge-base-en-v1.5` (768-d, MIT, ~110M params,
  runs on MPS via sentence-transformers) as the default; `text_encoder` is a
  config knob (bge-small 384-d for the laptop; EmbeddingGemma-300m is gated).
  Cache: `data/processed/<SET>/features.text.<encoder>.parquet`, index =
  card id, rebuilt only when the text changes. ~9.5k cards over 31 sets ≈ 2
  min. `data/raw` untouched.
- **Plumbing.** `assemble(set, stats, text=<encoder>|None)` inserts the text
  block between static and stats — no missing-masks (text never goes
  missing). The corpus scaler z-scores it per dim like any continuous column;
  **HOB never contributes** (quarantine covers normalisation statistics).
  New config key `text_features: <encoder>`; `scorer_from_checkpoint`,
  `deck_builder_from_checkpoint` and the HUD read it from the checkpoint cfg,
  `feature_columns.json` records it, so every existing checkpoint keeps
  working unchanged. `day0.py` builds the block too (Scryfall oracle text is
  release-day available); `bundle.py` ships it.
- **Size.** Card-MLP input 771 → 1539 (+197k params; trunk 7.7M → 7.9M).
  `text_pca: 128` (fit on corpus cards only) is an ablation, not the default.
- **Regulariser.** `text_dropout` (whole-block dropout per batch, default 0;
  probe 0.1) against memorising corpus card text.

### D2 — stat-reconstruction auxiliary head (the "embedding training regime")
- On every pretraining step, from the **statless** card table, predict a
  small vector of the set's full-season stats per card — GIH WR, GP WR, IWD,
  ALSA, ATA (z-scored) — with a Linear head on the feature half of the card
  embedding. NaN-masked MSE over the batch's set's whole card table (~300
  rows: negligible cost). `stat_aux_lambda` (probe 0.1 / 0.3).
- Why: the pick loss teaches "text → strength" only through pack-relative
  choices; this gives ~9k (text → measured strength) pairs of direct
  supervision across 30 solved sets, reprints included (same text, several
  measurements). The head is discarded at inference, but its output on a new
  set is a free text-only card rating — a diagnostic, and a day-0 HUD tooltip.

### D3 — set-contextualised text (deferred)
The paper's trick (attention over the set's cards before scoring) is partly
covered by our set token. Revisit only if D1/D2 show signal.

## 3. Experiments (P3.T5 rules: one lever per EXP, HOB val expert top-1, val only)

| EXP | = | lever | promote if |
|---|---|---|---|
| EXP-050 | EXP-034 recipe + text block (bge-base, raw 768) | text | none ≥ +0.5 over EXP-034; day3/week1/full ≥ EXP-034 − 0.2 |
| EXP-051 | EXP-050 + stat aux head λ 0.1 | aux | none ≥ +0.5 over EXP-050 |
| EXP-052 | EXP-051 + text_dropout 0.1 | regulariser | only if 050/051 diverge train vs val on `none` |
| EXP-139 | EXP-126 deck recipe + text block | text (decks) | HOB none trophy-F1 > .8477 |

Bars stay honest: MSH in-corpus full must hold (.6955). Diagnostics per run:
Spearman(aux-predicted GIH, actual full-season GIH) over HOB cards (how good
is the text-only rating?); P1P1 probes on Thranduil's Decree / Uncover the
Moon-Letters; day-0 rare-take ratio (1.52× today — text should cut rarity
reliance); day-0 skill-gap (negative today).

## 4. Cost, dependencies, risks
- Implementation ≈ half a day: `data/text_features.py`; assemble / loading /
  day0 / bundle plumbing; aux head in `models/modern.py` + loss in
  `train/pretrain.py`; tests (dims identical across sets; day-0 build; old
  checkpoints unaffected; aux loss finite under NaN stats).
- Compute: EXP-050/051 ≈ 5–6 h each (overnight slots after EXP-034/035).
- Dependency: `sentence-transformers` as an optional extra `[text]`. The
  laptop needs the encoder only to onboard a set on day 0; otherwise the
  bundle ships the parquet.
- Risks: memorisation of corpus text (the HOB holdout measures it;
  text_dropout is the knob); feature-dim drift breaking the HUD (cfg-driven,
  tested); encoder availability (bge is MIT and ungated).
- Fallback if the encoder disappoints: a regex mechanic bag over oracle text
  (~150 named mechanics + counts of draw / destroy / counter / exile /
  +1/+1 counter / token) — interpretable, zero dependencies.

## 5. Not doing
Fine-tuning the encoder or the LLM route: 43–66% in the literature vs our .68
with stats, and it does not fit the MPS budget. Frozen encoder + the card
MLP as a learned adapter captures the useful part.
