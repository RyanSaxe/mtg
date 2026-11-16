# Engineering hand-off: HUD deck-builder section (HUD H5 / PLAN P5)

## Final model selection (owner sign-off 2026-07-31) — LOCKED

Production is the **corpus-pretrained trunks**, not the best in-set scorers:

| role | ship | best-on-MSH | why the shipped one |
|---|---|---|---|
| draft | **EXP-061** (EXP-056 until 2026-11-15, EXP-033 until 2026-09-06) | EXP-013 | day-1 zero-shot on an unseen set + onboard-set fine-tune path; EXP-056 adds the oracle-text block (journal 2026-09-05/06) |
| deck | **EXP-126** | EXP-121 | same, for building |
| rebuild | **EXP-116** | — | only builder with `diffusion=True`; conditions on locks natively |

These are the `--models` / `--deck-models` defaults in `draftbot/hud/__main__.py`
(`PROD_DRAFT_MODELS` / `PROD_DECK_MODELS`). Change them there, not in docs.

**Best measured MSH accuracy — EXP-013**, test top-1 0.6756 / expert-subset
0.6985 (val 0.6778/0.7020): ModernDraftBot at small scale — decoder-only over
the 42 picks, set-attention pack encoder, pointer head, emb 128 / 4 layers
(~2M params), importance weighting on, priors off, **no hard skill filter**.
That last one was the ablation: soft weighting beat hard curation, and the
same lesson repeated on the deck side as EXP-105's failure. The scaled variant
EXP-023 (emb 256 / 6 layers) came in *below* it on test (0.6742) — scale did
not pay on the draft side either, a pattern now held across both models.

**Why EXP-033 ships anyway:** corpus-v2 trunk over 30 sets incl. post-promotion
MSH (Appendix-D refresh recipe). MSH week1 .6852exp / full .6962exp — same
class as EXP-013's .6985, and it is the one with *live* validation: Mark's real
MSH draft scored 93% top-1 agreement (39/42), every disagreement in the
pre-fix picks 1–4.

Both halves landed in the same place: a ~2M-param model at the practical
accuracy ceiling for imitation, with the corpus trunk as the production
artifact because generalization — not in-set accuracy — is what the extra
machinery buys.

**Two provenance gaps to close** (do not block shipping; PLAN §0 says
leaderboard rows are the source of truth):
1. EXP-033 has **no leaderboard row** — its MSH numbers live only in the
   2026-07-29 journal entry. Needs a `python -m draftbot.eval` row.
2. EXP-126 has **no MSH row** — only EOE (val full trophy-F1 0.8981, val none
   0.8347). If post-promotion MSH is in `decks_v2`, an MSH row would be
   contaminated and EOE is the honest generalization read; say so explicitly
   in the leaderboard so the gap does not read as an oversight.
   Sanity-checked at integration: on the real MSH pool EXP-126 returns a legal
   40 with the same 17 lands and same basics split as EXP-121, 20/21 mainboard
   overlap.


Audience: an agent implementing the deck-builder view in the HUD app.
Design spec: docs/DECKBUILDER_DESIGN_BRIEF.md (Claude Design round-trip).
Read CLAUDE.md + PLAN.md §0 first. Scope: `src/draftbot/hud/` ONLY — the
model/data layers are done and tested; do not modify `draftbot/{data,models,
train,eval}` or any test outside `tests/test_hud_*`.

## What exists (do not rebuild)

- **Models on disk** (gitignored; Mark's machine has them):
  - `checkpoints/EXP-127/` — production one-shot builder (val trophy-F1
    0.9011; corpus-pretrained trunk EXP-126 + MSH fine-tune). Use for the
    initial proposed build. (EXP-121 is the retired previous best.)
  - `checkpoints/EXP-116/` — diffusion variant: same quality class (0.8951)
    but supports **partial-deck conditioning** — use it for lock-and-rebuild.
- **Loading**: `draftbot.models.loading.deck_builder_from_checkpoint(ckpt_dir,
  set_code, stats) -> TorchDeckBuilder`. `stats` ∈ full|week1|none must match
  what the HUD's advisor already resolved for the set (day-0 → "none").
- **Inference contract**: `TorchDeckBuilder.build_all(arr: DeckArrays) ->
  [{"deck": {card_id: count}, "basics": ndarray(5)}]` — W/U/B/R/G order,
  always exactly 40 cards, deck ⊆ pool (tests/test_deck_model.py guarantees
  legality). Build a 1-row `DeckArrays` from the drafted pool:
  `draftbot.data.deck_dataset.DeckArrays` (pool_ids/pool_counts int16, PAD=-1,
  deck_counts/basics zeros, meta needs only a draft_id column).
- **Pool source**: `hud/state.py` tracks all 42 picks. The builder pool =
  the NONBASIC picks (drop ids where `cards.is_basic == 1` — drafted basics
  are not pool members; the basics head owns the manabase. See
  docs/DECK_DATA_PLAN.md "pool identity" note).
- **Membership confidences** for the boundary-zone UI: run the model forward
  directly (see `TorchDeckBuilder._heads`) and sigmoid the member logits;
  slot order matches your pool_ids row.
- **Server**: `hud/server.py` — stdlib HTTP, panel polls `/state` every
  500ms; `/art/<card_id>` serves cached Scryfall art. Extend `/state` with a
  `deck` section rather than adding a second poll loop.

## Lock-and-rebuild (the one genuinely new mechanic)

EXP-116 is `DeckBuilder(diffusion=True)`. Conditioning: build a `deck_state`
tensor (see `draftbot.models.builder.MASK_STATE`) — locked-IN card slots get
their count, locked-OUT get 0, everything else MASK_STATE — and drive
`TorchDeckBuilder._maskgit_probs`-style decoding from that initial state
instead of all-MASK. The current `_maskgit_probs` always starts fully masked:
add an optional `init_state=` parameter to it (this is the ONE permitted
core-code touch; keep it backward-compatible, default None = current
behavior, and extend `tests/test_deck_model.py` with a lock-respect test:
locked slots must appear/not-appear in the decoded 40 exactly as locked).

## Suggested flow

1. `state.py` detects pick 42 (or user opens the provisional view) → emits
   pool to a new `hud/deck.py` module.
2. `deck.py` holds both loaded builders, produces
   `{build, confidences, cuts, locks, model_id, stats_mode}` → merged into
   the server's `/state` payload.
3. Panel renders per the design spec; lock toggles POST to a new `/lock`
   endpoint; rebuild POSTs to `/rebuild` → EXP-116 path with the lock state.
4. Provisional mode (draft still live) reuses the same path with the partial
   pool — clearly badged, never stealing focus from the pick view.

## DoD

- Full replay of `tests/fixtures/real_draft_msh.log` through
  `python -m draftbot.hud --replay` ends in a proposed 40-card legal build
  rendered in the panel; lock 2 cards → rebuild honors both; `uv run pytest`
  green including the new lock-respect test; no regressions to the pick view
  (existing HUD tests untouched and passing).
- Journal entry (append-only) describing what shipped + a screenshot path.
- Do NOT add leaderboard rows (that is eval-harness territory) and do not
  touch `docs/PLAN.md` checkboxes — this is HUD_PLAN H5 work.
