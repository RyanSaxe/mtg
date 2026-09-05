# Development Plan: Modernizing the MTG Draft Transformer

**Audience:** an agentic coding loop (Claude Code or similar) executing autonomously, plus the human owner (Mark) reviewing gates.
**Origin:** this repo is a fork of RyanSaxe/mtg (the 2022 transformer draft AI, Mythic #27). We are evolving it into a modernized, larger, multi-set-pretrained drafter while preserving the original as reference.
**North star:** a model pretrained on many sets that drafts a *brand-new* set well on day 0–7 (card features only, or early card stats only), and reaches best-in-class accuracy on a target set after fine-tuning. Proven via a sandbagged-set protocol (§Phase 3).

---

## 0. Ground rules for the agentic loop

Read this section at the start of every session. It overrides convenience.

### Loop protocol
1. Read `docs/PLAN.md` (this file), `docs/JOURNAL.md` (append-only log), and `docs/LEADERBOARD.md`.
2. Work the **first unchecked task** in the current milestone unless the journal says otherwise. One task at a time; a task is done only when its **DoD (Definition of Done)** passes.
3. Before each training run: write a config YAML under `configs/`, assign an experiment ID `EXP-NNN`, log intent in the journal. After: append results to `docs/LEADERBOARD.md` (metrics + config path + git SHA + wall-clock).
4. Commit at task granularity with the task ID in the message (e.g. `[P1.T3] pointer-head eval harness`). Never commit `data/`, checkpoints, or wandb dirs (keep `.gitignore` current).
5. Update checkboxes in this file as tasks complete. Do not reorder or delete tasks; add new ones to the backlog section with rationale.

### Hard guardrails (never violate)
- **Quarantine:** the sandbagged set (initially **MSH**; promoted to corpus 2026-07-29 by owner ruling after the Phase-3 gate; **HOB** since 2026-09-05, its 17lands drop day — quarantined in pretrain_v2 / decks_v2 / sealed_v1) must NEVER appear in any pretraining corpus, warm-start source, feature-normalization statistics, or hyperparameter selection signal for the pretrained model. Enforced by `tests/test_quarantine.py` — that test failing is a stop-the-world event.
- **Split hygiene:** splits are by `draft_id`, persisted to disk once (`data/splits/<SET>.json`), and never regenerated. Test splits are evaluated only at declared gates, not during hill-climbing (use val).
- **Never edit a test fixture or golden file to make a failing test pass.** Fix the code, or escalate.
- **No metric laundering:** every leaderboard row must come from `python -m draftbot.eval` output, not hand-computed numbers.
- **Determinism:** every run has a seed in its config; log it. Eval is seed-free (full-split, no sampling).
- **Compute budget:** nothing longer than overnight (~10h) without explicit human sign-off in the journal. All runs >20min must checkpoint and be resumable (`--resume` restores model+optimizer+step).
- **Raw data is immutable.** Downloads go to `data/raw/` and are never modified; all transforms write to `data/processed/`. Disk budget ~25GB; ask before exceeding.

### Escalate to the human (stop and ask) when
- A phase gate fails twice with different remedies attempted.
- A guardrail conflicts with a task as written.
- You want >10h of compute or any paid/cloud resource.
- 17lands/Scryfall schema drift breaks assumptions recorded in `docs/data_cards/`.

### Environment (Phase 0 establishes this; recorded here for every session)
- Apple Silicon M4 Pro, 24GB unified RAM. Python **3.12** via `uv`; **PyTorch + MPS** is the framework (chosen over patching the 2022 TF/Keras-2 stack: healthier on Apple Silicon, torch.compile, standard LoRA/ecosystem tooling). `device="mps"`, fall back to CPU in CI/tests.
- Long runs: `caffeinate -is` wrapper, background with logging, monitor via tail on the run log.
- The original TF implementation stays untouched in `mtg/` as the reference implementation until P2 completes; new code lives in `src/draftbot/`.

---

## Phase 0 — Infrastructure, data plumbing, correctness fixes

Goal: a clean, tested data pipeline producing compact per-set shards; the known 2026 data-era bugs fixed; project scaffolding for everything after.

### Milestone M0.1 — Scaffolding
- [x] **P0.T1** `uv`-managed env; `pyproject.toml` (deps: torch, pyarrow, pandas, numpy, requests, pytest, pyyaml, tensorboard); `src/draftbot/` package skeleton (`data/`, `models/`, `train/`, `eval/`, `cli.py`); `tests/`; `configs/`; `docs/JOURNAL.md` + `docs/LEADERBOARD.md` seeded; `.gitignore` for data/checkpoints/logs. **DoD:** `uv run pytest` passes (≥1 trivial test); `uv run python -c "import torch; print(torch.backends.mps.is_available())"` → True.
- [x] **P0.T2** `CLAUDE.md` at repo root: 10-line orientation (what this repo is, where the plan lives, env commands, guardrail pointer). **DoD:** file exists, under 40 lines.

### Milestone M0.2 — Data acquisition & normalization
- [x] **P0.T3** Downloader: `draftbot.data.fetch --set MSH --event PremierDraft` → `data/raw/`, idempotent, resumable, probes S3 (known-good URL pattern: `https://17lands-public.s3.amazonaws.com/analysis_data/draft_data/draft_data_public.<SET>.<EVENT>.csv.gz`). Include `--probe-all` that regenerates `docs/data_cards/available.md` (28 sets confirmed as of 2026-07: STX…MSH, ~4.9GB gz total). **DoD:** MSH Premier + Trad downloaded; probe table written.
- [x] **P0.T4** CSV→Parquet converter with the **compact schema**: one row per pick — `draft_id, position (0..t-1), pack_cards (list<int16>, card ids), pick (int16), pick_2 (nullable), rank, user_win_rate, user_n_games (int16 — the 2026 data has buckets 500/1000; the old int8 loader overflows), event_wins, event_losses, draft_time`. Card ids come from a per-set `cards.parquet` vocab (see P0.T5). Pool is NOT stored (reconstructable from picks; verify in tests). Incomplete drafts (< t picks) dropped and counted. **DoD:** MSH parquet ≤ 400MB; round-trip test reconstructs a known draft's pool correctly; conversion streams (peak RSS < 8GB).
- [x] **P0.T5** Card table builder fixing the Scryfall breakage: for a set, fetch `set:<code>` **plus its companion sheets** (MSH requires union with `set:mar` — verified to cover all 339 CSV names), then **filter to exactly the names appearing in the CSV header** (drops the 36 non-draft extras; also basics get real vocab entries — MSH packs contain basics and 99,182 picks (3.4%) ARE basics, so basics are first-class draftable cards now; do not strip them). Companion-sheet mapping lives in a small registry dict (`msh: [mar]`, `bro: [brr]`, …) that's easy to extend. **DoD:** `cards.parquet` for MSH has exactly 339 rows, ids stable and persisted; test asserts every CSV `pick` value resolves to an id (run over the full file — zero KeyErrors).
- [x] **P0.T6** Card features: (a) **static features** from Scryfall (cmc, colors/pips, types, rarity, P/T, keywords, produced mana, flip) — always available, even day 0; (b) **stat features** from the 17lands ratings API (all 32 color filters, as the original did) — fetched with explicit `start_date/end_date` and saved as **immutable snapshots** `data/snapshots/<SET>/<start>_<end>.parquet` (the API supports date ranges; this is how we'll simulate week-1 knowledge honestly). Normalization: **z-score all continuous columns using train-corpus statistics only**, persist the scaler alongside; missing stats → 0 after scaling **plus a paired missing-mask feature** (no more silent fillna(0) meaning "0% win rate"). **DoD:** MSH full-season snapshot + a `release→release+7d` snapshot exist; scaler round-trips; unit test verifies masks flag genuinely-missing stats.
- [x] **P0.T7** Split builder: per set, deterministic `draft_id` split — train/val/test = 90/5/5 — plus a **temporal split variant** (train = first 80% of days, val/test = last 20%) for deployment-realistic eval. Persist to `data/splits/`. Also build the **expert eval subset** definition: test-split picks where `user_win_rate ≥ 0.62 AND user_n_games ≥ 100` (recompute thresholds if subset < 50k picks). **DoD:** split files exist for MSH; tests assert no draft_id crosses splits and splits are stable across reruns.
- [x] **P0.T8** Data card: `docs/data_cards/MSH.md` — rows, drafts (68,633 / 2,882,562 picks), t=42, basics-in-pack yes, companion sheet mar, quirks (`pick_2` empty), snapshot dates. Template reused for every set onboarded later. **DoD:** file complete; numbers generated by script, not typed.

**Phase 0 gate:** all tests green; MSH pipeline runs end-to-end from empty `data/` in one command (`make data-msh` or CLI equivalent) in < 30 min.

---

## Phase 1 — Baseline reproduction + the eval suite (the eval suite is the deliverable)

Goal: a faithful PyTorch port of Ryan's architecture trained on MSH, measured by an eval harness rich enough to drive every later decision. **Build the eval suite before training the model.**

### Milestone M1.1 — Eval & validation suite
- [x] **P1.T1** Heuristic baseline bots (each implements `pick(pack_ids, pool_ids, position) -> ranking`): `random`, `rarity-first`, `ALSA-greedy` (lowest avg-last-seen from snapshot), `GIH-greedy` (highest games-in-hand WR), `GIH-in-lane` (GIH restricted to pool's top-2 colors after pick 8). These are the floor every model must beat and the day-0/day-7 comparison points. **DoD:** all five run over the MSH test split in < 5 min combined.
- [x] **P1.T2** Eval harness `python -m draftbot.eval --model <ckpt|baseline-name> --set MSH --split val|test [--stats-snapshot week1|full|none]` producing a **scorecard JSON**: top-1/2/3 accuracy (overall, and sliced by pick position 0–41, by pack, by rarity of the human pick, by expert subset), NLL, ECE calibration, **rare-take rate vs human rare-take rate**, and pool-coherence (fraction of drafts where model's simulated picks end ≤ 2.3 colors — run greedy self-draft on 500 held-out pack sequences). Renders a compact Markdown summary for the leaderboard. **DoD:** scorecards generated for all 5 heuristic bots on MSH test; committed under `docs/scorecards/`.
- [x] **P1.T3** Behavioral fixture suite: ~30 hand-curated scenario tests in `tests/fixtures/scenarios/*.json` (pack + pool + expected-top-k-containment + rationale string). Cover: obvious bomb P1P1; committed 2-color pool at P3 → on-color common over off-color rare; late-wheel logic; fixing-land valuation for a splashy pool; do-not-take-basic-over-playable. Marked `xfail` until models exist; heuristic bots must pass the easy tier. **DoD:** suite runs in pytest; easy tier passes for GIH-greedy.
- [x] **P1.T4** Training-correctness tests (model-agnostic, reused for every architecture): (a) **causality test** — permuting future picks/packs must not change logits at step t; (b) **mask test** — cards absent from the pack must receive zero probability; (c) **overfit canary** — 100 drafts to >90% top-1 in < 5 min proves the wiring; (d) loss finiteness under empty-stat masks. **DoD:** tests parametrized to accept any model implementing the `DraftModel` protocol.
- [x] **P1.T5** Run infra: config-driven `python -m draftbot.train --config configs/EXP-001.yaml`; checkpoints every epoch + best-val; TensorBoard scalars; `--resume`; seed control; append-to-leaderboard helper. **DoD:** kill -9 mid-epoch then `--resume` reproduces within-tolerance final metrics.

### Milestone M1.2 — Baseline model
- [x] **P1.T6** Port the original architecture to PyTorch as `LegacyDraftBot`: encoder–decoder over t=42, mean-pooled pack embeddings, ConcatEmbedding (one-hot half + feature-MLP half), MLP output head masked to pack, CE loss + triplet embedding loss + rare/cmc priors, importance weighting (rank×WR×recency×position, per Ryan's `importance_weighting`), inverse-sqrt warmup. Basics included in vocab (modern necessity — see P0.T5). **DoD:** P1.T4 tests pass; params ≈ 2M at emb 128 / 2+2 layers.
- [x] **P1.T7** Train **EXP-001** (baseline) on MSH: batch 64 drafts, ~20 epochs, early-stop on val top-1 (patience 3). **DoD:** scorecard on val; must beat GIH-greedy top-1 by ≥ 5 points; leaderboard row added. If it fails, debug against P1.T4 before touching hyperparameters.
- [x] **P1.T8** Declare the **test-split gate**: run EXP-001 + all heuristics on MSH test once; freeze these as the Phase-1 reference scorecards. **DoD:** `docs/scorecards/phase1/` committed; journal entry with analysis (which pick positions are weakest, expert-subset gap, rare-take delta).

**Phase 1 gate:** baseline beats all heuristic bots on test top-1; eval suite + behavioral fixtures run end-to-end in < 15 min; everything on the leaderboard.

---

## Phase 2 — Modernized architecture (single-set hill-climb on MSH)

Goal: replace the 2017-vintage design with the modern one, ablating each change on val. Target: decisively beat EXP-001.

### Architecture v2 (`ModernDraftBot`) — the design to implement
- **Decoder-only.** One causal stack over 42 steps. Per-step input token = fusion (concat→proj) of: pack summary, previous-pick embedding, learned position embedding. No cross-attention (the streams are position-aligned; the enc/dec split in the original bought nothing).
- **Pack set-encoder instead of mean-pooling:** small permutation-invariant attention block (2-layer set attention / PMA) over the ≤15 card embeddings in the pack → pack summary vector. Cards as tokens, identity preserved.
- **Pointer head instead of dense-to-vocab:** score each in-pack card by dot product between the decoder state and that card's (projected) embedding; softmax over pack only. Parameter-light, set-size agnostic, and it makes the triplet loss redundant (drop it).
- **Modern internals:** Pre-LN (RMSNorm), SwiGLU FFN, no post-hoc softmax (CE from logits), AdamW + cosine schedule w/ short warmup, label smoothing 0.05, grad clip 1.0, bf16 autocast on MPS, batch 64–128 drafts.
- **Efficiency:** packs stored/consumed as ≤15 indices (gather), never the (B,42,n_cards,emb) broadcast.
- **Skill curation replaces behavioral priors as the primary lever:** train on picks with `user_win_rate ≥ 0.54 AND user_n_games ≥ 50` (keep ~top-60% of data), retain soft importance weighting (recency + skill) on top. Rare/cmc priors become config-optional, default **off**; the rare-take-rate eval metric is the watchdog instead.

### Tasks
- [x] **P2.T1** Implement `ModernDraftBot` behind the same `DraftModel` protocol; all P1.T4 tests pass. **DoD:** overfit canary < 3 min.
- [x] **P2.T2** **EXP-010** v2 at ~2M params (parity scale): same data, compare to EXP-001 on val. **DoD:** leaderboard row; if not ≥ baseline, ablate fusion/pointer/set-encoder individually before proceeding.
- [x] **P2.T3** Ablation ladder (one EXP each, val-only, ≤ 2h/run): (a) pointer head vs dense head; (b) set-encoder vs mean pool; (c) skill filter vs priors-on; (d) label smoothing 0 / 0.05 / 0.1; (e) Premier+Trad merged with event-type flag. Keep winners. **DoD:** each ablation is a leaderboard row with a one-line conclusion in the journal.
- [x] **P2.T4** Scale on MSH only: emb 256, 6 layers, 8 heads (~15–20M params). Watch val for overfit; if data-limited, stop scaling — that's the signal Phase 3 is where growth must come from. **DoD:** best-MSH-only model declared (**EXP-02x**), test-gate scorecard frozen.
- [x] **P2.T5** Behavioral fixture pass: v2 best must pass the full easy+medium tiers of P1.T3. Failures become journal analysis (and possibly new training-data curation ideas), not fixture edits.

**Phase 2 gate:** v2-best beats EXP-001 by ≥ 2 points test top-1 AND passes fixtures ≥ baseline AND rare-take-rate within 1.5× of human expert rate. Freeze scorecards in `docs/scorecards/phase2/`.

---

## Phase 3 — Multi-set pretraining + the sandbagged-MSH protocol

Goal: prove a large model pretrained on many sets drafts a **never-seen set** well with card features only (day 0) or an early stats snapshot (day 7). MSH is the sandbag: quarantined from all training until the gate passes.

### Design
- **Set-agnostic card identity:** card encoder = MLP over [static Scryfall features ∥ stat features ∥ missing-masks]; **no per-set one-hot in the pretrained trunk**. Optional small learned per-set card-embedding table exists but is **zero-initialized and only trained during fine-tuning** (so zero-shot = pure feature pathway).
- **Stat-dropout:** during pretraining, with p≈0.25 per draft, mask ALL stat features (masks flip on) — the model must learn to draft from static card properties alone. This single trick is what makes day-0 inference a trained-for condition instead of out-of-distribution.
- **Set conditioning:** a set-summary token prepended to the sequence (mean of the set's card feature vectors — captures format speed/color balance without a learned per-set lookup that would break zero-shot), plus event-type flag.
- **Heterogeneous t:** pad to max t across corpus, mask. Per-set data cards record each t.
- **Corpus:** all available sets EXCEPT the quarantine list. Cap per-set drafts (e.g. 40k) to bound epoch cost and balance eras; weight play-booster-era sets 2× (rules environment matches the future). Manifest = `configs/corpus/pretrain_v1.yaml`, consumed by the quarantine test.
- **Stats-leakage discipline:** pretraining uses each set's **full-season snapshot** (fine — those sets are long-solved), but feature scalers are fit on pretrain corpus only. The sandbag's snapshots are used at **eval time only**.

### Tasks
- [x] **P3.T1** Onboard the corpus: run the Phase-0 pipeline for all non-quarantined sets (start with the ~10 play-booster-era sets, then extend). Each gets a data card; each needs its companion-sheet registry entry verified (the P0.T5 zero-KeyError test is the acceptance check per set). **DoD:** ≥ 20 sets converted; `tests/test_quarantine.py` asserts MSH absent from every corpus manifest.
- [x] **P3.T2** Streaming multi-set dataloader (pyarrow, per-set shards, shuffle buffer; RAM ceiling 10GB). **DoD:** iterates full corpus < 20 min/epoch-equivalent without training; RSS test.
- [x] **P3.T3** Warm-start pilot (cheap transfer test, do FIRST): train v2 on one prior set (e.g. EOE), fine-tune on MSH, compare to MSH-from-scratch at equal total compute. **DoD:** journal verdict on transfer strength; if fine-tuned ≤ scratch, investigate feature-pathway bugs before any big pretrain.
- [x] **P3.T4** **EXP-030** pretrain-small: v2-feature-based, ~15M params, 4–6 sets, overnight. Zero-shot eval on MSH val in all three stat modes (`none`, `week1`, `full`). **DoD:** zero-shot(week1) beats ALSA-greedy; zero-shot(none) beats rarity-first by ≥ 10 points. These are sanity bars, not the real gate.
- [ ] **P3.T5** **Hill-climb loop on zero-shot MSH** (this is the core experimental phase — iterate): propose one recipe change → run at pilot scale (≤ 2h) → if val zero-shot improves ≥ 0.5 top-1, promote to overnight scale. Candidate levers, in priority order: stat-dropout rate; corpus size/mix/era-weighting; set-summary conditioning variants; model scale (15M → 30M → 50M); oracle-text embeddings appended to static features (frozen sentence-transformer, cached per card — the ceiling-raiser for day-0 since names/stats can't carry mechanics like "Ward 2" interactions); curriculum (pretrain all-era → continue on play-booster-era only). **Rules:** one lever per EXP; every EXP on the leaderboard; val only. **DoD:** documented plateau — 3 consecutive promoted-scale attempts < 0.3 improvement.
  - [ ] **P3.T5a** training length: EXP-034 (60k steps, HOB tracker) — launched 2026-09-05.
  - [ ] **P3.T5b** curation strength ladder: EXP-035 (reject bottom of barrel: WR ≥ .46 / ≥ 10 games) → EXP-037 (no filter). Owner ruling 2026-09-05.
  - [ ] **P3.T5c** oracle-text embeddings + stat-reconstruction aux head: proposal `docs/TEXT_EMBED_PLAN.md`; EXP-050/051/052 (+ EXP-139 decks).
- [ ] **P3.T6** Few-shot curves: fine-tune pretrain-best on 0 / 1k / 5k / 20k / all MSH drafts (train split only); plot top-1 vs data. This is the "early-days trajectory" evidence — how good is the bot on day 3, week 2, week 6 of a new format. **DoD:** curve committed to `docs/scorecards/phase3/`; compare vs MSH-only v2-best at each data budget.

**Phase 3 gate (the sandbag promotion test — run on MSH *test* split, once):**
1. Zero-shot(week1 stats) ≥ GIH-greedy(week1 stats), and
2. Zero-shot(week1) within 6 points of the Phase-2 MSH-only best, and
3. Fine-tuned-on-full-MSH ≥ Phase-2 best (pretraining must not hurt the data-rich case), and
4. Behavioral fixtures: zero-shot passes the easy tier fully.

**On pass:** journal entry declaring success; **MSH exits quarantine** and joins the corpus; the next Arena release becomes the new sandbag (update quarantine list + this file). On next-set release day, the day-0 claim gets tested for real — that's the ultimate eval.

---

## Phase 4 — Fine-tuning machinery & per-set adapters (LoRA)

Goal: make "new set support" a one-command, one-hour operation, and verify parameter-efficient tuning matches full fine-tuning.

- [x] **P4.T1** LoRA implementation (attention qkv/o + FFN, rank 8–32) plus always-full-rank per-set card-embedding table and pointer projection. Config-switchable: full-FT vs LoRA vs head-only. **DoD:** LoRA run touches < 5% of params; checkpoint stores adapter separately from base.
- [x] **P4.T2** Equivalence study on the (now unquarantined) MSH: full-FT vs LoRA vs head-only at 3 data budgets. **DoD:** leaderboard rows; decision recorded (expected: LoRA ≈ full-FT; if not, prefer full-FT — it's cheap at this scale and the base is reusable either way).
- [x] **P4.T3** `draftbot onboard-set <CODE>` — end-to-end: probe data, build cards/features/snapshots/splits/data card, evaluate zero-shot, fine-tune adapter when data exists, emit scorecard. **DoD:** dry-run on an existing set from scratch completes unattended.
- [x] **P4.T4** Continual-pretrain refresh recipe: how/when the base model absorbs newly unquarantined sets (scheduled full pretrain rerun, not incremental training, to avoid drift). **DoD:** documented in this file + one executed refresh.

**Phase 4 gate:** onboarding a set = one command; adapter-vs-full decision documented with data.

---

## Phase 5 — Stretch (only after P4 gate; each needs a journal proposal first)

### P5.T1 Deckbuilder — expanded 2026-07-29 by owner directive into a hill-climb program

Run this like Phases 1–3: eval suite first, baselines as the floor, a measured
human ceiling as the target, then one-lever-per-EXP hill-climbing. Data layer
already done (docs/DECK_DATA_PLAN.md, all 5 gate tests green). Deck EXPs are
numbered **EXP-1NN** (own leaderboard section). Stat mode for stage 1 is `full`
(MSH is long-solved); a `none`-stats builder is a hill-climb lever, not the start.

**Deck metrics (canonical for this phase):**
- **deck-F1**: per-build multiset F1 between predicted and actual 40-card
  maindeck **including basics** (primary); nonbasic-only F1 reported alongside.
- **trophy-F1** (v2, owner refinement 2026-07-30): take pools that DID trophy
  (some build reached ≥5 wins) and score against the specific build that won
  (winningest per draft; ties → most played). 7-win slice reported too.
  v1 (most-played build, filtered ≥5 wins) retired 2026-07-30.
- **winner-pref** (owner metric 2026-07-30): among drafts holding a ≥5-win
  build AND a strictly-worse build of the same pool, fraction where the
  model's deck is strictly closer (full-deck F1) to the build that won;
  ties 0.5. The deterministic stand-in for "would this deck win" — a learned
  P(≥5 wins) judge is a separate proposal (confounding; see P5.T3 caveats).
- **lands-MAE**: |predicted total lands − actual| (basics + nonbasic lands).
- **basics-L1**: L1 distance between predicted and actual W/U/B/R/G counts.
- **curve-L1**: L1 between cmc histograms (0–7+, nonbasic spells only).
- Eval protocol: input = nonbasic pool (deck ∪ side from decks.parquet);
  output = a complete legal 40-card build (legality asserted, not scored).
  Eval builds = most-played build per draft; train uses all builds. Weighting
  per DECK_DATA_PLAN curation: `(1 + n_wins) · soft_skill(user_win_rate)`.

### Milestone M5.1 — Deck-eval suite (the eval is the deliverable, again)
- [x] **P5.T1a** Deck eval harness: `python -m draftbot.eval --deck --model <ckpt|baseline> --set MSH --split val|test` → scorecard JSON + md under `docs/scorecards/`, leaderboard-row helper (deck table). Slices: F1 by n_wins (0–7+), by user-WR bucket, by deck archetype colors. **DoD:** scorecards for all deck baselines on MSH val committed.
- [x] **P5.T1b** Baselines: `random-legal` (floor), `gih-top23` (best-rated 23 nonbasics + pip-split basics), `gih-in-lane-build` (best 2-color lane by pool GIH mass, top 23 in-lane + fixing, pip-split basics — this is ~the HUD heuristic that misfired). **DoD:** all three run over MSH val in < 2 min.
- [x] **P5.T1c** Ceiling study: human rebuild self-agreement — deck-F1 between builds of the same draft (pairs from multi-build drafts, weighted toward played builds), overall and for ≥5-win drafts. This number is the "very close to upper echelon" target; journal entry declares the phase gate threshold from it. **DoD:** computed by script into the journal + scorecard-style JSON in `docs/scorecards/phase5/`.

### Milestone M5.2 — Stage-1 model (per-card membership; journal design note 2026-07-29)
- [x] **P5.T1d** Deck dataset loader (`deck_dataset.py`: pool/deck arrays + weights + split join), training loop `python -m draftbot.train.decks --config configs/EXP-1NN.yaml` (checkpoint/resume/seed per §0), correctness tests: pool-permutation invariance, pad-mask inertness, decode legality (always exactly 40, deck ⊆ pool), overfit canary (< 5 min), loss finiteness under missing-stat masks. **DoD:** tests green.
- [x] **P5.T1e** **EXP-101**: train the existing `DeckBuilder` (models/builder.py: set-attention trunk; membership + land-count + basics heads) with win-weighted BCE. **DoD:** beats all baselines on val deck-F1 AND trophy-F1; leaderboard rows; journal analysis of where it loses (wins-slice, land counts, splash handling).
### Milestone M5.3 — Hill-climb loop (auto-research; mirrors P3.T5 rules)
- [x] **P5.T1f** Iterate: propose one lever → EXP-1NN at ≤30 min scale → promote if val deck-F1 +≥0.5pt (or trophy-F1 +≥0.5pt at neutral deck-F1). Candidate levers in priority order: weighting variants (uniform vs win-weighted vs hard ≥N-win curation); land-count/basics decode variants (argmax vs expected-count); model scale (emb 256 / more blocks); stats `none` (day-0 builder); auxiliary `main_colors` head; joint decode (re-score after partial commitment — cheap stage-2 preview). **Rules:** one lever per EXP; every EXP a leaderboard row + one-line journal conclusion; val only. **DoD:** documented plateau (3 consecutive attempts < 0.2pt).
- [ ] **P5.T1g** Phase-5 gate, declared once on test: trophy-F1 within 1.0pt of the P5.T1c human rebuild ceiling (or above it), lands-MAE ≤ 0.8, fixture-style spot checks pass (no off-color tapland in a 2-color deck across 200 sampled val builds — the Subterranean Cavern regression check). Freeze `docs/scorecards/phase5/`; journal verdict. Stage-2 (masked discrete diffusion with partial-deck conditioning) gets its own proposal after this gate.
### P5.S Sealed deck builder — owner directive 2026-07-31 (live TLA event); plan in docs/SEALED_PLAN.md

- [x] **P5.S1** Sealed data layer as a pure ADDITION (`source="sealed"`, draft defaults untouched): extractor `--event sealed` (Sealed+TradSealed → decks.sealed.parquet, curation logged), per-set `<SET>.sealed.json` splits (persisted once), 5-test sealed gate (tests/test_sealed_decks.py). **DoD:** gate green on every extracted set; full draft suite untouched and green.
- [x] **P5.S2** TLA fast path: S2a controls (ceiling + baselines + EXP-126 zero-shot), EXP-130 fine-tune, `python -m draftbot.build` pool-paste CLI. **DoD:** owner can build for the live event; leaderboard sealed section.
- [x] **P5.S3** Sealed corpus: 27-set sweep (~394k builds), sealed_v1 manifest (EOE holdout), EXP-131 (warm-start from EXP-126) vs EXP-132 (scratch) pretrains, holdout day-0 rows, TLA comparison, production pick. **DoD:** leaderboard rows + journal verdict.
- [x] **P5.S5** Joint draft+sealed trunk with format token (owner directive 2026-08-01): EXP-135 joint pretrain (draft preserved ≤0.2pt, TLA sealed 0.7576 best-of-program), EXP-136/137 annealing probes (win_skill: +7-win-F1 signal, flat trophy — filed). Production: EXP-135 in CLI + bundle. **DoD:** leaderboard rows + journal verdict. ✓
- [x] **P5.S4** Sealed mode in the HUD (2026-08-01, live-validated on the owner's ArenaDirect TLA pool): SealedPool from the Course/CardPool log messages, SealedState, EXP-135 with the sealed format token, locks via probability pinning; sanitized real capture as fixture. **DoD:** replay of tests/fixtures/real_sealed_tla.log renders a legal 40; lock-respect test; draft suite untouched. ✓

- [ ] **P5.T2** Draft-table self-play sim (8 bots) for qualitative eval + fixture generation; compare bot-table pick orders to human ALSA.
- [ ] **P5.T3** Win-rate-aware objective: auxiliary head predicting event wins from the evolving pool; investigate reweighting picks by outcome advantage (careful: heavy confounding — proposal must address it).
- [ ] **P5.T4** Live-draft inference bridge (the MTGA_Draft_17Lands overlay protocol) for actually using the bot in Arena drafts.

---

## Appendix A — Metrics definitions (canonical, do not fork)
- **top-k:** human pick ∈ model's top-k of in-pack ranking. Primary: top-1 on test split; expert-subset top-1 is the headline number (agreement with skilled humans, not average humans).
- **NLL:** mean cross-entropy over in-pack softmax.
- **ECE:** 10-bin expected calibration error of the top-1 probability.
- **Rare-take rate:** fraction of picks where model's top-1 is rare/mythic when a non-rare human pick existed; report ratio vs expert-human rate.
- **Pool coherence:** self-draft 500 test pack-sequences greedily; mean distinct colors among ≥8-pip colors in final 45 cards; humans ≈ 2.0–2.3.
- **Stat modes:** `none` = static features only, masks on; `week1` = release→release+7d snapshot; `full` = full-season snapshot.

## Appendix B — Known facts & landmines (verified 2026-07-28)
- MSH: 68,633 drafts / 2,882,562 picks / t=42; ranks lowercase; `pick_2` empty; basics ARE picked (99,182 times); companion sheet = `mar` (58 names only there); 36 Scryfall names never in packs; `user_n_games_bucket` ∈ {1,5,10,50,100,500,1000} (int8 overflows); ratings API works without special headers and supports `start_date/end_date`.
- Scryfall `is:booster` returns 0 cards for play-booster-era sets — never rely on it.
- 17lands S3 has PremierDraft data for ~28 sets (STX→MSH); TradDraft spotty; replay data not at the public path.
- Original-repo reference behaviors worth preserving as options: importance weighting formula, PxP1 dampening rationale, anti-rare prior (off by default in v2, but the eval watchdog exists because the failure mode is real).
- 24GB unified RAM: keep dataloader RSS < 10GB, batch activations < 4GB; bf16 on MPS; if MPS op gaps bite, fall back per-op via `PYTORCH_ENABLE_MPS_FALLBACK=1` and note it in the journal.

## Appendix C — Backlog (ideas with rationale; promote via journal entry, never work directly)
- Wheel-prediction auxiliary loss (encoder's old job, now an aux head): may sharpen signal-reading.
- Distillation of the big pretrained model into a ≤5M-param on-device model for the live overlay.
- Rating-free stat features derived from raw game_data (compute GIH-WR ourselves with date cutoffs) — removes dependence on the ratings API and enables arbitrary-date snapshots for any set.
- Human-disagreement ceiling study: estimate top-1 noise floor from duplicate pack states across drafts, to contextualize plateaus.

## Appendix D — Continual-pretrain refresh recipe (P4.T4, executed 2026-07-29)

When a sandbagged set passes its gate and joins the corpus (or ~quarterly
otherwise), refresh the base trunk by a FULL pretrain rerun — never incremental
training on top of the old trunk (drift + scaler staleness):
1. New corpus manifest `configs/corpus/pretrain_vN.yaml` (previous one frozen —
   it documents what the old trunk trained under). Newly promoted set gets
   play-booster weight 2.0; new sandbag in `quarantine` from announcement day.
2. `uv run python -m draftbot.data.onboard --corpus configs/corpus/pretrain_vN.yaml --parallel 5`
   for any not-yet-processed sets.
3. New EXP: `python -m draftbot.train.pretrain --config configs/EXP-0NN.yaml`
   (corpus: vN). Scaler refits over the vN corpus automatically; zero-shot
   tracking runs against the CURRENT sandbag once it has val data (until then,
   the most recent promoted set is the tracking proxy).
4. Gate: new trunk's zero-shot on the tracking set must be ≥ old trunk's, and a
   spot fine-tune must be ≥ the old trunk's. Then repoint `--base` defaults /
   onboard-set to the new trunk directory.
Executed refresh: EXP-033 = pretrain on pretrain_v2 (MSH in-corpus), tracking
proxy MSH (its val is no longer quarantined — it is corpus now).
