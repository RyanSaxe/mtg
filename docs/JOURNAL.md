# Journal (append-only)

Format: `## YYYY-MM-DD — <entry title>` followed by prose. Never edit past entries.

## 2026-07-28 — Session start: enacting docs/PLAN.md from P0.T1

Owner directive for this session (Mark): deliver BOTH tryable models — (1) the modernized
single-set model trained on MSH only (Phase 2), and (2) the pretrained/general model
fine-tuned on MSH (Phase 3). Target set is MSH (confirmed; it is the currently
available Arena set). Additional owner requirement: **validate that models agree with
good drafters, not bad drafters** — implemented as a first-class eval metric: the
scorecard reports top-1 agreement on the expert subset (WR ≥ 0.62, ≥ 100 games) AND on
a low-skill subset (bottom tail by user_win_rate), plus the gap. A model that mimics
skilled play must show expert-agreement > low-skill-agreement; the gap must not be
worse than GIH-greedy's. Added to P1.T2 scope (journal-sanctioned scope addition, no
plan tasks reordered).

Environment: uv 0.11.19 present, system python 3.14 (project pins 3.12 via uv).
Starting P0.T1 scaffolding.

## 2026-07-28 — 17lands ratings-API drift; backlog item promoted into P0.T6

The anonymous `card_ratings/data` API no longer serves win rates (all null) and
ignores `start_date`/`end_date` (verified on MSH, FIN, EOE, TDM; counts also ~100×
too small). This breaks the original stats pathway recorded in PLAN Appendix B.
Remedy: promoted the backlog item **"rating-free stat features derived from raw
game_data"** into P0.T6. Snapshots are now computed from the public
`game_data_public.<SET>.<EVENT>.csv.gz` + our converted draft parquet:
ALSA/ATA/seen/pick counts from draft data; GP/OH/GD/GIH/GNS win rates + IWD from
game data, overall + all 31 exact-deck-color filters; win rates NaN (masked) under
5 games. Date windows are exact (game_time date ∈ [start, end]), which makes the
week-1 "day-7 knowledge" snapshots *more* honest than the API ever was. The two
API-fetched snapshot files (null win rates, useless) were deleted and rebuilt via
`features.build_snapshot`; on-disk contract unchanged. Note for future sets:
game_data must be fetched alongside draft_data (fetch.py now takes kind="game").

## 2026-07-28 — Phase 0 complete

All P0 tasks done, 12 tests green. MSH: 68,609 complete drafts / 2,881,578 picks,
t=42, vocab 339 (281 msh + 58 mar), Premier parquet 231MB. Snapshots computed from
raw data: full (2026-06-26→2026-07-28) and week1 (→2026-07-03); 576-col API shape
replaced by 356 computed columns (draft-stage stats unsuffixed once, game-stage
×32 filters). Expert subset auto-relaxed to WR≥0.58/≥50 games; low-skill
WR≤0.50/≥50 games (both ≥50k test picks). `make data SET=<code>` is the
one-command pipeline. 24 incomplete drafts dropped (Premier), 1 (Trad).

## 2026-07-28 — Phase 1 complete: eval suite + EXP-001 baseline

Heuristic floor (MSH test, full stats): random .232 / rarity-first .327 /
gih-greedy .379 / gih-in-lane .424 / alsa-greedy .467 top-1. Skill-gap metric
behaves as designed: gih bots agree with experts >> low-skill (+.06), rarity-first
gap NEGATIVE (-.007) — the metric catches bad-drafter mimicry.

EXP-001 (legacy port, 1.68M params, 6 epochs / 2.7 min on MPS): val .576, test
.574 top-1 (beats every baseline; +19.5 over gih-greedy, +10.6 over alsa-greedy).
Expert top-1 .598, skill-gap +.039 (well above gih-greedy bar). ECE .034.
Weak spots for Phase 2: rare-take 0.38× expert (the λ=10 anti-rare prior
overshoots — v2 drops priors for skill curation + watchdog); pool coherence 1.81
colors vs human 2.16 (slightly over-committed); pack-1 positions 0-3 weakest
early-pick accuracy. Behavioral fixtures: 21 scenarios (10 easy all pass for
gih-greedy; medium/hard reserved for trained models). Analysis per position shows
P1P1 .57 rising to ~.60+ late-pack (wheel picks easier).

## 2026-07-28 — EXP-001 pathology: priors + basics-in-vocab = P1P1 basic-taking

The demo CLI surfaced it; quantified on val: EXP-001 top-1 is a basic land in
24.9% of P1P1s (humans: 0.0%), 18.2% vs human 6.9% when a pack contains a basic.
Cause: the legacy cmc prior (hinge toward cheap picks) + rare prior both point at
basics (cmc 0, never rare) under uncertainty; the 2022 original excluded basics
from the vocab so this couldn't happen. Kept as-is for baseline fidelity —
EXP-001 is the reproduction, priors on. Phase-2 v2 (priors off, skill curation)
must clear the easy-no-basic-over-playable fixture EXP-001 fails. Watchdog for
this now exists in the demo + this probe.

## 2026-07-29 — EXP-010: v2 architecture validates decisively

val top-1 .6725 vs EXP-001 .5758 (+9.7). Expert .6977, skill-gap +.041, ECE .007,
rare-take 1.12× expert (priors OFF — watchdog healthy), pool coherence 2.07
(human 2.14), P1P1 basic pathology eliminated (0.0%; overall 3.3% ≈ human).
All 21 behavioral fixtures pass (easy+medium+hard) → P2.T5 done early.
1.27M params, 11 epochs / 10.4 min (early stop @8). Skill filter kept 28,618/61,573
drafts (46%). Ablation chain EXP-011..014 + EXP-020 scale-up running.
Corpus: all 29 non-MSH sets downloaded (6.7GB); onboarding in progress —
MKM needed PLST (The List) companion sheet, registry extended, politeness sleeps
raised after Scryfall 429s.

## 2026-07-29 — Owner directive: expert agreement is THE signal

Mark (citing Ryan Saxe's original intent): the target signal is "what good
drafters draft", not the average 17lands user. Changes:
1. Leaderboard ranking + Phase-2/3 winner declarations use expert-subset top-1
   (evaluation-side; uniform across all runs, past and future).
2. Early stopping / best-checkpoint selection now tracks expert-subset val
   top-1 (train/loop.py) — applies from EXP-014 onward; EXP-010..013 selected on
   overall val top-1 (metrics move together; scorecard comparison unaffected).
3. Pretrain zero-shot tracker selects best.pt on expert-subset week1 MSH-val.
EXP-013 (no skill filter) expected to look competitive on overall top-1 but
worse on expert agreement — that's the point of running it.

## 2026-07-29 — Design note for P5.T1 (deckbuilder), owner discussion

Mark floated diffusion; agreed direction: **masked discrete diffusion /
MaskGIT-style iterative infilling** over the deck set — the principled version of
the 2022 DAE (its sample-N-out corruption IS absorbing-state diffusion; its
iterative argmax IS single-path denoising). Plan: (1) one-shot baseline first:
set-transformer pool encoder (warm-start from draft trunk card encoder) +
per-card maindeck head + land-count head, trained on skill-curated winning
builds from game_data; (2) masked-infilling model with confidence-based commit
(5-10 steps), which gives partial-deck conditioning (lock cards, complete rest)
for free. Avoid continuous diffusion (discrete-rounding pain, no benefit at
40-slot scale). Eval: win-rate-weighted agreement (many builds co-optimal) +
curve/land sanity; expert builds are the reference, per the expert-signal
directive. AR pointer decoder rejected for decks: imposes order on a set.

## 2026-07-29 — Ablation ladder results (P2.T3): the skill-filter surprise

Ranked by expert-subset val top-1 (owner's headline metric):
EXP-013 no-hard-filter .7020 > EXP-014 ls0 .6994 > EXP-011 dense .6987 >
EXP-010 recipe .6977 > EXP-012 meanpool .6976.
Conclusions, one line each:
- (c) HARD skill filter is a net LOSS (-0.4pt expert): soft importance weighting
  already downweights weak drafters; the filter just burns 53% of data. Both
  owner and agent predicted the opposite — data > purity at this scale. KEEP
  soft weighting, DROP hard filter for the scale-up winner combo.
- (a) pointer vs dense head: tie (dense +0.1pt, noise). Pointer retained —
  required for Phase-3 vocab-agnostic transfer.
- (b) set-encoder vs mean pool: tie. Set encoder retained (cheap; Phase-3
  pack-conditioning rationale) but not a proven win at 1.3M single-set scale.
- (d) label smoothing 0 vs 0.05: tie on accuracy; 0.05 clearly better ECE
  (.007 vs .030). Keep 0.05.
- (e) Premier+Trad merge: NOT RUN this session (loader change); backlog.
EXP-020 (big, filtered) in flight; EXP-023 (big, no filter) queued after — the
two candidates for Phase-2 best.

## 2026-07-29 — PHASE 2 GATE PASSED; EXP-013 declared Phase-2 best

Test gate (run once, frozen in docs/scorecards/phase2/): EXP-013 test top-1
.6756 / expert .6985 — beats EXP-001 (.5736) by +10.2 (gate: ≥2) ✓; fixtures
21/21 ✓; rare-take 1.08× expert (≤1.5×) ✓; coherence 2.06 (human 2.16).
EXP-020 (8.5M filtered) .6902 expert and EXP-023 (8.5M unfiltered) .6962 expert
both UNDER the 1.3M EXP-013 → single-set MSH is data-limited; growth must come
from pretraining (the plan's predicted signal — P2.T4 conclusion).
Deliverable #1 for Mark = checkpoints/EXP-013 (demo: --models checkpoints/EXP-013).
Note: ceiling analysis (calibrated-confidence proxy) puts expert-agreement
ceiling in low-to-mid .70s; EXP-013 at .6985 is within ~2-4pts.

## 2026-07-29 — P3.T3 warm-start pilot: transfer VERDICT POSITIVE

EXP-021 (v2 on EOE, 42.3k skill-filtered drafts): EOE expert val .7040. EOE
quirk found: t=39 (13-pick packs); loading.py now infers t from checkpoint.
EXP-022 (EOE trunk → MSH fine-tune, 11 epochs, EOE scaler): MSH expert val
.6995 vs EXP-010 from-scratch .6977 at equal epochs — modest final gain, much
faster convergence (epoch 1: .6895 vs .6423 overall). Feature pathway confirmed
working cross-set → proceed to EXP-030 corpus pretrain (queued).
Also fixed this session: 4/4.07M EOE picks were 17lands glitch rows (pick not
in pack) inflating printed loss ×600 via the -1e9 mask; loader now drops such
drafts globally. Val metrics were never affected.

## 2026-07-29 — EXP-030 pretrain complete: zero-shot MSH is REAL

7.7M feature-only trunk + set token, 29 sets (2021 quartet kept via
no-skill-metadata fallback), stat-dropout 0.25, 20k steps / 98 min.
Zero-shot MSH val (never trained on MSH): none .5304/.5257exp,
week1 .6514/.6719exp, full .6587/.6845exp.
P3.T4 sanity bars demolished: zs(week1) vs ALSA-greedy +19pts; zs(none) vs
rarity-first +20pts. Phase-3 gate conditions 1-2 comfortably met on val
(GIH-greedy(week1) val = .3804/.4198exp; margin +27pts; gap to Phase-2 best
2.7pts < 6). Day-0 mode .53 = better than any stats-armed heuristic bot.
EXP-031 fine-tune launching via chain.

## 2026-07-29 — Phase-3 gate: 3 of 4 conditions pass; condition 3 = statistical tie

MSH TEST (declared gate run): (1) zs(week1) .6486 vs gih-greedy(week1) .3793
✓ +27pts; (2) gap to Phase-2 best .6756−.6486 = 2.7pts < 6 ✓; (4) zero-shot
easy fixtures 10/10 ✓. (3) EXP-031 FT .6752 vs EXP-013 .6756 — 0.04pt under
(≈0.3 SE on 154k picks; VAL had EXP-031 ahead +0.27pt expert). Reading: tie
within noise, not "pretraining hurts", but the gate is strict — one remedy
sanctioned: EXP-032 (gentler FT: peak_lr 7e-5, warmup 200, patience 5).
If it also lands under, escalate per protocol.
Zero-shot day-0 test .5239 with skill-gap −.011 (without stats the trunk can't
fully separate expert taste — noted for P3.T5 oracle-text lever).

## 2026-07-29 — Phase-3 gate condition 3: second attempt also ties → ESCALATED

EXP-032 (gentler FT) test: .6752/.6972exp — identical to EXP-031 (.6752/.6973).
Robust result: the pretrained+fine-tuned model TIES the MSH-only best
(Δ −0.0004 overall, −0.0013 expert; both under 1 SE). Val ordering favored the
fine-tunes (.7047/.7043 vs .7020) — test flip is textbook split noise.
"Pretraining must not hurt the data-rich case": demonstrated (no hurt). Strict
"≥": not met. Two remedies attempted → escalating the pass/hold decision to
Mark per §0. MSH stays quarantined until he rules. Both deliverables exist and
are demo-ready: EXP-013 (single-set best) and EXP-031 (pretrained→fine-tuned;
EXP-032 equivalent). Zero-shot results unaffected — those passed with +27pt
margins.

## 2026-07-29 — OWNER RULING: Phase-3 gate PASSED (tie accepted)

Mark ruled the condition-3 statistical tie acceptable ("good enough to move past
Phase 3"). Per the gate's on-pass protocol: MSH exits quarantine and joins the
corpus (configs/corpus/pretrain_v2.yaml — v1 frozen for EXP-030 provenance);
the NEXT Arena release becomes the new sandbag (code TBD at announcement —
update pretrain_v2 quarantine + PLAN §0 when known). P3.T5 (zero-shot
hill-climb) and P3.T6 (few-shot curves) deferred by owner in favor of Phase 4;
they remain unchecked in the plan and can be resumed any time. Committing and
pushing all session work before starting Phase 4.

## 2026-07-29 — P5.T4 promoted: real-time draft HUD plan (owner request)

Mark wants a HUD app running alongside Arena. Mechanism verified from
rconroy293/mtga-log-client source: tail ~/Library/Logs/Wizards Of The
Coast/MTGA/Player.log (Detailed Logs setting required), parse
Draft.Notify/LogBusiness/EventPlayerDraftMakePick JSON events; card ids are
Arena grpIds == Scryfall arena_id (375/375 coverage in our MSH cache → pure
lookup into cards.parquet). Full plan in docs/HUD_PLAN.md: LogFollower →
DraftState → Advisor (existing TorchScorer, calibrated probs) → rich-TUI v0
then pywebview panel with Scryfall art. Milestones H1-H5; H1 needs a captured
real draft log (Mark: enable Detailed Logs + save Player.log after next draft).

## 2026-07-29 — P4.T2 equivalence study: LoRA ≈ full-FT, decision LORA-BY-DEFAULT

Expert val top-1 across modes × MSH draft budgets (all init_from EXP-030):
            1k      5k      all(61.6k)   trainable
full-FT    .6935   .6966   .7043        100% (8.0M)
LoRA r4    .6919   .6946   .7021        4.2% (329k)
head-only  .6899   .6924   .6987        ~2% (174k)
LoRA sits 0.16-0.22pt under full-FT at every budget — equivalent within split
noise, at 1/24th the trainable params. DECISION: LoRA is the default for
onboard-set adapters (cheap, reusable base, separate adapter.pt); full-FT
reserved for flagship-set models where the last 0.2pt matters. Head-only is a
surprisingly strong floor (.6987 ≈ Phase-2 EXP-013's .6985 with 2% params) —
fine for day-3 quick adapters. Few-shot: ~1k drafts (≈ day 1-2 of a set)
reaches ~.69 expert in every mode — this table doubles as the P3.T6 evidence
at 3 budgets. LoRA device-placement + parametrized-checkpoint loading bugs
fixed along the way (params must be created on the base weight's device;
loader re-applies LoRA structure before load_state_dict).

## 2026-07-29 — PHASE 4 GATE PASSED

P4.T3 dry-run: `python -m draftbot.onboard_set --set KTK --budget-drafts 3000
--epochs 3` ran unattended: pipeline skip-detection → zero-shot scorecards
(KTK val .6475/.6719exp from the EXP-030 base) → LoRA adapter (0.32M trainable,
42s) → .6503/.6729exp scorecard. P4.T4 executed refresh: EXP-033 on corpus v2
(30 sets incl. MSH): MSH week1 .6852exp / full .6962exp — beats the v1 trunk on
every mode (recipe gate ✓); EXP-033 is now the recommended --base for future
onboard-set runs. Phase-4 gate: one-command onboarding ✓, adapter decision
documented with data ✓. Remaining open plan work: P3.T5/T6 (owner-deferred),
Phase 5 (deckbuilder implementation, HUD H1-H5 per docs/HUD_PLAN.md).

## 2026-07-29 — HUD H1+H2 landed (terminal advisor working in replay)

follower.py parses the exact message shapes from seventeenlands/mtga_follower
(Draft.Notify multi-line JSON buffering included); state.py maps grpId==
arena_id → vocab via the Scryfall cache and rebuilds (packs, prev, position) by
counting packs (Arena's reported numbers are display-only — base varies).
H1 DoD: synthesized Player.log of a REAL test draft replays to exact array
reconstruction (42+42 events, zero unresolved grpIds, mid-draft attach test).
H2: `python -m draftbot.hud --models checkpoints/EXP-033 [--replay log]`
renders live rich table: rank/name(rarity-tinted)/calibrated-prob bar/GIH/ALSA,
pool pips, model-disagreement flag in the title. Verified over a full 42-pick
replay with EXP-033. Awaiting Mark's captured real Player.log (Detailed Logs
on) to harden fixtures; H3 window UI awaits the design-brief round-trip.

## 2026-07-29 — HUD H3: window UI implemented from the Claude Design spec

Mark supplied a Claude Design doc (extracted from artifact bundle; tokens +
5 states + hierarchy spec). Implemented faithfully in hud/panel.html:
dark instrument theme (#08090A/#101216, oklch confidence cyan / split violet /
rarity golds), 56px hero row with 3px rail + fill-bar-as-probability, rows 2-3
at 60% scale with ALT #1 chip, sub-3% tail collapse, day-0 banner, idle/error
states, 220ms packIn only. hud/server.py: stdlib HTTP bridge (/, /state,
/art/<id> with Scryfall art_crop disk cache); __main__ --ui window opens
pywebview (if installed) else browser. Verified end-to-end: full replay into
the panel — state JSON, HTML 200, art fetch+cache 200. Remaining: H4 polish
(bot drafts, auto-onboard unknown sets), real captured Player.log from Mark.

## 2026-07-29 — HUD live-validated on a real Arena draft

Mark drafted MSH with the window HUD live. Two real-log format fixes shipped
mid-draft (request-envelope unwrap; set inference from pack grpIds via global
index — no join message needed) and verified against the in-progress log
before restart. HUD caught up mid-draft and tracked from pick 5 on. TODO:
freeze Mark's Player.log as the real H1 fixture when the draft ends.

## 2026-07-29 — First complete HUD-assisted draft; report card + fixture frozen

Mark completed a live MSH draft with the HUD. Report card vs EXP-033: 93% top-1
agreement (39/42); ALL disagreements were picks 1-4 — before the live-log fixes
shipped (his solo picks incl. P1P1 Baxter Building, model #7 @ 0.9%). Every
fixing land taken later was the model's #1 at that state (37-56%) — the
"splash enablement" is emergent expert imitation, not planning. Real log
sanitized (draft events only) into tests/fixtures/real_draft_msh.log as the
authoritative parser fixture. Live bugs found by the draft: request-envelope
unwrap, set inference from grpIds, variant-basic arena ids (odd/even printing
pairs — Scryfall unique=cards collapses them; now resolved via /cards/arena,
disk-cached). All fixed + tested same session.

## 2026-07-29 — P5.T1 ACTIVATED by owner ("the Subterranean Cavern incident")

The heuristic build suggester recommended a BG tapland in a WBr deck — the
exact class of error P5.T1 exists to kill. Owner ordered the deckbuilder.
Stage 1 (per the 2026-07-29 design note): per-card maindeck-membership model —
pool set-transformer, membership + land-count + basics heads, trained on
game_data winning builds (win-weighted), nonbasic lands treated as ordinary
pool cards so bad fixing gets bad membership probability. Stage 2: masked
discrete diffusion with partial-deck conditioning.

## 2026-07-29 — P5.T1 deck dataset landed (DECK_DATA_PLAN DoD met)

decks.parquet for MSH: 83,457 (draft, build) rows from 377,514 games, every
game accounted for. The pilot's per-group python loop was replaced outright
with the vectorized extractor the plan's perf note called for (whole-file
read ≈1GB, groupby size/sum + one representative row per build) — ~2 min for
MSH, so the 30-set scale-out needs no further work. All 5 validation-gate
tests green on the first run (tests/test_decks.py), full suite still green.

Surprises vs the 30k-game probe, now recorded in the plan + data card:
- Deck-size tail reaches 60 cards, not 43 (≤1% above 43; 92.6% exactly 40).
  Cap-at-43 stands for size-conditioned heads; oversize rows stay as
  membership examples.
- Basics range is wider (4–23 vs probe's 6–19); mean 14.8 unchanged.
- 83.5k builds, not the ~95k extrapolated; rebuilds go up to 7, not 5.
- Pleasant: game_data draft_ids ⊆ draft parquet ids → 100% split coverage,
  and pool identity vs the draft log held exactly on all 150 sampled drafts
  (game_data and draft_data column vocabularies agree perfectly).
- Trophy-view sizing for eval: 20.1% of builds have ≥5 wins, 6.1% ≥7.

Next: the stage-1 membership model (pool set-transformer, membership +
land-count + basics heads, win-weighted per the curation section).

## 2026-07-29 — Repo hygiene: src/draftbot/data was never in git

Committing P5.T1 exposed that .gitignore's unanchored `data/` pattern had
silently ignored the whole src/draftbot/data package since Phase 0 — every
"[P0.*]" commit shipped tests and docs but not the pipeline source. Pattern
anchored to `/data/`; package committed as it stands (c0d3506). Nothing was
lost (the working tree was always the source of truth), but any earlier
checkout of this branch would not have reproduced the pipeline.

## 2026-07-29 — P5 expanded into a hill-climb program; M5.1 (deck-eval suite) done

Owner directive: run the deck builder like the earlier phases — auto-research
hill-climb until builds look very close to upper-echelon decks given the pool.
PLAN.md Phase 5 expanded in place (P5.T1a–g, EXP-1NN namespace, deck metrics
defined). Eval suite built first, again.

M5.1 results (MSH val, 3,328 eval builds, 851 trophy):
- **rebuild-ceiling (human self-agreement): deck-F1 0.9545, trophy-F1 0.9627,
  lands-MAE 0.162, basics-L1 0.973** ← the target.
- gih-in-lane-build 0.7844 (best heuristic; the HUD-style builder), gih-top23
  0.7601, random-legal 0.6821.
- **Gate declared (P5.T1g, from the ceiling study):** test trophy-F1 within
  1.0pt of the rebuild ceiling, lands-MAE ≤ 0.8, plus the tapland regression
  check. Val reference ceiling: trophy-F1 0.9627.

Reading the numbers: the random floor is HIGH (0.68) — building 40 from a
~55-card pool forces overlap, so F1 progress compresses near the top; treat
every point above 0.78 as real work. The heuristics' failure is concentrated
in manabases: lands-MAE 2.2–2.7 and basics-L1 ~6.5 vs the ceiling's 0.16/0.97
— they take 17 basics and pip-split, humans play ~14.8 basics + the right
nonbasic lands. That's exactly the membership+basics+land-head territory.

## 2026-07-29 — EXP-101 intent (first trained deck builder, P5.T1e)

M5.2 infra landed: deck trainer (train/decks.py, win-weighted BCE + land/basics
heads, selection on val trophy-F1 through the real greedy decode), checkpoint
loading, 5 correctness tests green (permutation equivariance, pad inertness,
decode-legality ×200, loss finiteness, overfit canary F1≥0.80 in 8s).
EXP-101: the journal design-note architecture at emb 128/3 blocks (~0.9M
params), stats full, weighting (1+n_wins)·soft_skill. Must beat
gih-in-lane-build (val F1 0.7844 / trophy 0.7915) to graduate M5.2.

## 2026-07-29 — EXP-101 results (P5.T1e done): first model already 10pt over heuristics

EXP-101 val: **deck-F1 0.8829, trophy-F1 0.8920, lands-MAE 0.247, legal 1.000**
(2.3 min, ~0.9M params). +9.9pt F1 over gih-in-lane-build; gap to the rebuild
ceiling: 7.1pt trophy-F1. Analysis:
- F1 RISES with build wins (0.874 @ 0-win → 0.894 @ 7-win) — win-weighted
  imitation is aligning with winners, not just the average.
- Copy-error decomposition per deck: spells 6.55, basics 2.49, nonbasic lands
  0.34. The manabase problem the heuristics fail (2.2–2.7 lands-MAE) is
  essentially solved; remaining headroom = boundary spell slots + basics split.
- **Tapland regression check: 0/200 predicted decks contain an off-color
  nonbasic land.** The Subterranean Cavern error class is dead at EXP-101.
Hill-climb ladder launched: EXP-102 uniform weighting, EXP-103 stats none
(day-0 builder), EXP-104 scale (emb 256/4 blocks), EXP-105 hard ≥5-win
curation.

## 2026-07-30 — Hill-climb round 1 (EXP-102..105): one-line conclusions

- EXP-102 uniform weighting: trophy 0.8960 (+0.40 vs 101) — win-weighting
  slightly HURTS; the win signal is noisy at 4.5 games/build, echoes EXP-013.
- EXP-103 stats none: 0.8885 (−0.35) — a day-0 builder costs only ⅓ point;
  deck building is nearly stats-free (unlike drafting: −12pt zero-shot).
- EXP-104 emb256/4blk: 0.8946 (+0.26) — mild scale gain.
- EXP-105 min_wins≥5: 0.8906 (−0.14) — hard curation loses to soft, again.
Round 2: EXP-106 = uniform + scale (combine the two positive levers);
EXP-107 = uniform + basics_lambda 0.5 (attack the 2.3 basics-L1 bucket).
Best-known: EXP-102.

## 2026-07-30 — Owner refinement: trophy eval v2 + winner-pref metric

Owner: "take pools that would trophy and make decks that did trophy", and
"building a deck that also gets ≥5 wins is another excellent metric".
Implemented both:
- **trophy-F1 v2**: eval pools = drafts where ANY build hit ≥5 wins; target =
  the winningest build (was: most-played build filtered by its own wins). The
  practical delta is rebuild drafts where the FIX won — v2 targets the fix.
  Numbers barely moved (EXP-102 0.8960 v1 → 0.8960 v2; ceiling 0.9627→0.9625)
  because 81% of drafts are single-build, but the metric now says what we
  mean. Trainer selection switched to v2.
- **winner-pref**: P(model's deck strictly closer to the winning than the
  losing build of the same pool), ties 0.5, no learned judge. FINDING: every
  builder sits at ~0.5 (91 val pairs, ±0.05 noise) — even random-legal
  (0.527). Winner-vs-loser deltas are 1–2 card swaps; imitation F1 can't see
  which swap wins. This is the honest North-Star gap a win-aware objective
  (P5.T3-style aux head, own proposal needed) would have to close. A learned
  P(≥5 wins) deck judge is deliberately NOT snuck in (confounding).

## 2026-07-30 — Hill-climb round 2 + round-3 intent

- EXP-106 uniform+scale: trophy-v2 0.8942 — scale gain didn't survive uniform
  weighting (likely wants lower LR; parked).
- EXP-107 uniform+basics_lambda 0.5: **0.8977, new best** (+0.17 vs 102);
  basics-L1 2.32→2.22. Both rounds' gains now sum +0.6 over EXP-101.
Round 3 (one lever each vs EXP-107): EXP-108 decode=expected (membership mass
sets spell count — heads stop fighting over the spell/land boundary; legality
tests parametrized over both decodes); EXP-109 dropout 0.05.

## 2026-07-30 — Round 3: both levers negative; plateau counter at 2

- EXP-108 decode=expected: 0.8951 (−0.26 vs 107) — the land head's argmax was
  already calibrated (lands-MAE 0.25); membership mass is noisier. Keep greedy.
- EXP-109 dropout 0.05: 0.8967 (−0.10). Keep 0.1.
Round 4 (last before declared plateau): EXP-110 = EXP-107 + emb256/4blk at
peak_lr 2e-4, epochs 40 — "scale with the LR it wants" (EXP-106's flat result
looked like an LR artifact). If < +0.2: plateau documented → run the P5.T1g
gate with the best model.

## 2026-07-30 — Stage-1 plateau declared; P5.T1g gate run once on test: NOT passed

EXP-110 (scale + tuned LR) 0.8959 — third consecutive < +0.2pt → plateau per
P5.T1f DoD. Stage-1 best: **EXP-107** (uniform weighting, basics_lambda 0.5).

Test gate (frozen in docs/scorecards/phase5/):
- trophy-F1 v2 **0.8968** vs ceiling 0.9654 → 6.9pt short — **FAIL** (needs ≤1.0)
- lands-MAE 0.254 (≤0.8) — **PASS**
- tapland check 1/200 (spec: 0; val 0/200) — marginal fail, 0.5% residual
- winner-pref 0.386 (70 test pairs; val 0.473) — imitation is blind to which
  1–2-card swap wins, consistently.

Verdict: stage-1 per-card membership has taken pure imitation to ~0.90 F1 —
+11pt over the HUD-style heuristic, manabases solved, the Subterranean Cavern
class ~dead — but the last 7 points to "build the deck that trophies" are
JOINT structure (which 23 spells cohere; which swap wins), invisible to
independent per-card membership.

**Stage-2 proposal (needs owner sign-off per Phase-5 rules):** masked discrete
diffusion over deck slots, pool-conditioned with partial-deck conditioning
(lock cards, resample rest — the interactive HUD builder UX), trunk
warm-started from EXP-107; PLUS a win-aware auxiliary (P5.T3-style event-wins
head, confounding addressed by within-pool contrast pairs — exactly the
winner-pref construction). Success bars for the pilot (≤2h): val trophy-F1
+1.5pt over EXP-107 AND winner-pref ≥ 0.55. If the diffusion pilot can't beat
the one-shot model, the honest conclusion is that the remaining gap is pilot
variance, not build skill.

## 2026-07-30 — Stage 2 APPROVED by owner; EXP-111..113 intent

Owner sign-off on the stage-2 proposal; same hill-climb regime, keep levers
that pull weight. Implemented: DeckBuilder(diffusion=True) — zero-init
deck-state embedding (warm-start no-op proven by test), masked-count training
objective, deterministic MaskGIT decode (cosine commit schedule), quality head
+ within-pool winner/loser margin loss, rescore decode (deterministic
candidates × quality head). 6 new tests, 11/11 green.
- EXP-111: diffusion + maskgit decode, warm-start EXP-107.
- EXP-112: + quality contrast loss (the winner-pref play, as training signal).
- EXP-113: + rescore decode (quality head picks among candidates).
Bars (pre-declared): val trophy-F1 ≥ 0.9127 AND winner-pref ≥ 0.55.

## 2026-07-30 — Stage-2 round 1 (111-113) + two decisive diagnostics

- EXP-111 diffusion+maskgit 0.8936 / EXP-112 +quality 0.8946 / EXP-113
  +rescore 0.8948 — all below EXP-107 (0.8977). Decode decomposition:
  masked training −0.26, maskgit decode −0.15 further.
- Winner-pref UNMOVED (0.46-0.47) by the quality head.
- Diagnostic 1: rescore candidates — 1.84 distinct/5, identical in 42.5% of
  pools. Some diversity exists; the head just isn't choosing well.
- Diagnostic 2 (label reliability): Beta-posterior P(observed winner truly
  better) = **0.738** over the 91 val pairs → the winner-pref bar of 0.55 is
  reachable (oracle ≈0.74); our 0.47 is a model failure, not label noise.
Round 5 (trophy levers, one each vs 112): EXP-114 epochs 50; EXP-115
reveal_weight 0.2; EXP-116 cosine mask distribution. Round 6 planned:
quality-head v2 — contrast pairs from ALL strict win-gap≥2 rebuilds (~5x
data) and basics fed to the quality head (manabase deltas are currently
invisible to it).

## 2026-07-30 — Stage-2 hill-climb CLOSED: plateau on both bars; EXP-107 stands

Rounds 5-6 one-liners:
- EXP-114 epochs 50: 0.8946 (=112). EXP-115 reveal_weight: 0.8941 (−).
- EXP-116 cosine mask: 0.8951 — best diffusion, still −0.26 vs EXP-107.
- EXP-117 quality-v2 (5,351 pairs — 4×; basics visible to the head):
  winner-pref 0.462 → **0.511 with rescore** — right direction, but ±0.05
  noise at n=91 val pairs; bar 0.55 not demonstrably met. Trophy 0.8931.

Verdict per the owner's "bars or plateau": PLATEAU on both arms.
- trophy-F1: 7 stage-2 attempts, none beat EXP-107's 0.8977. Masked training
  costs ~0.3pt of imitation accuracy and iterative decode doesn't win it back
  at MSH scale.
- winner-pref: 0.47 → 0.51 across three quality-head variants; oracle ceiling
  is 0.738, so headroom exists, but the VAL METRIC has ±0.05 noise (91 pairs;
  test has 70) — the metric itself lacks the power to gate at 0.55. Flagged
  to owner: winner-pref needs a bigger pair population (cross-set corpus
  decks would give ~30-60k pairs) before it can be a gate.

Production recommendation unchanged: **EXP-107** (stage-1) for the HUD build
suggester. **EXP-116** is the best diffusion checkpoint and the only one with
partial-deck conditioning (lock cards, resample rest) — worth wiring into the
HUD for interactive rebuilds despite −0.26pt, since conditioning is the
feature stage-1 cannot do at all.
Backlog (promoted via this entry): (a) win-aware fine-tune of stage-1
directly (skip diffusion); (b) corpus-scale deck pretrain (the day-1 builder
path per DECK_DATA_PLAN scale-out) — also fixes winner-pref metric power;
(c) HUD integration of EXP-116 conditioning.

## 2026-07-30 — Deck corpus extracted: 4.48M builds / 28 sets; EXP-120 intent

Owner: corpus pretrain approved. Extraction sweep findings:
- 2021-era game files (AFR/MID/STX/VOW) are TAR-wrapped inside the gzip and
  use `user_win_rate_bucket` — extractor now handles both.
- AFR dropped: its game-data draft_ids join its draft logs at 0.3% (id
  universes don't match); MID keeps 79% coverage (~110k usable builds — the
  split join excludes the rest, hygiene intact).
- Corpus totals 4,477,523 builds (plan estimated 1.5–2M). Size/coverage
  outliers (LCI 98.4% in 40–43, PIO 97.3%, SNC 92% coverage) recorded;
  oversize rows remain valid membership examples.
EXP-120: pool→deck pretrain over 27 corpus sets (MSH HELD OUT for the day-1
measurement), uniform weighting + basics_lambda 0.5 (the stage-1 winners),
stat_dropout 0.25 (day-0 is a trained condition), corpus-only scaler,
selection on corpus proxy sets (EOE/BLB/KTK/SNC — holdout never drives it).
Bars: zero-shot MSH val (stats none) beats gih-in-lane-build (0.7915 trophy);
EXP-121 fine-tune ≥ EXP-107 (0.8977).

## 2026-07-30 — CORPUS PRETRAIN RESULTS: both bars passed; EXP-121 is the new MSH best

EXP-120 (27-set pool→deck pretrain, 3.70M train builds, 2.6h, 0.81M params,
MSH held out, stat-dropout 0.25):
- **Zero-shot MSH (stats none — true day-1): trophy-F1 0.8298** — beats the
  stats-informed gih-in-lane heuristic (0.7915) by +3.8pt with ZERO MSH data
  of any kind. Bar passed.
- **Zero-shot MSH (stats full): 0.8871** — within 1.1pt of the fully
  MSH-trained EXP-107, without ever seeing an MSH deck. Deck building
  transfers across sets far better than drafting (draft zero-shot gap was
  ~3pt at week1 after heavy tuning).
- **EXP-121 (MSH fine-tune, 3 min): trophy-F1 0.9005 — NEW BEST**, +0.28
  over EXP-107; best basics-L1 (2.118) and curve-L1 (3.671) of any model.
  Bar (≥ EXP-107) passed: pretraining helps even the data-rich case.
Day-1 playbook is now measured, not speculative: ship 0.83 on release day,
0.887 when the ratings API fills in (~day 3-7), 0.90+ the afternoon the
game_data drop lands (3-minute fine-tune).
NOT re-running the P5.T1g test gate with EXP-121 (would be a second gate
attempt after EXP-107's fail — §0 says escalate): 0.9005 val would still sit
~6pt from the 0.9625 rebuild ceiling. Owner decides whether the gate's
ceiling criterion stands or the corpus trajectory (bigger trunk, more sets)
gets another round first.

## 2026-07-30 — Bottleneck diagnostic (owner question: evals, data, or model?)

Measured on EXP-121, MSH val:
- F1 vs winningest build 0.8821 (multi-build drafts) vs F1 vs BEST-matching
  build of the same player: 0.8940 — build-choice ambiguity explains only
  +1.2pt of the ~6pt gap. The eval yardstick is fairer than suspected.
- Disagreement decomposition (5.98 slots/deck): 41% at p∈[.35,.65] — model-
  acknowledged coin flips (irreducible 23rd-card entropy); 26% confident
  errors (p<.2 or >.8) ≈ 1.5 slots/deck — the genuinely recoverable part.
Verdict: data volume exhausted (all public sets consumed); evals only
bottleneck winner-pref (91 pairs — fix = corpus-wide pairs); trophy-F1
headroom is (a) ~1-2pt from the confident-error slice — best remaining lever
is corpus-regime model scale (0.81M params on 3.7M builds is starved; MSH-only
scale was flat but that was the 75k regime) + un-capped pretrain epochs, and
(b) a measured irreducible slice — the P5.T1g "within 1pt of the same-player
ceiling" criterion is likely unreachable for ANY population model; a
population-fair target is ~0.92-0.93.

## 2026-07-30 — HUD deck-builder handoffs written; EXP-122/123 size probe intent

Owner: parallel tracks. docs/DECKBUILDER_DESIGN_BRIEF.md (Claude Design
prompt, extends the H3 design system) + docs/DECKBUILDER_HANDOFF.md
(engineering agent: EXP-121 one-shot, EXP-116 lock-and-rebuild, /state
extension, lock-respect test; scope fenced to hud/).
EXP-122: the corpus-regime scale probe from the bottleneck diagnostic —
emb 256/6 blocks (~5M params vs 0.81M) on the 3.7M-build corpus, batch 512,
lr 2.5e-4; targets the 26% confident-error slice. EXP-123 = its MSH
fine-tune. Owner compute sign-off given ("kick off the size probe");
estimated 5-7h, checkpointed/resumable. Bars: EXP-122 zero-shot(full) >
0.8871; EXP-123 > 0.9005 by ≥ +0.2 to justify the size for production.

## 2026-07-30 — Owner: probe epochs too; keep probing until plateau

Standing directive for the corpus-regime hill-climb. Probe queue (one lever
per EXP, sequential — MPS fits one corpus run; each probe = corpus pretrain +
3-min MSH fine-tune; production metric = fine-tuned MSH val trophy-F1):
1. EXP-122/123 scale probe (running).
2. Epochs probe: best-so-far corpus config with the cap lifted (EXP-120
   was still improving at its 12-epoch cap).
3. Then by results: MSH-in-corpus production pretrain (holdout was for the
   day-1 science; production may include it — zero-shot numbers stay frozen
   from EXP-120/122), stat_dropout rate, corpus win_skill weighting, era
   weights, fine-tune recipe (LoRA vs full, FT epochs).
Plateau rule: 3 consecutive probes with fine-tuned gain < +0.2pt. Each probe
journaled with a one-line conclusion + leaderboard rows. Compute: each run
kept ≤10h (§0); owner sign-off on the series given in this entry's directive.

## 2026-07-30 — HUD H5 shipped: deck-builder view running the real models

Owner request: finish the HUD, adapt the updated Claude Design treatment
(deck-builder section added to the H3 pick view), and run the deck-builder
model inside the HUD app. Design source: the "Draft Buddy MTG HUD Design"
project's `Draft Buddy HUD.dc.html` (read via DesignSync; the plain share URL
403s) plus its published artifact. Scope per docs/DECKBUILDER_HANDOFF.md.

**What shipped**

- `hud/deck.py` — `DeckAdvisor`: drafted nonbasic pool → `DeckArrays` → a
  proposed 40. EXP-121 one-shot for the initial build, EXP-116 (diffusion) for
  lock-conditioned rebuilds. Produces the whole panel payload: cmc groups,
  membership confidences, the shared boundary zone, cut list, mana summary,
  curve/pips, Arena-importable list, rebuild diff. Checkpoints load on a
  background thread so a live pick never waits on them.
- `models/builder.py` — the ONE permitted core touch: optional
  `init_state=` on `TorchDeckBuilder._maskgit_probs`. Locked slots start
  committed (count = in, 0 = out) and are returned OUTSIDE the sigmoid range,
  so greedy assembly can never trade a lock away on a tie. `init_state=None`
  is byte-identical to the old path (asserted).
- `hud/server.py` — `update_deck()` merges a `deck` block into the existing
  `/state` poll (no second loop; the deck survives pick-view updates) and
  `POST /lock|/rebuild|/clear_locks` dispatch to an action handler.
- `hud/panel.html` — the design's B1–B5 as a second view of the same window:
  build (scrolling, **every card present, never summarised**, sticky cmc
  headers, full mana cost per row as generic-numeral + WUBRG glyphs), cuts,
  lock/rebuild diff, provisional, day-0. New tokens only: `--add`/`--cut`
  diff rails + the five mana fills. Degrades at 240px (pips and counts drop,
  name + confidence bar survive).
- `hud/__main__.py` — `--deck-models` (default EXP-121,EXP-116), `--no-deck`,
  `--provisional-from` (30). Terminal v0 prints the build too.

**Verified** (`tests/fixtures/real_draft_msh.log` full replay, MSH):
40 cards · 17 lands · legal, 7 Plains / 6 Swamp / 1 Mountain + 3 nonbasic
lands, groups 1–5+ drops, boundary .67/.59/.52 in ↔ .38/.16 out. Lock two
cards through the real HTTP surface → rebuild honours both (0.54s), stays
legal at 40. Screenshots: `docs/hud/deck-{build,cuts,diff,prov,day0,narrow,
pick_prov}.png`.

**Bug caught by the end-to-end smoke, not by unit tests:** the first
`draft_finished()` used "last pack holds ≤1 card", which is true at the end of
EVERY booster — the HUD declared the draft over at P1P14 and proposed a
27-card-pool build with 30 lands. Now requires 3× the observed pack size as
well; the test walks the whole fixture and asserts False at picks 14 and 28.

`uv run pytest` green (incl. new `tests/test_hud_deck.py` — 6 tests — and two
lock-respect tests in `tests/test_deck_model.py`). No leaderboard rows (not
eval-harness territory); PLAN.md untouched — this is HUD_PLAN H5.

## 2026-07-30 — Probe 1 (scale) NEGATIVE; probe 2 (epochs) launched

EXP-122 (4.81M params, 8.1h): proxy 0.9084 (+0.16 vs small) but held-out MSH
unchanged — zero-shot full 0.8878 (=), none 0.8283 (=), fine-tuned EXP-123
0.8998 (−0.07 vs EXP-121). One line: 6× capacity buys in-corpus fit, zero
transfer; the confident-error slice is not capacity-starved at this scale.
Plateau counter 1/3. Probe 2: EXP-124 = small corpus config with epoch cap
24 (both corpus runs were still improving at cap 12); EXP-125 = its MSH FT.

## 2026-07-30 — Owner objective refined: current-set AND next-set-day-1, one recipe

Directive: the deliverable is a corpus trunk (day-1 model) + per-set
fine-tune (current-set model). Probe scoring is now TWO-AXIS: zero-shot
(stats none) day-1 + fine-tuned current-set; keep levers that help one axis
without hurting the other. Probe 3 redesigned: corpus decks_v2 = all sets
INCLUDING MSH, EOE held out as the next-set stand-in — measures MSH-in-corpus
production lift and replicates the day-1 result on a second set (EOE was an
EXP-120 proxy set → first direct seen-vs-unseen gap on the same set).

## 2026-07-30 — Probe 2 (epochs): KEEP for day-1; FT plateau counter 2/3

EXP-124 (epochs 24): proxy 0.9082 (+0.14); zero-shot none **0.8330 (+0.32 —
the day-1 axis)**; zero-shot full 0.8859 (flat); EXP-125 FT 0.9002 (tie).
One line: longer pretraining converts to day-0 robustness, not to in-set
accuracy. Kept per the two-axis rule; production trunk = 24 epochs.
Probe 3 launched: EXP-126 = corpus decks_v2 (MSH IN — production; EOE held
out as next-set stand-in), epochs 24; EXP-127 = MSH FT. Two-axis bars:
EXP-127 > 0.9005 (MSH-in-corpus should lift current-set); EOE zero-shot
gap vs its seen-in-corpus baseline quantifies the true day-1 cost.

## 2026-07-31 — Probe 3: EXP-127 new production best (0.9011); FT-axis PLATEAU declared; day-1 replicated

- **EXP-127 (FT from MSH-in-corpus trunk): MSH val trophy-F1 0.9011** — new
  best (+0.06 vs EXP-121); third consecutive sub-+0.2 FT result → the
  current-set axis is formally plateaued per the 3-probe rule. Production:
  trunk EXP-126, MSH model EXP-127.
- **Day-1 replication (EOE, never seen): 0.8347 none / 0.8981 full** — mirrors
  MSH (0.8298/0.8871). The recipe generalizes across holdout choice.
- **Seen-vs-unseen priced on the same set (EOE, EXP-124 vs EXP-126)**:
  no-stats 0.8997 vs 0.8347 (−6.5pt); full-stats 0.9101 vs 0.8981 (−1.2pt).
  Day-1 cost is almost entirely erased by card ratings; note stat-dropout
  lets a SEEN set build at full quality with zero stats.
Probe 4 (day-1 axis; FT axis plateaued): EXP-128 = v2 corpus +
stat_dropout 0.4; EXP-129 = its MSH FT (guard: FT must not regress).

## 2026-07-31 — Probe 4 negative; PROBE SERIES CLOSED — both axes at plateau

EXP-128 (stat_dropout 0.4): EOE day-1 0.8336 (−0.11), EOE full 0.8991 (+0.10
noise), MSH FT via EXP-129 0.9002 (−0.09). Dropout 0.25 was already right.
Queue empty; FT axis plateaued at probe 3, day-1 axis now 2 consecutive flats.

**FINAL PRODUCTION RECIPE (frozen):**
- Trunk: **EXP-126** — corpus decks_v2 (28 sets incl. MSH, AFR excluded),
  epochs 24, stat_dropout 0.25, uniform weighting, basics_lambda 0.5.
- Current-set model: **EXP-127** = EXP-126 + 8-min MSH fine-tune —
  **val trophy-F1 0.9011** (best of 29 experiments).
- Interactive builder: EXP-116 (lock-and-rebuild) per DECKBUILDER_HANDOFF.
- Next-set playbook (measured, not estimated): announcement day → set joins
  quarantine in decks manifest; release day → zero-shot ≈ 0.83 trophy-F1
  (no data), ≈ 0.90 once ratings API fills (~ −1.2pt vs a seen set);
  game_data drop → extract (~2 min) + fine-tune (~8 min) → ≈ 0.90+ in-set.
Probe ledger: scale ✗, epochs ✓ (day-1 +0.32), MSH-in-corpus ✓ (+0.06 FT,
production-correct), stat_dropout 0.4 ✗. 29 deck EXPs total, all on the
leaderboard.

## 2026-07-31 — Production model lock: corpus trunks ship, not the in-set winners

Owner sign-off on the researcher's final read. Recorded in
docs/DECKBUILDER_HANDOFF.md ("Final model selection — LOCKED") and encoded as
`PROD_DRAFT_MODELS` / `PROD_DECK_MODELS` in `draftbot/hud/__main__.py`, so
`python -m draftbot.hud --ui window` now needs no model flags at all.

Ship: **EXP-033** (draft) + **EXP-126** (deck) + **EXP-116** (lock-rebuild;
the only builder with diffusion=True, so the only one that conditions on locks
natively). NOT shipped: EXP-013 (test top-1 0.6756 / 0.6985exp, the best
measured MSH drafter) and EXP-121 (trophy-F1 0.9005, the best measured MSH
builder). Generalization — day-1 zero-shot on an unseen set plus the
onboard-set fine-tune path — is what the corpus machinery buys, and it is what
a HUD needs; in-set accuracy is a wash (EXP-033 MSH full .6962exp vs .6985).

Two findings that repeated across both halves of the project:
- **Scale did not pay.** EXP-023 (emb 256/6) came in BELOW EXP-013 on test
  (0.6742 vs 0.6756), the same shape as EXP-122/123 landing below EXP-121 on
  the deck side. ~2M params is the practical imitation ceiling here.
- **Soft weighting beat hard curation.** The no-hard-skill-filter ablation on
  the draft side is the same lesson EXP-105 relearned on the deck side.

Integration check before promoting EXP-126: on the real MSH pool it returns a
legal 40 with the same 17 lands and the same basics split as EXP-121, 20/21
mainboard overlap. The one substantive disagreement is real, not a bug — the
pool holds 3× Stolen Stark Tech; EXP-126 plays all three, EXP-121 played one.
tests/test_hud_deck.py now derives its checkpoints from PROD_DECK_MODELS so
the DoD cannot drift from what ships. `uv run pytest` green.

**Provenance gaps left open** (flagged in the handoff; PLAN §0 makes the
leaderboard the source of truth, and neither shipped model satisfies it):
1. EXP-033 has NO leaderboard row — its MSH numbers exist only in the
   2026-07-29 journal entry. Needs a `python -m draftbot.eval` row.
2. EXP-126 has no MSH row, only EOE (val full trophy-F1 0.8981 / none 0.8347).
   If post-promotion MSH is inside decks_v2 then an MSH row would be
   contaminated and EOE is the honest read — but that reasoning belongs in the
   leaderboard, not in an agent's head.

## 2026-07-31 — Owner question: is the MSH fine-tune needed on the v2 trunk? NO

EXP-126 (trunk, no FT) on MSH val: trophy-F1 0.9004 vs EXP-127 (FT) 0.9011 —
+0.07, inside noise. For sets IN the corpus, fine-tuning is redundant; the
FT step matters only in the release→next-refresh gap window (there it's
worth +1.3pt, per EXP-120→121). Deployment simplification: EXP-126 alone
covers current-set AND next-set-day-1; EXP-127 remains marginally best for
the HUD as already wired.

## 2026-07-31 — Laptop-deployment bundler productized

`python -m draftbot.bundle --set MSH [--set <NEXT> ...]` → laptop-bundle.tar.gz
(checkpoints minus resume state, per-set cards/features/decks/splits/snapshots,
Scryfall cache; training data excluded). Defaults to the frozen production
checkpoints (EXP-033/116/126/127). Next-set workflow documented in the module
docstring. Smoke test in tests/test_bundle.py.

## 2026-07-31 — Sealed fast path: TLA event support in half a day (P5.S)

Owner request: help building decks for the live Avatar sealed event. Corrected
the target first — Scryfall has no "AVA"; the set is TLA, already onboarded
with full stats and 114k draft builds inside decks_v2. 17lands DOES publish
sealed game data (Sealed + TradSealed, every corpus set, schema-identical to
draft game files) — and sealed needs NO draft-log join: pool = deck ∪ side by
construction, so the AFR failure class doesn't exist here.

Shipped as a pure ADDITION (owner constraint mid-build: nothing draft-side
may move): `source="sealed"` threads through extractor/splits/dataset/eval/
trainer with `draft` defaults everywhere; new artifacts only
(decks.sealed.parquet, <SET>.sealed.json, tests/test_sealed_decks.py — the
5-test gate with pool-size sanity replacing the draft-log identity test).
Full suite green, all 11 skips pre-existing. TLA: 9,628 sealed builds /
5,950 pools (extraction seconds).

S2a controls (TLA sealed val, full stats) tell the story: human rebuild
ceiling 0.9060 F1 / 0.9224 trophy — LOWER than draft's 0.95, as predicted
(84-card unfiltered pools admit more legitimate builds). EXP-126 zero-shot
0.6928/0.7201: beats every heuristic (in-lane 0.6029, gih-top23 0.5213,
random 0.4194) but ~21pt off ceiling — the draft→sealed shift is COLOR choice:
lands-MAE is already excellent (0.30) while basics-L1 is 9.2. EXP-130
(EXP-127 recipe on TLA sealed, 0.8min) closes to 0.7261/0.7578 (+3.8pt
trophy). 9.6k builds can't teach sealed color selection alone — that is the
sealed-corpus pretrain's job (next entry).

Owner interface: `python -m draftbot.build --set TLA --pool pool.txt` —
Arena-export parsing, per-card membership confidence, bubble view (weakest
in / best out). Smoke-run on a real 7-0 val pool produced a legal, coherent
W-base build with the fixing land honestly at 51%. Winner-pref on sealed val
is 14 pairs — meaningless at this scale, reported but ignored.

## 2026-07-31 — Sealed corpus program closed: EXP-134 is the sealed trunk (P5.S3)

27-set sweep first: the gate earned its keep immediately — 1-2%/set of sealed
"builds" are 44-91-card unbuilt piles, and ~1% of LCI/EOE entries disagree
with their own pool by one card. Both dropped at extract, logged, TLA rows
refreshed (deltas ≤0.001). Final corpus: 26 sets / 335k train builds,
sealed_v1 manifest = decks_v2 weights, EOE holdout.

Three trunk runs, one plateau (~0.71 proxy trophy): EXP-131 (EXP-126 warm
start, FT-class lr 1.5e-4) starts at 0.711 and never improves — a draft trunk
won't re-shape at that lr. EXP-132 (scratch, 3e-4) climbs 0.658→0.705.
EXP-134 (warm start × 3e-4, patience 4) = 0.7123, best of the three, lowest
loss. The sealed-data wall is volume, not init: 335k builds is 1/11 of
decks_v2.

Holdout story (EOE sealed val, ceiling 0.8971 trophy): sealed corpus
genuinely transfers — EXP-134 0.6421 full / 0.5432 none vs the draft trunk's
0.5708/0.4795 (+7.1pt / +6.4pt) and vs in-lane 0.5504. Day-1 sealed is
~64% of the way from heuristics to the human ceiling with ratings, less
without — honest but far from the draft-side story (data-bound).

In-corpus-needs-no-FT replicates on sealed: TLA val trophy — EXP-130 (FT of
draft trunk) 0.7570, EXP-133 (FT of EXP-134) 0.7548, EXP-134 zero-shot
0.7516, EXP-131 0.7533: a noise band. PRODUCTION: EXP-134 (CLI default,
added to bundle defaults). EXP-130/133 remain on disk as TLA-max variants.

Gap to ceiling on TLA sealed: 0.7516 vs 0.9066/0.9224 — the big remaining
lever is sealed data volume; a second one worth a future proposal is joint
draft+sealed training (shared trunk, format token). Both are new-EXP
territory, not tweaks.

## 2026-08-01 — Joint draft+sealed training: EXP-135 is production for BOTH deck formats (P5.S5)

Owner directive: "a good deck is a good deck in limited" — joint-train with a
format token, keep the model aimed at trophy-winning decks, and test the
sealed-bombs hypothesis by fine-tuning toward winning sealed decks.

Mechanism (additive; old checkpoints unaffected, tests/test_format_token.py):
DeckBuilder(formats=True) adds a zero-init 2-row format embedding to every
pool slot; DeckCorpusTrainer takes corpora=[{manifest, boost}] and reports
per-format proxy val. EXP-135: decks_v2 + sealed_v1 (boost 4.0 → sealed 27%
of samples), warm start EXP-126 × lr 3e-4 (the EXP-134 recipe), 4.0M builds,
best at epoch 5 of 9 (2.4h).

Result — the joint bet paid on the seen-set axis and cost nothing on draft:
- Draft decks preserved: MSH 0.8989 (EXP-126: 0.9004), EOE 0.8987 (0.8981),
  EOE none 0.8327 (0.8347) — all ≤0.2pt, noise.
- TLA sealed 0.7576 trophy — best of the program, zero per-set fine-tune
  (EXP-130 needed one to hit 0.7570; EXP-134 alone: 0.7516).
- Epoch 5 beat BOTH solo trunks simultaneously on proxies (draft 0.9048 /
  sealed 0.7146) — the draft gradient does lift sealed past its solo plateau.
- The one concession: EOE sealed holdout 0.6328/0.5231 vs EXP-134's
  0.6421/0.5432 — the sealed-only trunk keeps a 1-2pt day-0/day-1 edge on an
  unseen set. Next-set recipe stays "refresh the joint corpus on release."

Annealing probes (sealed-only continue-train of EXP-135): EXP-136 (uniform)
flat everywhere. EXP-137 (win_skill — the bombs hypothesis) gained +1pt on
sealed proxies AND +0.9pt on TLA 7-win-F1 (0.7775, best of program), but was
flat on TLA/EOE trophy-F1 — directional support, not promotion-grade. Filed
as the first lever to revisit when sealed data grows.

PRODUCTION: EXP-135 (build CLI default with --format sealed; bundle swaps
EXP-134 → EXP-135). HUD deck models untouched (EXP-126/127 per handoff — a
swap to EXP-135 would need its own HUD-side validation). One trunk now
builds trophy-class decks in both limited formats.

## 2026-08-01 — HUD sealed mode shipped against a LIVE pool (P5.S4)

Owner joined the ArenaDirect TLA sealed event; the Player.log capture became
the parser spec: the Event_Join response carries a Course object with
CardPool (84 grpIds) — same shape in the relaunch courses list. Port landed
in one pass because every risky piece pre-existed: grp→vocab mapping went
84/84 with zero /cards/arena fallbacks, DeckAdvisor.build() already takes a
flat pool, and the panel auto-switches to the build view on a
non-provisional deck.

Additive surface: SealedPool event + buffer-scanning _sealed_pools() in the
follower ("Sealed" in InternalEventName gates it — draft courses can't
hijack), SealedState (reuses DraftState's resolver, drops basics per pool
identity), DeckAdvisor.format_id, PROD_SEALED_DECK_MODELS=EXP-135. Locks in
sealed use the probability-pinning path deliberately — it keeps the
sealed-aware joint model in charge instead of the draft-trained EXP-116.
Draft flow untouched; suite green.

Fixture tests/fixtures/real_sealed_tla.log (sanitized: GUIDs zeroed, styles/
collation/summary stripped, CardPool intact). tests/test_hud_sealed.py:
parser → state (80 nonbasics after dropping 4 pool basics) → legal 40 →
lock-respect. Replay renders the same build the CLI produced for the same
pool — WU Katara/Fortune-Teller with a 2-Mountain/1-Forest splash off North
Pole Gates.

## 2026-08-01 — Production wart (filed, not fixed): hybrid pips inflate basics

Live TLA sealed build: the basics head proposed 6P/7I/2M/1F for a deck whose
every spell casts off WU — Suki ({G/W}{G/W}) and Messenger Hawk ({U/B})
contribute phantom G/B pip mass to the static features, and the pip-aware
basics allocation obliges. Owner registered 8P/8I (correct). Fix candidates
when worth it: hybrid-aware pip features, or a castability check in decode.

## 2026-08-11 — Day-0 onboarding: HOB drafted on release day, zero 17lands data (P4.T3)

The Hobbit went live on Arena and the HUD could not see it. Three gaps, all
load-bearing, none of them previously exercised — every set we had ever
onboarded already had 17lands data by the time we touched it:

1. `build_cards` derives the vocab from the `pack_card_*` columns of the
   17lands CSV. `probe("HOB", "PremierDraft")` → None; no CSV, no vocab.
2. `scorer_from_checkpoint(..., stats="none")` still calls
   `assemble(set, "full")` to get the stat block's SHAPE before zeroing it —
   so day-0 mode needed a snapshot that by definition cannot be computed.
3. `grp_to_vocab` bridges Arena grpIds via Scryfall `arena_id`. Scryfall had
   all 193 HOB + 103 HOC cards listed as `games: [..., arena]` with
   `arena_id: null` — the backfill lags the Arena release. The
   `/cards/arena/<id>` fallback 404s for the same reason. Not one pack would
   have parsed.

HUD_PLAN's "static-only mode still works via Scryfall + `none` stats" was
therefore aspirational. It is real now.

**Source of truth for day 0 is the Arena client itself.** It ships a SQLite
card database (`Downloads/Raw/Raw_CardDatabase_<hash>.mtga`); `Cards` ⋈
`Localizations_enUS` is grpId → name for every set the client has downloaded —
i.e. available strictly before anything else in the world knows the set exists.
New `draftbot.data.arena_db` reads it `mode=ro`; `draftbot.data.day0` turns it
into the standard artifacts:

    uv run python -m draftbot.data.day0 --set HOB

vocab (Arena names → Scryfall card objects) → static features → an all-NaN
stat snapshot registered under both `full` and `week1`. NaN is the point:
`apply_scaler(add_masks=True)` maps it to 0 with every missing-mask on, which
is exactly the stat-dropout condition the trunk was pretrained under. Day-0 is
a trained-for condition, not a degraded one.

Two Arena-DB gotchas, both caught by validating against MSH's known-good
339-card vocab before trusting anything:
- **`DraftContent` is not draftability.** Basics carry 0, and so does every
  one of HOC's 104 cards. Filtering on it silently produced a 167-card vocab.
  The filter is `IsToken = 0 AND IsPrimaryCard = 1`, which covers 339/339 of
  MSH at the cost of 36 bonus-sheet cards that never reach a pack — the right
  way round, since an over-wide vocab costs nothing at inference and a missing
  card is a hole in the pack.
- **Names carry display markup.** Arena wraps hyphenated names to prevent line
  breaks (`<nobr>Lake-town</nobr> Lookout`); unstripped, 26 HOB names missed
  Scryfall entirely.

HOB result: **296 cards** (= Scryfall's 193 HOB + 103 HOC exactly), 308 grpIds,
zero unresolved names, static feature columns identical to MSH's. Bonus sheet
discovered from Scryfall's `parent_set_code` graph rather than the hardcoded
`COMPANION_SHEETS`, and persisted to `sources.json`, which now overrides that
registry everywhere (`source_sets()`) — the next set needs no code edit.
Verified end to end: EXP-033 ranks a P1P1 (prob mass 1.0), EXP-126/116 build a
legal 40, EXP-135 builds a legal sealed 40, `infer_set` recognises HOB from
raw grpIds, and `--stats full` produces a bit-identical table to `--stats none`
for a day-0 set, so the flag cannot be got wrong.

Everything is PROVISIONAL and says so: a `.day0` marker sits next to
cards.parquet, and `build_cards` now regenerates from the CSV (dropping the
marker) the moment 17lands publishes. **Card ids change at that boundary** —
nothing trained on day-0 ids may outlive it. No leaderboard row: HOB has no
val split to evaluate against, and won't until 17lands publishes.

Bundle now carries the day-0 sidecars (`sources.json`, `.day0`) — without
sources.json the target machine drops the entire bonus sheet from both the
grpId bridge and the art map. tests/test_day0.py: 13 tests, the Arena-DB half
hermetic against a miniature fixture DB. Full suite green (11 pre-existing
skips).
