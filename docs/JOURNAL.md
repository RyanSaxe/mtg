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

## 2026-09-05 — HOB data landed: the day-0 claim measured for real; recipe review before the HOB runs

17lands published HOB (PremierDraft 43,639 drafts / 1,832,838 picks, t=42,
dates 2026-08-11 → 08-29; game data for Premier/Trad/Sealed/TradSealed).
Protocol followed in order: HOB → `quarantine` in pretrain_v2 / decks_v2 /
sealed_v1 (test_latest_manifest_has_sandbag is no longer xfail; PLAN §0 names
HOB), THEN download, THEN `onboard_set --set HOB --base EXP-033
--skip-finetune`. HOB is the first set to cross the day-0 → CSV boundary, and
crossing it found two bugs in last month's day-0 path:
- **Stale static features survived the id change.** `onboard()` rebuilds
  `features.static.parquet` only when absent; `build_cards` dropped the
  `.day0` marker but left the table keyed on day-0 ids — every feature row
  would have been silently mis-indexed. `build_cards` now sweeps the marker,
  the static table and the placeholder snapshot/registry
  (`_discard_day0_artifacts`, hermetic test).
- **HOC is not a bonus sheet.** The CSV has exactly 193 pack names =
  Scryfall's HOB; HOC ("The Hobbit Eternal", `set_type: eternal`) is the
  Commander companion Arena downloads for Brawl. `discover_sources` now keeps
  only `masterpiece`-type children (BRR/EOS/FCA/MUL/STA/WOT all are). Day-0
  impact: 103 non-draft cards in the vocab, so the set-summary token averaged
  the wrong universe; pack ranking itself was unaffected (the pointer head
  scores in-pack cards only).

**The ultimate eval** (PLAN Phase 3: "on next-set release day the day-0 claim
gets tested for real"). HOB val, 2,202 drafts; nothing below has seen an HOB
pick (scorecards docs/scorecards/onboard/HOB/, leaderboard rows added):

| expert top-1 (top-1)        | none = day 0    | day3            | week1           | full            |
|-----------------------------|-----------------|-----------------|-----------------|-----------------|
| EXP-033 (production trunk)  | .5387 (.5453)   | .6644 (.6536)   | .6801 (.6636)   | .6855 (.6675)   |
| EXP-030 (v1 trunk)          | .5358 (.5435)   | —               | .6754 (.6595)   | —               |
| best heuristic              | rarity .3255    | —               | gih-greedy .3844| alsa .4637      |

Reference: EXP-033 on MSH (in corpus) full .6955 expert (harness; the
tracker printed .6962); EXP-013 (MSH-only
best) .6985 on test. Reading: the day-0 claim holds — .54 with zero stats
beats every stats-armed heuristic by ≥ 8 pts and matches MSH's day-0
(.5239 test / .537 val). Three days of data recover 12.5 of the 15-pt gap;
week-1 stats are essentially final for a popular set (GIH-WR corr .991 with
full-season, median 4,025 games/card at week 1, 2,497 at day 3). Zero-shot
full sits 1.1 pt under MSH-in-corpus — that gap is the fine-tune's job.
Watchdogs: rare-take 1.52× expert at day 0 (1.07× by week 1 — without stats
the trunk leans on rarity), ECE ≤ .025 throughout, skill-gap −.009 at day 0
(same sign as MSH day-0 −.011: without stats the trunk cannot separate expert
taste — the text lever's job, §3 below).

Deck builder, HOB val draft decks (53,242 builds extracted): EXP-126 trophy-F1
none .8477 / full .8855; EXP-135 full .8887; gih-in-lane-build .8291; rebuild
ceiling .9590. Third holdout, same shape as EOE (.8347 / .8981) and MSH
(.8298 / .8871). Sealed (15,152 builds, 552 val): EXP-135 full .6809 / none
.4957 vs ceiling .8949 and gih-in-lane .5617 (TLA .7576, EOE .6328).

**Recipe review** (owner ask: anything big we're getting wrong, or quick wins,
before the HOB runs). Solid and not worth touching: the v2 architecture, the
eval suite with its skill slices and watchdogs, split/quarantine hygiene,
stat masks + stat-dropout (day-0 is demonstrably a trained condition), soft
weighting on the deck side, LoRA ≈ full-FT, the deck recipe probed to plateau.
What is not:
1. **The draft trunk was never hill-climbed.** P3.T5 is still unchecked;
   EXP-033 is the P3.T4 "pretrain-small" recipe run once more on corpus v2.
   Its tensorboard curve was still rising when the 20k-step cosine reached
   zero (MSH expert none .5324@13k → .5370@20k, full .6942 → .6962, week1
   .6832 → .6852). 20k × 64 = 1.28M draft-samples ≈ **1.2 epochs** of the
   1.07M-draft used corpus, against 24 epochs for the deck trunk — where
   epochs 12 → 24 bought +0.32 day-1 (EXP-124). Lever: total_steps 60k
   (~5h, resumable). The tracker is now configurable (`track_set`, default
   MSH for provenance); new runs track HOB — MSH is in-corpus, so EXP-033's
   "zero-shot" MSH signal was an in-corpus fit.
2. **The pretrain still hard-filters by skill, against the project's own
   repeated finding.** `skill_filter 0.54/50` discards 1.47M of the 3.07M
   train drafts, then the 40k cap discards 0.53M more: the trunk trains on
   35% of the corpus. Both halves of the project found soft weighting beats
   hard curation (EXP-013 vs the filtered runs; EXP-105 on decks), the deck
   trunk trains unfiltered, and the production-lock entry recorded it as a
   lesson — but the pretrain configs never absorbed it. Lever: drop
   skill_filter (importance weights stay). At the same 40k cap this is the
   clean equal-volume test of the hypothesis.
3. **Card text is invisible, and that is the entire day-0 gap.** none → day3
   is 12.5 pts. No static feature separates Thranduil's Decree from Sound
   the Trumpets (2026-08-11 analysis); the P3.T5 "ceiling-raiser" lever
   (frozen sentence embedding of oracle text appended to the static block)
   was never run. Literature check: arXiv 2607.04782 (Jul 2026) builds
   set-contextualised card embeddings from oracle text with a 768-d text
   encoder; UrzaGPT (arXiv 2508.08382, LLM + LoRA on draft logs) reports 66%
   fine-tuned and GPT-4o 43% zero-shot — under our supervised numbers, so the
   LLM route is unattractive and the feature route is the one to take. HOB is
   now an honest yardstick for it (day-0 .5387 to beat).
4. Minor: no bf16 autocast despite the P2 design note (throughput only);
   `MIN_GAMES_FOR_WR = 5` is generous; EXP-033 had no leaderboard row (its
   MSH rows are added below). Not a lever: stat-noise augmentation — week-1
   stats are near-final (above) and day-3 costs only 1.6 pt.

Proposed next runs — configs written, NOT launched, owner picks the order:
EXP-034 (EXP-033 recipe, 60k steps, track HOB), EXP-035 (= 034 minus the
skill filter), EXP-036 (full-FT EXP-033 → HOB: the current-set drafter;
expect ≥ .69 expert from the MSH FT curves), EXP-138 (EXP-127 recipe:
EXP-126 → HOB deck fine-tune, ~8 min). The text-embedding lever is a P3.T5
proposal: a new static-feature block + scaler + a pretrain (~1 day). After
any of these ship: rebuild the laptop bundle — the day-0 HOB artifacts over
there are dead (296 → 193 cards, new ids).

## 2026-09-05 — Launched: EXP-138 → EXP-036 → EXP-034 (owner: "kick off 138 and 036, queue 034 overnight")

Chained on the single MPS device in that order (logs/<EXP>.log, caffeinate).
Each run is followed by its HOB val eval into docs/scorecards/onboard/HOB/;
EXP-034 gets the three zero-shot modes. Leaderboard rows follow from those
JSONs. Bars: EXP-138 > .8855 trophy-F1 (EXP-126 zero-shot); EXP-036 > .6855
expert (EXP-033 zero-shot full); EXP-034 vs EXP-033 on HOB val, all modes
(none .5387 / week1 .6801 / full .6855).
- **EXP-138 done (2 min):** HOB val trophy-F1 **0.9140** vs EXP-126 zero-shot
  0.8855 (+2.85; ceiling 0.9590), basics-L1 1.58 vs 2.33. Bigger lift than the
  MSH gap-window estimate (+1.3) — HOB's stats-armed zero-shot started lower.
  Production HOB deck builder candidate; bundle after EXP-036.

## 2026-09-05 — Next steps for the three recipe changes (owner discussion)

Owner: the text lever "would capture quite a bit of alpha for day-0 drafts
next set" — proposal written: **docs/TEXT_EMBED_PLAN.md** (frozen bge-base
oracle-text block in the static features + a stat-reconstruction auxiliary
head so the text pathway is trained directly on 30 sets of measured card
strength; EXP-050/051/052 + EXP-139 for decks; HOB val `none` is the bar).
Training length: EXP-034 (60k) is in the overnight slot; read its curve at
2k-step resolution — still rising at 60k → EXP-034b at 120k; promote at
≥ +0.5 on any HOB mode with the others within 0.2.
Curation: owner ruling — "17lands players are mostly self-selecting good
players; we only need to reject the bottom of the barrel." Measured over the
2.81M v2 train drafts with skill metadata: the median drafter is ≈ .54 game
WR, so the 0.54/50 filter keeps 48% — it rejects the bottom HALF of an
already-selected population. EXP-035 is now that light cut (WR ≥ .46,
≥ 10 games; keeps 87%; soft weights stay); EXP-037 = no filter, the ladder's
control. Cap stays 40k so 034/035/037 are equal-volume; raising the cap is
the next lever if the light cut wins (more sets would then hit it).
Order after EXP-034 lands: EXP-035 → text implementation (½ day) → EXP-050
→ EXP-037 / EXP-051 as overnight slots allow.
- **EXP-036 done (16 min, early stop epoch 5):** HOB val top-1 **0.6905 /
  expert 0.7100**, NLL 0.809, ECE 0.006, rare-take 1.04× expert, colors 2.25
  (human 2.31). +2.45 expert over EXP-033 zero-shot full (.6855); above every
  drafter this project has measured on any set (EXP-013 MSH val .7020,
  EXP-031 .7047). Production HOB drafter candidate alongside EXP-138 for decks:
  `--models checkpoints/EXP-036 --deck-models checkpoints/EXP-138`.

## 2026-09-05 — EXP-034 (60k steps) NEGATIVE: training length is not the draft-trunk lever; selection signal fixed

EXP-034 = EXP-033 recipe × 3 steps (60k, 306 min), tracked on HOB. Harness,
HOB val expert top-1 (EXP-033 in brackets): none .5391 (.5387), week1 .6831
(.6801), full .6889 (.6855) — +0.0 / +0.3 / +0.3, all under the +0.5 promote
bar. EXP-033 stays the production trunk. The curve says why: every stats mode
plateaued by step 16k (≈ 1 epoch of the filtered corpus) — week1 .6828@16k
vs .6831@44k (best) vs .6804@60k. The trunk saturates on the corpus AS
FILTERED, which is exactly the EXP-035 question (light cut = 1.8× the
drafts). Two lessons kept:
1. **Longer pretraining moved the day-0 axis, not the stats axes** — the same
   shape as the deck probe (EXP-124): `none` rose .5327@16k → .5451@52k
   (+1.2 in-run), while week1/full sat flat. But best.pt is chosen on week1
   expert, so it landed on step 44k and the harness reads none .5391; the
   final checkpoint reads **.5421** (row added as EXP-034@last). Selection was
   blind to the axis that moved. `select_on: [none, week1]` (mean of expert
   top-1 over the listed modes) is now a config key — default stays the
   historical week1 rule for provenance; EXP-035/037 use the two-axis rule.
2. Eval noise at this scale: 2,202 val drafts, ±0.3 between adjacent 2k-step
   evals; promote bars of +0.5 are the right size, ±0.3 results are ties.

## 2026-09-05 — P3.T5c built and calibrated: text block + aux head land; probe says +2 realistic, not +10

Owner: "get cracking on the card text plan… any estimates, or quick
experiments to calibrate? Any ablations?" Built D1 and D2 in one pass
(commits 00aa40e + this): `data/text_features.py` (frozen bge encoders,
name-masked card strings, sha-cached parquet), `assemble(text=)` threaded
through train / corpus / loading / deck / day0 / bundle, `stat_aux_lambda`
(GIH/GP/IWD reconstruction head on the statless feature half) and
`text_dropout` in the pretrain loop, hermetic tests, and a smoke run of the
whole path (feat_dim 1539, head saved, loads through the standard eval).
sentence-transformers is the optional `[text]` extra; the laptop needs it
only to onboard a set on day 0 (the bundle ships the parquet).

Calibration (docs/TEXT_EMBED_PLAN.md §6, 15 min of CPU): text roughly
doubles what the model can know about a card's strength before anyone plays
it — ridge → GIH ρ on HOB .18 static → .40 static+text (LOSO .33 → .45) —
but the channel is bounded: the MLP read does not beat ridge, kNN is weaker,
bge-large is WORSE than bge-base (.33 vs .40), masked format wins. Two
negatives worth keeping: a text-GIH greedy bot drafts at .29 (below
rarity-first), and imputing predicted stats into EXP-033's stat block costs
1.2 pts with rare-take 2.2× — the trunk trusts stats; text must come in
through the embedding with masks honest, which is D1+D2's design.
**Estimate: +2 expert top-1 at day 0 realistic, +4 optimistic; < +0.5 is a
negative.** The unmeasured upside is pool-context reading in picks 5+.

Runs (configs written; EXP-033's 20k recipe so each is ~2h and one lever):
EXP-050 text block; EXP-051 + aux head λ 0.1; EXP-052 + text dropout 0.1
(only if 050/051 diverge on `none`). Ablations settled by the probe, not by
5-h runs: encoder (base), format (masked). Left for the runs: aux λ (0.1 vs
0.3) and PCA-128 vs raw 768, both only if 051 is positive. Deck side
EXP-139 after the draft verdict. EXP-035 (curation) still queued.
- **Launched** (owner: "are we going to do 050?"): EXP-050 → EXP-051 chained
  (logs/, caffeinate), each followed by HOB val none/week1/full + MSH val
  full (the in-corpus guard); rows land via the watcher.

## 2026-09-05 — EXP-050/051: text buys +1.2 at day 0 and costs ~1 with stats on the holdout; EXP-053 makes text a day-0 fallback

HOB val expert top-1 (EXP-033 in brackets); MSH full = the in-corpus guard:

| | none | week1 | full | MSH full |
|---|---|---|---|---|
| EXP-050 text block | .5513 (+1.26) | .6686 (−1.15) | .6737 (−1.18) | .6989 (+0.34) |
| EXP-051 + aux head λ .1 | .5509 (+1.22) | .6732 (−0.69) | .6780 (−0.75) | .6945 (−0.10) |

Reading: the day-0 gain is real and replicated (+1.2 twice — inside the
probe's "+2 realistic" band, at its low end). The stats-mode loss on the
never-seen set is the memorisation signature the plan flagged: in-corpus MSH
with stats goes UP while held-out HOB with stats goes DOWN — the model learns
text×stats combinations that fit corpus cards and do not transfer. The aux
head halves that loss (−0.7) at no day-0 cost. Bars: none ✓ (≥ +0.5),
week1/full ✗ (not within 0.2). Neither promotes. Side readings at day 0:
skill-gap −.0001 vs −.0085 (text closes half the expert-taste gap — the
hypothesis from 2026-08-11 holds), rare-take 1.47×/1.41× vs 1.52×, but ECE
.075 vs .013 — text makes the day-0 model over-confident.

Fix by construction rather than by regularisation: `text_when: statless` —
the text block is present only in stat-dropout batches and, at inference,
only when stats are missing (`zero_text_block` in every loader path, the
tracker included). week1/full then run the EXP-033 pathway exactly; text is
a day-0 fallback. EXP-053 = 051 + that; EXP-052 (text_dropout .1, the
pre-registered soft regulariser) runs after it as the comparison. Launched
053 → 052 chained with the same evals + row watcher.

## 2026-09-05 — Why the text block memorises: it is a card id. Control + noise queued (EXP-054/055)

Owner: "I was imagining the embedding space to be small and that the model
would cluster on semantic distance — in reality the dimensionality is enough
to just give us an identity per card." Measured on the 9,210 corpus cards
(8,665 names, 545 reprints): random-pair cosine .66 (sd .06), nearest
DIFFERENT card .90 — every card sits ~4 sd from anything it is not. PCA-32
keeps 58% of the variance and still separates (nearest .83 vs random .00):
any smooth injective projection keeps identity; only lossiness or noise
removes it. The semantic structure is there (probe kNN/ridge), but nothing
makes the trunk prefer it over the id when both are on offer and the id fits
the corpus better. Not a capacity problem — scale failed twice here already;
a bigger trunk memorises more.
Queued behind the 053/052 chain: **EXP-054** = EXP-050 with the text block
replaced by a fixed random unit vector per card name (all identity, no
semantics): if it reproduces 050's in-corpus gain and holdout loss with
stats, the effect is identity; its `none` axis says how much of the +1.2 was
semantic. **EXP-055** = EXP-053 + `text_noise 1.0` (per-dim jitter of 1 sd
on every text-present batch): smoothness by force. Also possible later:
coarse quantisation (K prototypes) and paraphrase views.

## 2026-09-05 — EXP-053 (text only when stats are missing): day-0 gain GONE; the stats batches are where text learns

HOB val expert (EXP-033 / EXP-051 in brackets): none **.5384** (.5387 /
.5509), week1 .6739 (.6801 / .6732), full .6832 (.6855 / .6780), MSH full
.6938 (.6955 / .6945). Bars: week1/full ✓ (within noise of 033), none ✗ —
the whole +1.2 evaporated. Reading: confining the text block to the 25% of
batches that have no stats starves it. In 050/051 the text dims sit next to
the stats on every batch, and the shared card MLP learns text↔strength
alignment with the stats as teacher; that alignment is what transfers to an
unseen set at day 0. The aux head alone (present every step in 053) did not
substitute for it. So the design must keep text in every batch and stop the
id from being memorisable instead — EXP-055 revised to EXP-051 + text_noise
1.0 (was 053 + noise; the queued chain reads the config at start). EXP-054
(random-vector control) unchanged: it tells whether the +1.2 was semantic.
EXP-052 (051 + text_dropout .1) at step 13k reads .5498 / .6749 / .6799 —
051-like; whole-block dropout is not the fix either, as predicted.
- **EXP-052 done** (051 + text_dropout .1): none .5497 (+1.1 vs 033), week1
  .6764 (−0.4), full .6799 (−0.6), MSH .6967 (+0.1). Best text run so far on
  the combined bars — whole-block dropout does trim the stats-mode loss
  (051 −0.7 → −0.4/−0.6) but not to within 0.2. Ledger of the text series,
  HOB expert (none / week1 / full): 033 .5387/.6801/.6855 · 050
  .5513/.6686/.6737 · 051 .5509/.6732/.6780 · 052 .5497/.6764/.6799 · 053
  .5384/.6739/.6832. Every always-on-text run pays ~0.5–1.2 with stats on
  the holdout; 054/055 running.

## 2026-09-05 — EXP-055 PROMOTES: text + aux head + embedding noise beats EXP-033 on every HOB mode; the control shows identity was the whole problem

HOB val expert top-1 (EXP-033 in brackets), MSH full = in-corpus guard:

| | none | week1 | full | MSH full |
|---|---|---|---|---|
| EXP-054 random-id control (= 050, random unit vectors) | .5043 (−3.4) | .6542 (−2.6) | .6589 (−2.7) | .6945 (−0.1) |
| EXP-055 = 051 + text_noise 1.0 | **.5650 (+2.6)** | **.6861 (+0.6)** | **.6898 (+0.4)** | .6965 (+0.1) |

The control: with a fixed random unit vector per card the trunk fits the
corpus exactly as well (MSH unchanged) and loses ~3 pts on every mode of the
never-seen set — a 768-d block is used as a card id, and an unseen id is
out-of-distribution noise at inference. Real text (050) is that same id
damage partly bought back by semantics (+1.3 at day 0, −1.2 with stats).
Noise removes the id and keeps the semantics: 055 is +1.4 over 051 at day
0 and turns the stats-mode losses into gains. Bars: none ≥ +0.5 ✓ (+2.6),
week1/full within 0.2 ✓ (both positive), MSH ≥ .6955 ✓. Watchdogs: ECE
.049 at day 0 (033 .013, 050 .075 — better, still over-confident), rare-take
1.54× (≈ 033), colours 2.16/2.32/2.30 vs human 2.31, skill-gap at day 0
−.002 vs −.0085. The probe's "+2 realistic, +4 optimistic" landed at +2.6.
P3.T5c lever CONFIRMED; **EXP-055 is the trunk candidate to replace EXP-033**
pending (a) the fixture easy tier, (b) the HOB fine-tune path (EXP-058), (c)
the deck side (EXP-139). Overnight chain (owner directive "keep probing"):
EXP-058 HOB full-FT from 055 (bar: ≥ EXP-036 .7100) → EXP-056 noise 2.0 →
EXP-059 = 050 + noise 1.0 (no aux: is the head needed once noise is in?) →
EXP-057 noise 0.5. Text series ledger (none / week1 / full): 033
.5387/.6801/.6855 · 050 .5513/.6686/.6737 · 051 .5509/.6732/.6780 · 052
.5497/.6764/.6799 · 053 .5384/.6739/.6832 · 054 .5043/.6542/.6589 · **055
.5650/.6861/.6898**.
- **Fixture suite (Phase-3 gate condition 4 analogue), MSH scenarios:** with
  full stats both trunks pass 17/17. In day-0 mode EXP-033 fails 4
  (easy-best-common-p1p2, easy-common-over-trap-mythic,
  medium-fixing-for-splash, medium-second-color-commit); **EXP-055 passes
  17/17** — the failures were card-reading cases. Precondition (a) for the
  production swap met; (b) EXP-058 and the overnight sweep in flight.
- **EXP-058 done (28 min, early stop epoch 9):** HOB full fine-tune from the
  EXP-055 text trunk — val top-1 **0.6957 / expert 0.7133** vs EXP-036 (from
  EXP-033) 0.6905 / 0.7100: +0.5 / +0.33, NLL .796 vs .809, ECE .009.
  Precondition (b) met: the production path keeps the text trunk's gain.
  New best drafter on any set. Production swap (HUD / bundle / onboard-set
  defaults) deferred to the morning: pick the best of 055/056/057/059 once
  the noise sweep lands, then swap once — and add a clear failure when a
  text-trained trunk meets a set with no text block and no encoder
  installed (day-0 on the laptop without the [text] extra).

## 2026-09-06 — Noise sweep: more noise is better up to 2.0; the aux head is redundant once noise is in

HOB val expert (none / week1 / full), MSH full; EXP-055 = σ 1.0 + aux:

| | none | week1 | full | MSH full | day-0 ECE |
|---|---|---|---|---|---|
| EXP-057 σ 0.5 + aux | .5593 | .6823 | .6857 | .6977 | .062 |
| EXP-055 σ 1.0 + aux | .5650 | .6861 | .6898 | .6965 | .049 |
| EXP-059 σ 1.0, no aux | .5678 | .6861 | .6888 | .6963 | .052 |
| **EXP-056 σ 2.0 + aux** | **.5675** | **.6869** | **.6930** | .6965 | **.029** |

σ 0.5 loses 0.6 at day 0 vs 1.0; 2.0 gains +0.25 / +0.1 / +0.3 over 1.0 —
under the +0.5 promote bar, so a tie by the rule, but monotone on every
mode and day-0 ECE halves (.049 → .029; EXP-033 .013). Dropping the aux head
at σ 1.0 changes nothing (059 ≈ 055): the head was a second route to the same
smoothness and noise makes it redundant. Two flat probes toward the plateau
rule (3 × < 0.3); one more on the noise axis: **EXP-060 = 056 + σ 4.0**,
launched now (machine idle). Production pick after it: the best of 056/060.
- **EXP-060 (σ 4.0):** none .5626 (−0.5 vs 056), week1 .6896 (+0.3), full
  .6942 (+0.1), MSH .6935 (−0.3); day-0 ECE .014 (= EXP-033's). Third probe
  under 0.3 on the composite → **noise axis at plateau** (P3.T5 rule).
  **Production trunk = EXP-056** (σ 2.0 + aux): best day-0 of the series,
  tie elsewhere, MSH held. Swapping HUD / bundle / onboard-set defaults now;
  EXP-033 stays in the bundle as the no-text fallback.

## 2026-09-06 — PRODUCTION SWAP: EXP-056 is the draft trunk (HUD default, bundle default, onboard-set --base)

`PROD_DRAFT_MODELS` → checkpoints/EXP-056; bundle defaults lead with EXP-056
and keep EXP-033 as the no-text fallback; `onboard_set --base` default was
still EXP-030 (!) and is now EXP-056, and the generated fine-tune config
inherits `text_features` / `text_when` from the base checkpoint (without
that a fine-tune from a text trunk would build a 771-column table against a
1,539-column trunk). README, DECKBUILDER_HANDOFF and PLAN P3.T5 updated.
Deployment note: a text trunk needs `features.text.bge-base.parquet` for the
set — the bundle ships it; day-0 onboarding builds it only with the [text]
extra installed, otherwise `TextBlockUnavailable` says what to do (or use
EXP-033). Deck side untouched (EXP-126/116/135); EXP-139 = the same lever
for the deck trunk is the next proposal.

## 2026-11-15 — Two sets behind: FRA (Reality Fracture) onboarding, TRK (Star Trek) day 0

Back after ten weeks. Scryfall: FRA released 2026-10-02, TRK 2026-11-13.
17lands S3: FRA Premier/Trad draft + game files present (dropped
2026-11-14, Premier draft 124MB gz); TRK nothing yet (403 on every file,
expected ~3 weeks after release). Intent:
- **ONB-FRA-full** = `onboard_set --set FRA --base checkpoints/EXP-056
  --finetune-mode full` (the EXP-058 recipe; HOB precedent .7133 expert).
  Zero-shot week1/full scorecards from EXP-056 come first — FRA is the second
  never-seen set for the text trunk (HOB was the selection set, so FRA is the
  cleaner read). Bar for the fine-tune: beat EXP-056 zero-shot full by ≥ +1.0
  expert (HOB: +2.0).
- **TRK**: `draftbot.data.day0 --set TRK` (Arena card DB + Scryfall, stats
  all-missing, text block built) → EXP-056 on the `--stats none` path. No
  training possible until 17lands publishes.
- Sandbag bookkeeping: HOB is still the quarantined set in pretrain_v2. FRA is
  NOT added to any pretraining corpus by this work (fine-tune only). Whether
  HOB is promoted and FRA/TRK takes over as sandbag is an owner ruling.
- Scryfall client User-Agent made generic (it carried a contact address).
- **Owner ruling (2026-11-15): HOB and FRA promoted; TRK is the new held-out
  set.** `configs/corpus/pretrain_v3.yaml` = v2 + HOB + FRA at weight 2.0,
  quarantine [TRK] (declared before any TRK data exists); v2 frozen.
  **EXP-061** = EXP-056 recipe on v3, one lever (corpus). TRK has no val data,
  so the tracker is FRA as an in-corpus proxy (`track_proxy: true`, the
  EXP-033/MSH precedent; pretrain.py's corpus assert now takes that flag) —
  its numbers are NOT zero-shot. Gate (Appendix D, adapted): (a) MSH val full
  expert ≥ .6955 (EXP-056 .6965; in-corpus for both, apples to apples);
  (b) a spot full fine-tune from EXP-061 ≥ the same from EXP-056. The real
  zero-shot read on EXP-061 waits for TRK's 17lands drop (~early December).
  Chain: `scripts/chain_061.sh` (in the repo this time) — waits for
  ONB-FRA-full, scores EXP-056 on FRA `none` (true day-0 zero-shot), builds
  corpus text features, pretrains, scores FRA/HOB ×3 modes + MSH full.
  Deck side NOT refreshed: decks_v2 / sealed_v1 still quarantine HOB and
  contain no FRA; needs FRA deck extraction + sealed download first.
- README rewritten around the current system; `docs/ARCHITECTURE.md` added
  (Ryan Saxe's model credited as origin and baseline, not as our architecture).
- **ONB-FRA-full done (81 min train, early stop epoch 13, best epoch 10).**
  FRA: 81,612 drafts / 3.43M picks, t=42. EXP-056 zero-shot on FRA val —
  week1 .6507 / expert .6717, full .6574 / .6785, ECE .007, rare-take
  1.10×, colours 2.40 vs human 2.62. Full fine-tune: **.6761 / expert
  .6991**, NLL .852 — +2.06 expert over zero-shot full, bar (≥ +1.0) ✓.
  FRA's expert subset is defined at win rate ≥ .58 (HOB/MSH differ), so
  cross-set expert numbers are not directly comparable. Almost all of the
  fine-tune gain lands in epoch 1 (.6949); epochs 2–13 add +0.4. Rows via
  `scripts/add_rows.py` (now in the repo). EXP-056 FRA `none` is scored by
  the EXP-061 chain.
- **EXP-061 done (118 min, 20k steps; pretrain_v3).** True zero-shot first —
  EXP-056 on FRA `none` (day 0): **.5408 / expert .5383**, ECE .025,
  rare-take 1.22×, colours 2.12 vs human 2.62 (HOB day 0 was .5718 / .5675).
  EXP-061, all IN-CORPUS (not zero-shot), val expert none / week1 / full:
  FRA .6176 / .6798 / .6899 (056 zero-shot: .5383 / .6717 / .6785); HOB
  .6092 / .7004 / .7042 (056 zero-shot: .5675 / .6869 / .6930; EXP-058 FT
  .7133); MSH full .6958 (056: .6965). **Gate (a) ✓** (MSH ≥ .6955).
  Reading: seeing a set's drafts is worth +8 expert at `none` and only ~+1
  with stats — with no stats the trunk still learns a set's cards from
  their feature vectors, so the in-corpus `none` number is recall, not card
  reading. The honest text number is the .538 on FRA; week-1 stats close
  13 of the 16 points to the fine-tune. Gate (b): **EXP-062** = ONB-FRA-full
  recipe from EXP-061, launched now; bar ≥ .6971 (within 0.2 of .6991).
- **EXP-062 done (49 min, early stop epoch 8, best epoch 5): gate (b) ✓.**
  FRA full fine-tune from EXP-061: val .6759 / expert **.6998** vs
  ONB-FRA-full (from EXP-056) .6761 / .6991 — a tie, bar ≥ .6971. Having FRA
  in the trunk buys nothing once you fine-tune on FRA (and little: epoch 1 is
  .6956 vs .6949). **PRODUCTION SWAP: EXP-061 is the draft trunk** — HUD
  default, bundle default, `onboard_set --base`. EXP-056 stays on disk as the
  last trunk with a measured zero-shot (HOB, FRA); EXP-061's zero-shot is
  unmeasured until TRK's 17lands drop. For drafting FRA itself the best model
  is EXP-062 / ONB-FRA-full (tie); HOB: EXP-058 .7133 still beats EXP-061's
  in-corpus .7042 — per-set fine-tunes remain worth shipping.
  Open: deck/sealed corpora refresh (decks_v2 / sealed_v1 still quarantine HOB,
  no FRA); TRK day 0 needs the Arena client updated; next text lever.

## 2026-11-15 (evening) — TRK day 0 live; deck/sealed corpus refresh queued (owner: "refresh decks first")

- **TRK day 0:** owner updated the Arena client; `draftbot.data.day0 --set
  TRK` → 341 cards / 356 grpIds, sources TRK+SDS, text block built, stats
  all-missing. EXP-061 `--stats none` loads and ranks: P1P1 over the whole set
  puts 35 mythics + 15 rares in the top 50, basics at the bottom, removal
  instants at the top of the commons — sane, and as rarity-led as day 0
  always is. Draftable now: `python -m draftbot.hud --ui window --stats none`.
- **FRA deck data:** Sealed + TradSealed game files downloaded; 101,256 draft
  builds, 20,445 sealed builds (108 dropped by the 40–43 size rule, 0
  pool-inconsistent); `data/splits/FRA.sealed.json` persisted (12,251 / 651 /
  686 entries). Sealed 5-test gate green for FRA.
- **Manifests:** `decks_v3` = decks_v2 + HOB + FRA, `sealed_v2` = sealed_v1 +
  HOB + FRA; both quarantine TRK and keep EOE held out (so the new trunks
  are comparable to the old on a set neither trained on). v2 / v1 frozen.
- **EXP-140** = EXP-126 recipe on decks_v3 (from scratch per Appendix D;
  126 took 4.9h). Gate: EOE val trophy-F1 full ≥ .8961, none ≥ .8327.
  **EXP-141** = EXP-135 recipe on decks_v3 + sealed_v2 from EXP-140 (135 took
  2.4h). Gate: EOE/sealed val full ≥ .6278, EOE draft val full ≥ .8967.
  ~7.5h total, inside the overnight budget. Chain `scripts/chain_140.sh`.
  EXP-116 (lock-and-rebuild) is MSH-only and not corpus-trained — untouched.
  (EXP-139 stays reserved for the deck-side text lever.)

### 2026-11-15 night — EXP-140 / EXP-141 verdict: gates NOT met, deck defaults unchanged

Trophy-F1 v2, val. EXP-140 3.3h (early stop ep 11/24, proxy .9045); EXP-141
2.0h (early stop ep 6/14).

| | EXP-126 | EXP-140 | bar | | EXP-135 | EXP-141 | bar |
|---|---|---|---|---|---|---|---|
| EOE draft full (unseen) | .8981 | .8981 | ≥ .8961 ✓ | | .8987 | .8957 | ≥ .8967 ✗ |
| EOE draft none (unseen) | .8347 | .8275 | ≥ .8327 ✗ | | .8327 | .8309 | — |
| EOE sealed full (unseen) | | | | | .6328 | .6375 | ≥ .6278 ✓ |
| EOE sealed none (unseen) | | | | | .5231 | .5368 | — |
| HOB draft full (now in-corpus) | .8855 | .9100 | | | .8887 | .9074 | |
| FRA draft full (now in-corpus) | .8899 | .8996 | | | .8885 | .8980 | |
| HOB sealed full (now in-corpus) | | | | | .6809 | .6735 | |
| FRA sealed full (now in-corpus) | | | | | .6436 | .6546 | |
| MSH draft full (in both) | .9004 | .8977 | | | .8989 | — | |

- **Verdict:** each model misses one arm of its pre-registered gate (140: −0.72
  at `none`; 141: −0.30 on draft full, 0.10 under the bar). Per the rule, no
  production swap: `PROD_DECK_MODELS` stays EXP-126 (+116), sealed stays EXP-135.
- **Reading:** on an unseen set the refresh is a wash with stats and slightly
  worse without — the builder, unlike the draft trunk, gets nothing from two
  more sets for generalisation (30 vs 28). What it buys is in-corpus fit:
  HOB +2.5 (close to the per-set fine-tune EXP-138 .9140), FRA +1.0.
  TRK today is the unseen/no-stats case, where EXP-126 is the better model.
- Chain bug: `${4:+--event sealed}` is ONE word under zsh, so all four sealed
  evals died on argparse; fixed in `scripts/chain_140.sh`, evals rerun by hand.
  `scripts/add_rows.py` now writes deck rows too. EXP-126/135 FRA zero-shot
  rows added for the comparison.
- Open (owner): use 140/141 for HOB/FRA only (per-set default), or drop them.
