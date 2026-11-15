# mtg-draft-buddy

Multi-set-pretrained MTG draft + deck-building assistant (PyTorch rewrite;
began as a fork of RyanSaxe/mtg, the 2022 transformer draft AI). Overview:
`README.md`, `docs/ARCHITECTURE.md`. Held-out set: TRK (since 2026-11-15).

- **Master plan:** `docs/PLAN.md` — read its §0 ground rules at session start; work
  tasks in order; log to `docs/JOURNAL.md` (append-only) and `docs/LEADERBOARD.md`.
- **New code:** `src/draftbot/` (PyTorch + MPS). **Reference only:** `mtg/` (original TF, do not modify).
- **Env:** `uv sync --extra dev`; run things via `uv run …` (Python 3.12, torch on MPS).
- **Tests:** `uv run pytest`. Data lives under `data/` (never committed, raw/ immutable).
- **Hard guardrails** (PLAN.md §0): the held-out set (now TRK) is quarantined from all pretraining; splits by
  draft_id persisted once; leaderboard rows only from `python -m draftbot.eval`;
  seeds in configs; runs >20min must be resumable.
