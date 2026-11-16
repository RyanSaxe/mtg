#!/bin/zsh
# Deck-corpus refresh chain (journal 2026-11-15). From the repo root:
#   nohup caffeinate -is scripts/chain_140.sh > logs/chain_140.log 2>&1 &
# EXP-140 (deck trunk, decks_v3) -> evals -> EXP-141 (joint draft+sealed) -> evals.
# Resumable: rerun; each training picks up its last.pt, finished ones are skipped.
set -u
stamp() { echo "[$(date '+%F %T')] $*"; }
ev() {  # ev <exp> <set> <stats> [sealed]
  uv run python -m draftbot.eval --deck --model "checkpoints/$1" --set "$2" --split val \
    --stats-snapshot "$3" ${4:+--event sealed} --out docs/scorecards >> "logs/$1.eval.log" 2>&1 \
    || stamp "EVAL FAILED $*"
}
run() {  # run <exp>
  if [[ -f "logs/$1.wall" ]]; then stamp "SKIP $1 (already finished)"; return 0; fi
  stamp "START $1"; local t0=$(date +%s)
  local resume=(); [[ -f "checkpoints/$1/last.pt" ]] && resume=(--resume)  # deck trainer needs last.pt
  uv run python -m draftbot.train.decks --config "configs/$1.yaml" $resume >> "logs/$1.log" 2>&1
  local rc=$?
  [[ $rc -eq 0 ]] && echo $(( ($(date +%s) - t0) / 60 )) > "logs/$1.wall"
  stamp "END $1 rc=$rc wall=$(( ($(date +%s) - t0) / 60 ))min"
  return $rc
}
run EXP-140 || exit 1
for m in none full; do ev EXP-140 EOE $m; done
for s in HOB FRA MSH; do ev EXP-140 $s full; done
run EXP-141 || exit 1
for m in none full; do ev EXP-141 EOE $m; ev EXP-141 EOE $m sealed; done
for s in HOB FRA; do ev EXP-141 $s full; ev EXP-141 $s full sealed; done
stamp "CHAIN DONE"
