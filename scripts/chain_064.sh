#!/bin/zsh
# TXT-002 -> EXP-064 chain (journal 2026-11-17): production strength prior on
# pretrain_v3, then the EXP-061 recipe with it. Both stages resumable: rerun.
#   nohup caffeinate -is scripts/chain_064.sh > logs/chain_064.log 2>&1 &
set -u
stamp() { echo "[$(date '+%F %T')] $*"; }
ev() {  # ev <model dir> <set> <stats>
  uv run python -m draftbot.eval --model "$1" --set "$2" --split val \
    --stats-snapshot "$3" --out docs/scorecards >> "logs/$(basename $1).eval.log" 2>&1 \
    || stamp "EVAL FAILED $1 $2 $3"
}
stamp "START TXT-002"
uv run python -m draftbot.train.text_strength --config configs/TXT-002.yaml >> logs/TXT-002.log 2>&1
rc=$?
stamp "END TXT-002 rc=$rc"
[[ $rc -eq 0 && -f checkpoints/TXT-002/prior.TRK.parquet ]] || { stamp "ABORT: TXT-002"; exit 1; }
stamp "START EXP-064"
t0=$(date +%s)
uv run python -m draftbot.train.pretrain --config configs/EXP-064.yaml --resume >> logs/EXP-064.log 2>&1
rc=$?
echo $(( ($(date +%s) - t0) / 60 )) > logs/EXP-064.wall
stamp "END EXP-064 rc=$rc wall=$(cat logs/EXP-064.wall)min"
[[ $rc -eq 0 ]] || exit $rc
for m in none week1 full; do ev checkpoints/EXP-064 HOB $m; ev checkpoints/EXP-064 FRA $m; done
ev checkpoints/EXP-064 MSH full
stamp "CHAIN DONE"
