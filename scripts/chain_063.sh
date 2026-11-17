#!/bin/zsh
# EXP-063 chain (journal 2026-11-16): EXP-056 recipe + TXT-001 strength prior.
#   nohup caffeinate -is scripts/chain_063.sh > logs/chain_063.log 2>&1 &
# Waits for the AFR prior, pretrains on pretrain_v2 (resumable: rerun), then
# scores HOB and FRA (both unseen by v2) in all stat modes, and MSH full.
set -u
stamp() { echo "[$(date '+%F %T')] $*"; }
ev() {  # ev <model dir> <set> <stats>
  uv run python -m draftbot.eval --model "$1" --set "$2" --split val \
    --stats-snapshot "$3" --out docs/scorecards >> "logs/$(basename $1).eval.log" 2>&1 \
    || stamp "EVAL FAILED $1 $2 $3"
}
while pgrep -f "afr_prior.py" > /dev/null; do sleep 20; done
[[ -f checkpoints/TXT-001/prior.AFR.parquet ]] || { stamp "ABORT: no AFR prior"; exit 1; }
stamp "START EXP-063"
t0=$(date +%s)
uv run python -m draftbot.train.pretrain --config configs/EXP-063.yaml --resume \
  >> logs/EXP-063.log 2>&1
rc=$?
echo $(( ($(date +%s) - t0) / 60 )) > logs/EXP-063.wall
stamp "END EXP-063 rc=$rc wall=$(cat logs/EXP-063.wall)min"
[[ $rc -eq 0 ]] || exit $rc
for m in none week1 full; do ev checkpoints/EXP-063 HOB $m; ev checkpoints/EXP-063 FRA $m; done
ev checkpoints/EXP-063 MSH full
stamp "CHAIN DONE"
