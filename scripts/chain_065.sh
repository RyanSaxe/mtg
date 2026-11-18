#!/bin/zsh
# Fine-tuned-embedding arms (journal 2026-11-18): EXP-066 (ft emb + prior) then
# EXP-065 (ft emb alone), both on pretrain_v2. Resumable: rerun.
#   nohup caffeinate -is scripts/chain_065.sh > logs/chain_065.log 2>&1 &
set -u
stamp() { echo "[$(date '+%F %T')] $*"; }
ev() {
  uv run python -m draftbot.eval --model "$1" --set "$2" --split val \
    --stats-snapshot "$3" --out docs/scorecards >> "logs/$(basename $1).eval.log" 2>&1 \
    || stamp "EVAL FAILED $1 $2 $3"
}
stamp "START export"
uv run python scripts/export_ft_emb.py --afr >> logs/export_ft_emb.log 2>&1 || { stamp "ABORT: export"; exit 1; }
for x in EXP-066 EXP-065; do
  stamp "START $x"
  t0=$(date +%s)
  uv run python -m draftbot.train.pretrain --config configs/$x.yaml --resume >> logs/$x.log 2>&1
  rc=$?
  echo $(( ($(date +%s) - t0) / 60 )) > logs/$x.wall
  stamp "END $x rc=$rc wall=$(cat logs/$x.wall)min"
  [[ $rc -eq 0 ]] || continue
  for m in none week1 full; do ev checkpoints/$x HOB $m; ev checkpoints/$x FRA $m; done
  ev checkpoints/$x MSH full
done
stamp "CHAIN DONE"
