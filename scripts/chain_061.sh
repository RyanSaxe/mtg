#!/bin/zsh
# EXP-061 corpus-refresh chain (journal 2026-11-15). Run from the repo root:
#   nohup caffeinate -is scripts/chain_061.sh > logs/chain_061.log 2>&1 &
# Waits for the FRA onboarding run, scores the old trunk on FRA with no stats
# (the true zero-shot day-0 number), then pretrains EXP-061 on pretrain_v3 and
# writes its scorecards. Resumable: rerun and the pretrain picks up last.pt.
set -u
stamp() { echo "[$(date '+%F %T')] $*"; }
ev() {  # ev <model dir> <set> <stats>
  uv run python -m draftbot.eval --model "$1" --set "$2" --split val \
    --stats-snapshot "$3" --out docs/scorecards >> "logs/$(basename $1).eval.log" 2>&1 \
    || stamp "EVAL FAILED $1 $2 $3"
}
while pgrep -f "draftbot.onboard_set --set FRA" > /dev/null; do sleep 30; done
if [[ ! -f checkpoints/ONB-FRA-full/best.pt ]]; then
  stamp "ABORT: FRA onboarding left no checkpoints/ONB-FRA-full/best.pt"; exit 1
fi
stamp "FRA onboarded"
ev checkpoints/EXP-056 FRA none
stamp "START text features (pretrain_v3)"
uv run python -m draftbot.data.text_features --corpus configs/corpus/pretrain_v3.yaml \
  >> logs/EXP-061.log 2>&1 || { stamp "ABORT: text features failed"; exit 1; }
stamp "START EXP-061"
t0=$(date +%s)
uv run python -m draftbot.train.pretrain --config configs/EXP-061.yaml --resume \
  >> logs/EXP-061.log 2>&1
rc=$?
echo $(( ($(date +%s) - t0) / 60 )) > logs/EXP-061.wall
stamp "END EXP-061 rc=$rc wall=$(cat logs/EXP-061.wall)min"
[[ $rc -eq 0 ]] || exit $rc
for m in none week1 full; do ev checkpoints/EXP-061 FRA $m; ev checkpoints/EXP-061 HOB $m; done
ev checkpoints/EXP-061 MSH full
stamp "CHAIN DONE"
