#!/usr/bin/env bash
# Does the GRPO result survive more seeds and two controls? Runs everything, one step after another.
#
#   mkdir -p results/variants && nohup scripts/seeds_and_controls.sh > results/variants/run.log 2>&1 &
#   tail -f results/variants/run.log             # watch it; Ctrl-C stops watching, not the run
#   DRY_RUN=1 scripts/seeds_and_controls.sh      # print the steps without running anything
#   ONLY=random_s1 scripts/seeds_and_controls.sh # just one variant (and the summary)
#
# Every variant is an adapter on top of the fused SFT model, and each goes through the same steps:
# fuse, evaluate at full precision, quantize to 4 and 3 bits with MLX, evaluate both, then measure
# margins and flips. The variants, in the order they run:
#   grpo_s0           the GRPO adapter you already have (adapters/grpo_r16), for a like-for-like row
#   sftmore_lr1e-5    control: GRPO seed 0's 400 questions and 100 steps, with the SFT loss instead of RL
#   sftmore_lr2e-4    the same at the SFT run's learning rate, 20 times higher
#   random_s1..s5     control: a random change the same size as GRPO's in every layer (Exercise 7)
#   grpo_s1..s3       three more GRPO seeds
# scripts/variants_summary.py then compares each one with the SFT model.
#
# Takes about 5 hours and 60 GB of disk (models/variants/; delete it when you're done). Keep the Mac
# plugged in; caffeinate keeps it awake. Safe to rerun: a finished step leaves a marker in
# results/variants/done/ and is skipped, and a failed step is logged while the rest keep going. Until
# Exercise 7 is implemented the random_s* steps fail and everything else still runs.

set -u
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
OUT=results/variants
DONE=$OUT/done
mkdir -p "$DONE"
[ -z "${DRY_RUN:-}" ] && caffeinate -ims -w $$ &
FAILED=()

SFT=models/Qwen3-1.7B-mine_r16-fused

log() { echo "[$(date '+%a %H:%M')] $*"; }

# step NAME CMD...: run CMD unless NAME already finished.
step() {
  local name=$1; shift
  if [ -e "$DONE/$name.done" ]; then log "skip   $name (already done)"; return 0; fi
  if [ -n "${DRY_RUN:-}" ]; then log "would run $name: $*"; return 0; fi
  log "start  $name: $*"
  local t0=$SECONDS
  if "$@"; then
    touch "$DONE/$name.done"
    log "done   $name in $(( (SECONDS - t0) / 60 )) min"
  else
    FAILED+=("$name")
    log "FAILED $name after $(( (SECONDS - t0) / 60 )) min"
  fi
}

evaluate() { $PY -m shrinkwrap.evaluate --model "$1" --limit 1034 --out "$2"; }   # full dev set
# A step without a marker never finished, so whatever it left behind is stale. `mlx_lm convert` refuses
# to write into an existing directory.
fuse()     { rm -rf "models/variants/$1"; $PY -m mlx_lm fuse --model $SFT --adapter-path "$2" --save-path "models/variants/$1"; }
quantize() { rm -rf "models/variants/$1-q$2g64"; $PY -m mlx_lm convert --hf-path "models/variants/$1" --mlx-path "models/variants/$1-q$2g64" -q --q-bits "$2" --q-group-size 64; }
grpo() {
  # train_grpo.py always starts from step 1, so move a stopped run aside instead of mixing logs.
  [ -e "adapters/grpo_r16_s$1" ] && mv "adapters/grpo_r16_s$1" "adapters/grpo_r16_s$1.stopped-$(date +%m%d-%H%M)"
  $PY scripts/train_grpo.py --model $SFT --steps 100 --seed "$1" --out "adapters/grpo_r16_s$1"
}

# wanted NAME: true unless ONLY names a different variant.
wanted() { [ -z "${ONLY:-}" ] || [ "$ONLY" = "$1" ]; }

# variant NAME ADAPTER: fuse the adapter into the SFT model, then evaluate, quantize and measure (~14 min).
variant() {
  wanted "$1" || return 0
  local name=$1 adapter=$2 m=models/variants/$1 r=$OUT/$1
  step "fuse-$name"      fuse "$name" "$adapter"
  step "eval-$name-bf16" evaluate "$m" "$r/bf16"
  for b in 4 3; do
    step "quant-$name-q$b" quantize "$name" "$b"
    step "eval-$name-q$b"  evaluate "$m-q${b}g64" "$r/q$b"
  done
  step "margins-$name"   $PY scripts/margin_analysis.py run --ref "$m" --other "q4=$m-q4g64" "q3=$m-q3g64" --out "$r/margins"
}

# 1. The GRPO adapter you already have, through the same steps as everything below.
variant grpo_s0 adapters/grpo_r16

# 2. Control: the same extra training without RL (~15 min of training each).
for lr in 1e-5 2e-4; do
  wanted "sftmore_lr$lr" || continue
  step data-control $PY scripts/prepare_control_data.py
  step "train-sftmore_lr$lr" $PY scripts/train_sft.py --config "configs/sft_more_lr$lr.yaml"
  variant "sftmore_lr$lr" "adapters/sftmore_lr$lr"
done

# 3. Control: a random change the same size as GRPO's (no training).
for s in 1 2 3 4 5; do
  wanted "random_s$s" || continue
  step "make-random_s$s" $PY scripts/make_random_adapter.py --like adapters/grpo_r16 --seed "$s" --out "adapters/random_s$s"
  variant "random_s$s" "adapters/random_s$s"
done

# 4. Three more GRPO seeds (~43 min of training each). Each seed draws its own 400 questions.
for s in 1 2 3; do
  wanted "grpo_s$s" || continue
  step "train-grpo_s$s" grpo "$s"
  variant "grpo_s$s" "adapters/grpo_r16_s$s"
done

# 5. Every finished variant against the SFT model.
[ -z "${DRY_RUN:-}" ] && $PY scripts/variants_summary.py 2>&1 | tee "$OUT/summary.txt"

log "finished. Failed steps: ${FAILED[*]:-none}"
