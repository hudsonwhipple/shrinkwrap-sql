#!/usr/bin/env bash
# Does the GRPO result survive more seeds and two controls? Runs everything, one step after another.
#
#   mkdir -p results/variants && nohup scripts/seeds_and_controls.sh >> results/variants/run.log 2>&1 &
#   tail -f results/variants/run.log             # watch it; Ctrl-C stops watching, not the run
#   DRY_RUN=1 scripts/seeds_and_controls.sh      # print the steps without running anything
#   ONLY=random_s1 scripts/seeds_and_controls.sh # just one variant (and the summary), once no run is going
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
# Takes about 5.5 hours and 60 GB of disk (models/variants/; delete it when you're done). Keep the Mac
# plugged in; caffeinate keeps it awake. Safe to rerun: a finished step leaves a marker in
# results/variants/done/ and is skipped. A failed step is logged, the rest of that variant is skipped
# (so a half-trained adapter is never evaluated), and the other variants keep going. Only one run at a
# time: a second one exits with a message. To redo a variant from a new adapter, delete its markers
# in results/variants/done/ first, or its old evaluations are kept.

set -u
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
OUT=results/variants
DONE=$OUT/done
mkdir -p "$DONE"
FAILED=()

SFT=models/Qwen3-1.7B-mine_r16-fused

log() { echo "[$(date '+%a %H:%M')] $*"; }

if [ -z "${DRY_RUN:-}" ]; then
  # One run at a time: two would write the same folders and share the GPU. The check looks at the
  # process itself, so a pid file left behind by a crash or a reboot doesn't block the next run.
  PIDFILE=$OUT/.pid
  if [ -e "$PIDFILE" ] && ps -p "$(cat "$PIDFILE")" -o command= 2>/dev/null | grep -q seeds_and_controls; then
    log "Another run is already going (pid $(cat "$PIDFILE")). Not starting a second one."
    exit 1
  fi
  echo $$ > "$PIDFILE"
  trap 'rm -f "$PIDFILE"' EXIT
  caffeinate -ims -w $$ &

  # What produced these results: the code, the libraries and the machine.
  changed=$(git status --porcelain -- src scripts configs 2>/dev/null)
  log "run started. commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)${changed:+ plus uncommitted changes in src, scripts or configs}"
  log "$($PY -c 'import mlx.core as mx, mlx_lm, platform; print(f"mlx {mx.__version__}, mlx-lm {mlx_lm.__version__}, Python {platform.python_version()}, macOS {platform.mac_ver()[0]}")'), $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
fi

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

# ready STEP: did STEP finish? A variant is only built on a step that did. train_grpo.py saves its
# adapter every 25 steps, so without this check a run that crashed at step 60 would be fused,
# evaluated and reported as a full seed. Always true in a dry run, so every step gets listed.
ready() {
  [ -n "${DRY_RUN:-}" ] || [ -e "$DONE/$1.done" ] && return 0
  log "skip   what depends on $1: it did not finish"
  return 1
}

# variant NAME ADAPTER: fuse the adapter into the SFT model, then evaluate, quantize and measure (~18 min).
variant() {
  wanted "$1" || return 0
  local name=$1 adapter=$2 m=models/variants/$1 r=$OUT/$1
  step "fuse-$name"      fuse "$name" "$adapter"
  ready "fuse-$name" || return 0
  step "eval-$name-bf16" evaluate "$m" "$r/bf16"
  for b in 4 3; do
    step "quant-$name-q$b" quantize "$name" "$b"
    step "eval-$name-q$b"  evaluate "$m-q${b}g64" "$r/q$b"
  done
  step "margins-$name"   $PY scripts/margin_analysis.py run --ref "$m" --other "q4=$m-q4g64" "q3=$m-q3g64" --out "$r/margins"
}

# 1. The GRPO adapter you already have, through the same steps as everything below.
variant grpo_s0 adapters/grpo_r16

# 2. Control: the same extra training without RL (~6 min of training each).
for lr in 1e-5 2e-4; do
  wanted "sftmore_lr$lr" || continue
  step data-control $PY scripts/prepare_control_data.py
  ready data-control || continue
  step "train-sftmore_lr$lr" $PY scripts/train_sft.py --config "configs/sft_more_lr$lr.yaml"
  ready "train-sftmore_lr$lr" && variant "sftmore_lr$lr" "adapters/sftmore_lr$lr"
done

# 3. Control: a random change the same size as GRPO's (no training).
for s in 1 2 3 4 5; do
  wanted "random_s$s" || continue
  step "make-random_s$s" $PY scripts/make_random_adapter.py --like adapters/grpo_r16 --seed "$s" --out "adapters/random_s$s"
  ready "make-random_s$s" && variant "random_s$s" "adapters/random_s$s"
done

# 4. Three more GRPO seeds (~43 min of training each). Each seed draws its own 400 questions.
for s in 1 2 3; do
  wanted "grpo_s$s" || continue
  step "train-grpo_s$s" grpo "$s"
  ready "train-grpo_s$s" && variant "grpo_s$s" "adapters/grpo_r16_s$s"
done

# 5. Every finished variant against the SFT model.
[ -z "${DRY_RUN:-}" ] && $PY scripts/variants_summary.py 2>&1 | tee "$OUT/summary.txt"

log "finished. Failed steps: ${FAILED[*]:-none}"
