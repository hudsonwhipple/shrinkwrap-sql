#!/usr/bin/env bash
# Run every experiment that doesn't have results yet, one after another, while you're away.
#
#   nohup scripts/overnight.sh > results/overnight.log 2>&1 &
#   tail -f results/overnight.log        # watch it; Ctrl-C stops watching, not the run
#   DRY_RUN=1 scripts/overnight.sh       # print the steps without running anything
#
# Takes about 4 hours. Keep the Mac plugged in; caffeinate keeps it awake until the script ends.
# Safe to rerun: each finished step leaves a marker in results/overnight/ and is skipped next time,
# and a failed step is logged while the rest keep going (steps that need its output fail quickly).
# Each step is its own process, so memory is released between steps.

set -u
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1            # otherwise Python holds its output back and the log lags
PY=.venv/bin/python
DONE=results/overnight
mkdir -p "$DONE"
[ -z "${DRY_RUN:-}" ] && caffeinate -ims -w $$ &
FAILED=()

BASE=Qwen/Qwen3-1.7B
SFT=models/Qwen3-1.7B-mine_r16-fused                       # where quant_sweep.py fuses mine_r16 too
GRPO=adapters/grpo_r16
GRPO_FUSED=models/Qwen3-1.7B-mine_r16-fused-grpo_r16-fused  # where the GRPO sweep fuses it

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

quantize() { $PY scripts/quantize_model.py "$@"; }
evaluate() { $PY -m shrinkwrap.evaluate --model "$1" --limit 1034 --out "$2"; }   # full dev set
grpo() {
  # train_grpo.py always starts from step 1, so move a stopped run aside instead of mixing logs.
  [ -e $GRPO ] && mv $GRPO "$GRPO.stopped-$(date +%m%d-%H%M)"
  $PY scripts/train_grpo.py --model $SFT --steps 100 --out $GRPO
}

# 1. The fine-tune under your own quantizers, next to the same quantizers on the base model (~40 min).
#    Full dev set, so each pair below can be compared question by question.
step fuse-sft              $PY -m mlx_lm fuse --model $BASE --adapter-path adapters/mine_r16 --save-path $SFT
step eval-sft-fused        evaluate $SFT results/sft_fused
step quant-sft-rtnreal-q4  quantize --model $SFT --method rtn --bits 4 --real --out models/Qwen3-1.7B-sft-rtnreal-q4g64
step eval-sft-rtnreal-q4   evaluate models/Qwen3-1.7B-sft-rtnreal-q4g64 results/sft_rtnreal_q4g64
step quant-sft-gptq-q3     quantize --model $SFT --method gptq --bits 3 --calib-samples 64 --out models/Qwen3-1.7B-sft-gptq-q3g64
step eval-sft-gptq-q3      evaluate models/Qwen3-1.7B-sft-gptq-q3g64 results/sft_gptq_q3g64
step quant-sft-rtnreal-q3  quantize --model $SFT --method rtn --bits 3 --real --out models/Qwen3-1.7B-sft-rtnreal-q3g64
step eval-sft-rtnreal-q3   evaluate models/Qwen3-1.7B-sft-rtnreal-q3g64 results/sft_rtnreal_q3g64
step quant-base-rtnreal-q3 quantize --method rtn --bits 3 --real --out models/Qwen3-1.7B-rtnreal-q3g64
step eval-base-rtnreal-q3  evaluate models/Qwen3-1.7B-rtnreal-q3g64 results/rtnreal_q3g64
step eval-base-gptq-q3     evaluate models/Qwen3-1.7B-gptq-q3g64 results/gptq_q3g64   # existing model; only 200 so far

# 2. GPTQ with the naive zero point, for the guide's section 4 table, on its 200 questions (~7 min).
step quant-base-gptq-q3-naive quantize --method gptq --bits 3 --naive-zero --calib-samples 64 --out models/Qwen3-1.7B-gptq-q3g64-naive
step eval-base-gptq-q3-naive  $PY -m shrinkwrap.evaluate --model models/Qwen3-1.7B-gptq-q3g64-naive --limit 200 --out results/gptq_q3g64_naive_n200

# 3. The planned sweep with MLX's quantizer: base, adapter on the quantized base, and fused then
#    quantized, at 8, 6, 4 and 3 bits (~75 min). It reruns bf16 base and adapter as its references.
step sweep-sft $PY scripts/quant_sweep.py --adapter-path adapters/mine_r16 --limit 1034 --out results/sweep_sft

# 4. GRPO from the SFT model (100 steps at ~22 s each, ~40 min), then the same sweep with the SFT model
#    as the base, so every row is GRPO vs SFT at one bit width (~70 min). Its bf16-adapter run is the
#    GRPO result.
step grpo       grpo
step sweep-grpo $PY scripts/quant_sweep.py --model $SFT --adapter-path $GRPO --limit 1034 --out results/sweep_grpo
#    The fused GRPO model at full precision. Its quantized versions are made from it, so it is the one to
#    measure their cost against. (The sweep's bf16-adapter run differs from it by a few answers.)
step eval-grpo-fused evaluate $GRPO_FUSED results/grpo_fused

# 5. Your GPTQ on the GRPO model: MLX's 3-bit RTN mostly breaks models, so this is the 3-bit point
#    worth comparing with the SFT model's (~10 min).
step quant-grpo-gptq-q3 quantize --model $GRPO_FUSED --method gptq --bits 3 --calib-samples 64 --out models/Qwen3-1.7B-grpo-gptq-q3g64
step eval-grpo-gptq-q3  evaluate models/Qwen3-1.7B-grpo-gptq-q3g64 results/grpo_gptq_q3g64

# 6. Paired comparisons (b minus a), also saved to results/overnight/comparisons.txt.
compare() {
  if [ -e "$2" ] && [ -e "$3" ]; then echo; echo "== $1"; $PY -m shrinkwrap.compare "$2" "$3"; fi
}
[ -z "${DRY_RUN:-}" ] && {
  compare "fusing the adapter (should be about 0)"  results/mine_r16.jsonl          results/sft_fused.jsonl
  compare "SFT gain, bf16"                          results/baseline_dev1034.jsonl  results/mine_r16.jsonl
  compare "SFT gain, your 4-bit RTN"                results/rtnreal_q4g64.jsonl     results/sft_rtnreal_q4g64.jsonl
  compare "SFT gain, your 3-bit GPTQ"               results/gptq_q3g64.jsonl        results/sft_gptq_q3g64.jsonl
  compare "SFT gain, your 3-bit RTN"                results/rtnreal_q3g64.jsonl     results/sft_rtnreal_q3g64.jsonl
  compare "cost of your 4-bit RTN, SFT model"       results/sft_fused.jsonl         results/sft_rtnreal_q4g64.jsonl
  compare "cost of your 3-bit GPTQ, SFT model"      results/sft_fused.jsonl         results/sft_gptq_q3g64.jsonl
  compare "GPTQ vs RTN at 3 bits, base"             results/rtnreal_q3g64.jsonl     results/gptq_q3g64.jsonl
  compare "GPTQ vs RTN at 3 bits, SFT"              results/sft_rtnreal_q3g64.jsonl results/sft_gptq_q3g64.jsonl
  compare "GRPO gain over SFT, bf16, adapter"       results/sweep_grpo/bf16-base.jsonl  results/sweep_grpo/bf16-adapter.jsonl
  compare "GRPO gain over SFT, bf16, fused"         results/sft_fused.jsonl             results/grpo_fused.jsonl
  compare "cost of MLX 4-bit, GRPO model"           results/grpo_fused.jsonl            results/sweep_grpo/q4g64-merged.jsonl
  compare "GRPO gain, MLX 4-bit, fused"             results/sweep_grpo/q4g64-base.jsonl results/sweep_grpo/q4g64-merged.jsonl
  compare "GRPO gain, your 3-bit GPTQ"              results/sft_gptq_q3g64.jsonl    results/grpo_gptq_q3g64.jsonl
} 2>&1 | tee "$DONE/comparisons.txt"

log "finished. Failed steps: ${FAILED[*]:-none}"
