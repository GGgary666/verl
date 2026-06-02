#!/usr/bin/env bash
# W4A4 QAT micro-train ablation: isolate perf gains from each optimization.
#
# Shared baseline (upstream-like core):
#   - observer update/sync interval = 1
#   - no fused RMSNorm activation FQ
#   - no HF monkey patch (no shared QKV act-quant; isolates the two knobs below)
#
# Cases (vs QAT baseline 00_baseline):
#   01_observer_interval  — only interval/sync = UPDATE_INTERVAL (default 8)
#   02_rmsnorm_fuse       — only --hf-fuse-w4a4-rms-norm-activation
#   04_all_optimized      — interval + fuse (production-style)
#   05_bf16               — pure BF16 train (--no-hf-enable-qat), upper-bound reference
#
# Optional cumulative stack (RUN_MODE=cumulative adds 2 extra runs):
#   11_baseline_plus_interval
#   12_baseline_plus_interval_fuse  (same as 04_all_optimized)
#
# Usage:
#   MODEL_PATH=/path/to/Qwen3-1.7B-Base-NVFP4-CALIB-ONLY \
#   bash exp_scripts/profile_b200_1/test_qat_opt_vs_upstream_like.sh
#
# Quick smoke:
#   STEPS=8 WARMUP=2 bash exp_scripts/profile_b200_1/test_qat_opt_vs_upstream_like.sh
#
# Multi-GPU:
#   NPROC=8 MODEL_PATH=... bash exp_scripts/profile_b200_1/test_qat_opt_vs_upstream_like.sh
#
# Run a subset:
#   CASES="00_baseline,01_observer_interval,04_all_optimized,05_bf16" bash ...
#
# RUN_MODE=isolated (default) | cumulative | all
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${VERL_ROOT}" || exit 1

MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3-1.7B-Base-NVFP4-CALIB-ONLY}"
BASELINE_UPDATE_INTERVAL="${BASELINE_UPDATE_INTERVAL:-1}"
BASELINE_SYNC_INTERVAL="${BASELINE_SYNC_INTERVAL:-1}"
UPDATE_INTERVAL="${UPDATE_INTERVAL:-8}"
SYNC_INTERVAL="${SYNC_INTERVAL:-8}"
STEPS="${STEPS:-24}"
WARMUP="${WARMUP:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEQ_LEN="${SEQ_LEN:-1024}"
FAKE_QUANT_KERNEL="${FAKE_QUANT_KERNEL:-nvfp4}"
OPTIMIZER="${OPTIMIZER:-adamw}"
NPROC="${NPROC:-1}"
FORWARD_ONLY="${FORWARD_ONLY:-false}"
HF_USE_TILED_MLP="${HF_USE_TILED_MLP:-false}"
RUN_MODE="${RUN_MODE:-isolated}"
RUN_BF16="${RUN_BF16:-true}"
CASES="${CASES:-}"
LOG_DIR="${LOG_DIR:-${VERL_ROOT}/qat_profile_traces/qat_ablation}"
mkdir -p "${LOG_DIR}"

COMMON_ARGS=(
  --stack hf
  --model-path "${MODEL_PATH}"
  --mode w4a4
  --hf-enable-qat
  --hf-attn-implementation flash_attention_2
  --activation-observer static_minmax
  --fake-quant-kernel-impl "${FAKE_QUANT_KERNEL}"
  --batch-size "${BATCH_SIZE}"
  --seq-len "${SEQ_LEN}"
  --steps "${STEPS}"
  --warmup "${WARMUP}"
  --optimizer "${OPTIMIZER}"
  --no-hf-use-monkey-patch
  --no-enable-batched-amax-sync
)

if [[ "${FORWARD_ONLY}" == "true" ]]; then
  COMMON_ARGS+=(--forward-only --forward-only-no-grad)
fi
if [[ "${HF_USE_TILED_MLP}" == "true" ]]; then
  COMMON_ARGS+=(--hf-use-tiled-mlp)
fi

should_run_case() {
  local tag="$1"
  if [[ -z "${CASES}" ]]; then
    return 0
  fi
  local c
  IFS=',' read -ra _wanted <<< "${CASES}"
  for c in "${_wanted[@]}"; do
    c="${c// /}"
    if [[ "${c}" == "${tag}" ]]; then
      return 0
    fi
  done
  return 1
}

run_profile() {
  local tag="$1"
  local update_iv="$2"
  local sync_iv="$3"
  local fuse="$4"        # true | false
  local verify_fuse="$5" # true | false

  if ! should_run_case "${tag}"; then
    echo "SKIP case ${tag} (not in CASES)"
    return 0
  fi

  local log_file="${LOG_DIR}/${tag}.log"
  local -a extra_args=(
    --activation-observer-update-interval "${update_iv}"
    --activation-observer-sync-interval "${sync_iv}"
  )

  if [[ "${fuse}" == "true" ]]; then
    extra_args+=(--hf-fuse-w4a4-rms-norm-activation)
  else
    extra_args+=(--no-hf-fuse-w4a4-rms-norm-activation)
  fi
  if [[ "${verify_fuse}" == "true" ]]; then
    extra_args+=(--verify-fuse-w4a4)
  fi

  echo ""
  echo "========== ${tag} (update=${update_iv}, sync=${sync_iv}, fuse=${fuse}) =========="
  set -x
  if [[ "${NPROC}" -gt 1 ]]; then
    torchrun --nproc_per_node="${NPROC}" --standalone \
      scripts/profile_qat_train_step.py \
      "${COMMON_ARGS[@]}" \
      "${extra_args[@]}" \
      2>&1 | tee "${log_file}"
  else
    python3 scripts/profile_qat_train_step.py \
      "${COMMON_ARGS[@]}" \
      "${extra_args[@]}" \
      2>&1 | tee "${log_file}"
  fi
  set +x

  if [[ "${verify_fuse}" == "true" ]] && ! grep -q "FUSE W4A4 CHECK: PASSED" "${log_file}"; then
    echo "ERROR: ${tag} failed FUSE W4A4 CHECK (see ${log_file})"
    exit 1
  fi
}

run_bf16_profile() {
  local tag="05_bf16"

  if [[ "${RUN_BF16}" != "true" ]]; then
    echo "SKIP ${tag} (RUN_BF16=false)"
    return 0
  fi
  if ! should_run_case "${tag}"; then
    echo "SKIP case ${tag} (not in CASES)"
    return 0
  fi

  local log_file="${LOG_DIR}/${tag}.log"
  local -a bf16_args=(
    --stack hf
    --model-path "${MODEL_PATH}"
    --mode w4a4
    --no-hf-enable-qat
    --dtype bf16
    --hf-attn-implementation flash_attention_2
    --batch-size "${BATCH_SIZE}"
    --seq-len "${SEQ_LEN}"
    --steps "${STEPS}"
    --warmup "${WARMUP}"
    --optimizer "${OPTIMIZER}"
    --no-hf-use-monkey-patch
  )
  if [[ "${FORWARD_ONLY}" == "true" ]]; then
    bf16_args+=(--forward-only --forward-only-no-grad)
  fi
  if [[ "${HF_USE_TILED_MLP}" == "true" ]]; then
    bf16_args+=(--hf-use-tiled-mlp)
  fi

  echo ""
  echo "========== ${tag} (pure BF16, hf_enable_qat=false) =========="
  set -x
  if [[ "${NPROC}" -gt 1 ]]; then
    torchrun --nproc_per_node="${NPROC}" --standalone \
      scripts/profile_qat_train_step.py \
      "${bf16_args[@]}" \
      2>&1 | tee "${log_file}"
  else
    python3 scripts/profile_qat_train_step.py \
      "${bf16_args[@]}" \
      2>&1 | tee "${log_file}"
  fi
  set +x

  if ! grep -q "hf_enable_qat=False" "${log_file}"; then
    echo "ERROR: ${tag} did not disable QAT (hf_enable_qat=False not in log). See ${log_file}"
    exit 1
  fi
}

print_ablation_summary() {
  python3 - "${LOG_DIR}" "${BASELINE_UPDATE_INTERVAL}" "${UPDATE_INTERVAL}" <<'PY'
import os
import sys

log_dir = sys.argv[1]
baseline_u = int(sys.argv[2])
opt_u = int(sys.argv[3])

CASES = [
    ("00_baseline", "baseline (no opts)", baseline_u, baseline_u, False),
    ("01_observer_interval", f"+observer interval ({opt_u}/{opt_u})", opt_u, opt_u, False),
    ("02_rmsnorm_fuse", "+rmsnorm fuse", baseline_u, baseline_u, True),
    ("04_all_optimized", f"all ({opt_u}/{opt_u}+fuse)", opt_u, opt_u, True),
    ("11_cumulative_interval", f"cumulative +interval ({opt_u})", opt_u, opt_u, False),
    ("12_cumulative_interval_fuse", f"cumulative +interval+fuse ({opt_u})", opt_u, opt_u, True),
]


def metric(tag, name):
    path = os.path.join(log_dir, f"{tag}.log")
    raw = open(path, "r", encoding="utf-8", errors="ignore").read() if os.path.isfile(path) else ""
    import re
    m = re.search(rf"^{re.escape(name)}:\s*([0-9]+(?:\.[0-9]+)?)$", raw, re.M)
    return float(m.group(1)) if m else float("nan")


def spd(base, x):
    if base != base or x != x or x == 0:
        return float("nan")
    return base / x


def pct_saved(base, x):
    if base != base or x != x or base == 0:
        return float("nan")
    return 100.0 * (base - x) / base


rows = []
for tag, label, *_ in CASES:
    path = os.path.join(log_dir, f"{tag}.log")
    if not os.path.isfile(path):
        continue
    step = metric(tag, "mean_step_ms")
    fwd = metric(tag, "mean_fwd_ms")
    bwd = metric(tag, "mean_bwd_ms")
    tps = metric(tag, "tokens_per_s")
    rows.append((tag, label, step, fwd, bwd, tps))

if not rows:
    print("No log files found for summary.")
    raise SystemExit(0)

base_step = next((r[2] for r in rows if r[0] == "00_baseline"), rows[0][2])

bf16_row = None
bf16_path = os.path.join(log_dir, "05_bf16.log")
if os.path.isfile(bf16_path):
    bf16_row = (
        "05_bf16",
        "pure BF16 (no QAT)",
        metric("05_bf16", "mean_step_ms"),
        metric("05_bf16", "mean_fwd_ms"),
        metric("05_bf16", "mean_bwd_ms"),
        metric("05_bf16", "tokens_per_s"),
    )

print("\n=== Ablation table (post-warmup, vs QAT baseline 00) ===")
print(f"{'case':<28} {'step_ms':>10} {'fwd_ms':>10} {'bwd_ms':>10} {'tok/s':>10} {'vs_qat0':>8} {'save%':>8}")
print("-" * 86)
if bf16_row:
    tag, label, step, fwd, bwd, tps = bf16_row
    print(
        f"{label:<28} {step:10.3f} {fwd:10.3f} {bwd:10.3f} {tps:10.2f} "
        f"{'(ref)':>8} {'—':>8}"
    )
    print("-" * 86)
for tag, label, step, fwd, bwd, tps in rows:
    print(
        f"{label:<28} {step:10.3f} {fwd:10.3f} {bwd:10.3f} {tps:10.2f} "
        f"{spd(base_step, step):8.3f}x {pct_saved(base_step, step):7.1f}%"
    )

if bf16_row:
    bf16_step = bf16_row[2]
    print("\n=== QAT overhead vs pure BF16 (05_bf16) ===")
    print(f"{'case':<28} {'step_ms':>10} {'vs_bf16':>10} {'slowdown':>10}")
    print("-" * 62)
    for tag, label, step, *_ in [bf16_row] + rows:
        if tag == "05_bf16":
            print(f"{label:<28} {step:10.3f} {'1.000x':>10} {'—':>10}")
            continue
        ratio = step / bf16_step if bf16_step == bf16_step and bf16_step else float("nan")
        ov = step - bf16_step
        print(f"{label:<28} {step:10.3f} {ratio:10.3f}x {ov:+10.3f}ms")

print("\n=== Marginal gain vs baseline (isolated knobs) ===")
isolated = [
    ("01_observer_interval", "observer_interval"),
    ("02_rmsnorm_fuse", "rmsnorm_fuse"),
]
base = next((r for r in rows if r[0] == "00_baseline"), None)
if base:
    b_step = base[2]
    for tag, name in isolated:
        r = next((x for x in rows if x[0] == tag), None)
        if not r:
            print(f"  {name}: (not run)")
            continue
        saved = b_step - r[2]
        print(
            f"  {name}: step {b_step:.3f} -> {r[2]:.3f} ms, "
            f"saved {saved:.3f} ms ({pct_saved(b_step, r[2]):.1f}%), speedup {spd(b_step, r[2]):.3f}x"
        )
    all_r = next((x for x in rows if x[0] == "04_all_optimized"), None)
    if all_r:
        saved = b_step - all_r[2]
        print(
            f"  all_combined: step {b_step:.3f} -> {all_r[2]:.3f} ms, "
            f"saved {saved:.3f} ms ({pct_saved(b_step, all_r[2]):.1f}%), speedup {spd(b_step, all_r[2]):.3f}x"
        )

print("\n=== Cumulative marginal (stack order: interval -> fuse) ===")
stack = [
    ("00_baseline", "baseline"),
    ("11_cumulative_interval", "+interval"),
    ("12_cumulative_interval_fuse", "+fuse"),
]
prev = None
for tag, label in stack:
    r = next((x for x in rows if x[0] == tag), None)
    if not r:
        continue
    if prev is None:
        print(f"  {label}: mean_step_ms={r[2]:.3f}")
    else:
        saved = prev[2] - r[2]
        print(
            f"  {label}: {prev[2]:.3f} -> {r[2]:.3f} ms, "
            f"marginal save {saved:.3f} ms ({pct_saved(prev[2], r[2]):.1f}%)"
        )
    prev = r

if base:
    b = base[2]
    parts = []
    for tag, _ in isolated:
        r = next((x for x in rows if x[0] == tag), None)
        if r and r[2] == r[2]:
            parts.append(b - r[2])
    if parts:
        print(
            f"\n(note) sum of isolated step savings = {sum(parts):.3f} ms; "
            "combined effect is not additive due to interactions."
        )
PY
}

run_isolated_cases() {
  local bu="${BASELINE_UPDATE_INTERVAL}" bs="${BASELINE_SYNC_INTERVAL}"
  local ou="${UPDATE_INTERVAL}" os="${SYNC_INTERVAL}"

  run_profile "00_baseline" "${bu}" "${bs}" false false
  run_profile "01_observer_interval" "${ou}" "${os}" false false
  run_profile "02_rmsnorm_fuse" "${bu}" "${bs}" true true
  run_profile "04_all_optimized" "${ou}" "${os}" true true
}

run_cumulative_cases() {
  local bu="${BASELINE_UPDATE_INTERVAL}" bs="${BASELINE_SYNC_INTERVAL}"
  local ou="${UPDATE_INTERVAL}" os="${SYNC_INTERVAL}"

  run_profile "00_baseline" "${bu}" "${bs}" false false
  run_profile "11_cumulative_interval" "${ou}" "${os}" false false
  run_profile "12_cumulative_interval_fuse" "${ou}" "${os}" true true
}

echo "=== QAT W4A4 ablation (observer interval + rmsnorm fuse) ==="
echo "VERL_ROOT=${VERL_ROOT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "NPROC=${NPROC} STEPS=${STEPS} WARMUP=${WARMUP} BATCH=${BATCH_SIZE} SEQ=${SEQ_LEN}"
echo "baseline observer interval: update=${BASELINE_UPDATE_INTERVAL} sync=${BASELINE_SYNC_INTERVAL}"
echo "optimized observer interval: update=${UPDATE_INTERVAL} sync=${SYNC_INTERVAL}"
echo "RUN_MODE=${RUN_MODE} RUN_BF16=${RUN_BF16} FORWARD_ONLY=${FORWARD_ONLY} LOG_DIR=${LOG_DIR}"

run_bf16_profile

case "${RUN_MODE}" in
  isolated)
    run_isolated_cases
    ;;
  cumulative)
    run_cumulative_cases
    ;;
  all)
    run_isolated_cases
    run_cumulative_cases
    ;;
  *)
    echo "ERROR: RUN_MODE must be isolated | cumulative | all (got ${RUN_MODE})"
    exit 1
    ;;
esac

print_ablation_summary

echo ""
echo "=== Done ==="
echo "Logs under: ${LOG_DIR}/"
echo "  05_bf16 (pure BF16), 00_baseline, 01_observer_interval, 02_rmsnorm_fuse, 04_all_optimized"
if [[ "${RUN_MODE}" == "cumulative" || "${RUN_MODE}" == "all" ]]; then
  echo "  11_cumulative_interval, 12_cumulative_interval_fuse"
fi
