#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Part E: HellaSwag improvement pipeline
#
# Full evaluation pipeline for HellaSwag inference-time optimization.
#
# Steps:
#   1. Prepare data (download HellaSwag from HuggingFace)
#   2. Run baseline evaluation (zero-shot, forced single-token, greedy)
#   3. Run optimized evaluation (few-shot + self-consistency)
#   4. Generate the results summary
#
# Usage:
#   cd improve && bash eval.sh              # run everything
#   cd improve && bash eval.sh --quick      # small subset (20 examples)
#   cd improve && bash eval.sh --skip-data  # skip data download
#
# Called by:  make improve  (from project root)
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
N_VAL=200
N_TRAIN=500
N_EXAMPLES=0          # 0 = use all downloaded examples
SKIP_DATA=false
MODEL="${OLLAMA_MODEL:-qwen2.5:7b}"
BASE_URL="${OLLAMA_URL:-http://localhost:11434}"
SEED=42

# Parse flags
for arg in "$@"; do
    case $arg in
        --quick)
            N_VAL=30
            N_TRAIN=100
            N_EXAMPLES=20
            ;;
        --skip-data)
            SKIP_DATA=true
            ;;
        --model=*)
            MODEL="${arg#*=}"
            ;;
        --seed=*)
            SEED="${arg#*=}"
            ;;
    esac
done

echo "============================================================"
echo "  HellaSwag Inference-Time Optimization Pipeline"
echo "============================================================"
echo "  Model:      $MODEL"
echo "  Ollama URL: $BASE_URL"
echo "  Seed:       $SEED"
echo "  N_VAL:      $N_VAL    N_TRAIN: $N_TRAIN"
if [ "$N_EXAMPLES" -gt 0 ]; then
    echo "  N_EXAMPLES: $N_EXAMPLES"
else
    echo "  N_EXAMPLES: all"
fi
echo "============================================================"
echo ""

# ── Verify Ollama is reachable ────────────────────────────────
if ! curl -sf "${BASE_URL}/api/tags" > /dev/null 2>&1; then
    echo "ERROR: Ollama not reachable at ${BASE_URL}" >&2
    echo "Start it first:  make serve" >&2
    exit 1
fi
echo "Ollama OK at ${BASE_URL}"
echo ""

# Build common args for infer.py
COMMON_ARGS="--model $MODEL --base-url $BASE_URL --seed $SEED"
if [ "$N_EXAMPLES" -gt 0 ]; then
    COMMON_ARGS="$COMMON_ARGS --n-examples $N_EXAMPLES"
fi

cd "$PROJECT_DIR"

# ── Step 1: Prepare Data ─────────────────────────────────────
if [ "$SKIP_DATA" = false ]; then
    echo ""
    echo "──────────────────────────────────────────"
    echo "  Step 1: Preparing HellaSwag data"
    echo "──────────────────────────────────────────"
    python3 improve/prepare_data.py --n-val "$N_VAL" --n-train "$N_TRAIN"
else
    echo ""
    echo "──────────────────────────────────────────"
    echo "  Step 1: Skipping data preparation (--skip-data)"
    echo "──────────────────────────────────────────"
fi

# ── Step 2: Baseline ─────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────"
echo "  Step 2: Baseline evaluation (zero-shot, forced single-token)"
echo "──────────────────────────────────────────"
python3 improve/infer.py --baseline $COMMON_ARGS

# ── Step 3: Fully Optimized ──────────────────────────────────
echo ""
echo "──────────────────────────────────────────"
echo "  Step 3: Optimized evaluation (few-shot + self-consistency)"
echo "──────────────────────────────────────────"
python3 improve/infer.py --optimized \
    --template-name direct \
    --n-shot 3 \
    --sc-k 5 \
    --sc-temp 0.3 \
    $COMMON_ARGS

# ── Step 4: Summary ──────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────"
echo "  Step 4: Summary"
echo "──────────────────────────────────────────"
echo ""

python3 - <<'PYEOF'
import json
from pathlib import Path

results_dir = Path("improve/results")
if not results_dir.exists():
    print("No results directory found.")
    exit(0)

print("=" * 72)
print(f"  {'Strategy':<25} {'Accuracy':>10} {'95% CI (Wilson)':>22} {'Avg Lat':>10}")
print("-" * 72)

for tag in ["baseline", "optimized"]:
    latest = results_dir / f"{tag}_latest.json"
    if latest.exists():
        with open(latest) as f:
            data = json.load(f)
        m = data["metrics"]
        ci = m.get("ci_95_wilson", [0, 0])
        print(f"  {tag:<25} {m['accuracy']:>9.4f}  "
              f"[{ci[0]:.4f}, {ci[1]:.4f}]  "
              f"{m.get('avg_latency_s', 0):>8.3f}s")

print("=" * 72)

# Compute improvement
baseline_f = results_dir / "baseline_latest.json"
optimized_f = results_dir / "optimized_latest.json"
if baseline_f.exists() and optimized_f.exists():
    with open(baseline_f) as f:
        b = json.load(f)["metrics"]
    with open(optimized_f) as f:
        o = json.load(f)["metrics"]
    delta = (o["accuracy"] - b["accuracy"]) * 100
    print(f"\n  Improvement: {b['accuracy']:.4f} -> {o['accuracy']:.4f}  "
          f"(+{delta:.2f} percentage points)")
    if delta >= 3.0:
        print("  TARGET MET: >= +3.0 pp improvement achieved!")
    else:
        print(f"  Target: +3.0 pp | Gap: {3.0 - delta:.2f} pp remaining")
PYEOF

echo ""
echo "Pipeline complete. Results saved in improve/results/"
echo ""
