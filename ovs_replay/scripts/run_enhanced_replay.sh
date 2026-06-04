#!/bin/bash
# Enhanced OVS/Mininet Controller-in-the-Loop Replay
#
# Three-switch topology, 500 steps, TCP+UDP traffic, 5 controllers
#
# Prerequisites:
#   - Open vSwitch installed and running
#   - Mininet installed
#   - iperf3 installed
#   - Root privileges (sudo)
#
# Usage:
#   Step 1: Export the enhanced replay plan (no root needed):
#     python src/experiments/export_enhanced_replay_plan.py --per-label 125 --bins 5
#
#   Step 2: Run the enhanced replay (needs root):
#     sudo python ovs_replay/topology/enhanced_replay.py \
#       --plan-csv new_experiments/enhanced_ovs_replay/enhanced_replay_plan.csv \
#       --out-dir new_experiments/enhanced_ovs_replay/results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PLAN_CSV="${1:-$BASE_DIR/new_experiments/enhanced_ovs_replay/enhanced_replay_plan.csv}"
OUT_DIR="${2:-$BASE_DIR/new_experiments/enhanced_ovs_replay/results}"

mkdir -p "$OUT_DIR"

echo "=== Enhanced OVS/Mininet Replay ==="
echo "Plan: $PLAN_CSV"
echo "Output: $OUT_DIR"
echo ""

if [ "$EUID" -ne 0 ]; then
    echo "Error: This script requires root privileges"
    echo "Usage: sudo bash $0 [plan_csv] [out_dir]"
    exit 1
fi

if ! command -v ovs-vsctl &> /dev/null; then
    echo "Error: Open vSwitch not found"
    exit 1
fi

if ! command -v mn &> /dev/null; then
    echo "Error: Mininet not found"
    exit 1
fi

echo "Starting enhanced replay..."
python3 "$SCRIPT_DIR/enhanced_replay.py" \
    --plan-csv "$PLAN_CSV" \
    --out-dir "$OUT_DIR" \
    2>&1 | tee "$OUT_DIR/enhanced_replay_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Replay complete. Results saved to $OUT_DIR"
echo ""
echo "Output files:"
echo "  - enhanced_step_trace.csv   (per-step metrics)"
echo "  - enhanced_summary.csv      (per-controller summary)"
echo "  - enhanced_family_summary.csv (per-family aggregated)"
