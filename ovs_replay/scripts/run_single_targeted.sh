#!/bin/bash
# Single-Switch Targeted Replay Script
# 
# This script runs the single-switch targeted replay using Mininet.
# It requires root privileges and Open vSwitch installed.
#
# Usage: sudo bash run_single_targeted.sh [replay_plan.csv] [output_dir]
#
# Prerequisites:
#   - Open vSwitch installed and running
#   - Mininet installed
#   - iperf3 installed
#   - Root privileges

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPLAY_PLAN="${1:-../data/precomputed/mininet_closed_loop_xlarge_multiseed_targeted/closed_loop_step_trace.csv}"
OUTPUT_DIR="${2:-../outputs/logs}"

mkdir -p "$OUTPUT_DIR"

echo "=== Single-Switch Targeted Replay ==="
echo "Replay plan: $REPLAY_PLAN"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check prerequisites
if [ "$EUID" -ne 0 ]; then
    echo "Error: This script requires root privileges"
    echo "Usage: sudo bash $0"
    exit 1
fi

if ! command -v ovs-vsctl &> /dev/null; then
    echo "Error: Open vSwitch not found"
    echo "Install with: apt-get install openvswitch-switch"
    exit 1
fi

if ! command -v mn &> /dev/null; then
    echo "Error: Mininet not found"
    echo "Install with: apt-get install mininet"
    exit 1
fi

# Run the replay
echo "Starting Mininet replay..."
python3 "$SCRIPT_DIR/../topology/run_replay.py" \
    --topology single_targeted \
    --replay-plan "$REPLAY_PLAN" \
    --output "$OUTPUT_DIR/single_targeted_$(date +%Y%m%d_%H%M%S).log" \
    2>&1 | tee "$OUTPUT_DIR/single_targeted_latest.log"

echo ""
echo "Replay complete. Results saved to $OUTPUT_DIR"
