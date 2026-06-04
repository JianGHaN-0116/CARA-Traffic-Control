#!/bin/bash
# Two-Switch Mixed Bottleneck Replay Script
# 
# This script runs the two-switch mixed bottleneck replay using Mininet.
# It requires root privileges and Open vSwitch installed.
#
# Usage: sudo bash run_two_switch_mixed.sh [replay_plan.csv] [output_dir]
#
# Prerequisites:
#   - Open vSwitch installed and running
#   - Mininet installed
#   - iperf3 installed
#   - Root privileges

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPLAY_PLAN="${1:-../data/precomputed/mininet_closed_loop_xlarge_multiseed_mixed_two_switch/closed_loop_step_trace.csv}"
OUTPUT_DIR="${2:-../outputs/logs}"

mkdir -p "$OUTPUT_DIR"

echo "=== Two-Switch Mixed Bottleneck Replay ==="
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
    --topology two_switch_mixed \
    --replay-plan "$REPLAY_PLAN" \
    --output "$OUTPUT_DIR/two_switch_mixed_$(date +%Y%m%d_%H%M%S).log" \
    2>&1 | tee "$OUTPUT_DIR/two_switch_mixed_latest.log"

echo ""
echo "Replay complete. Results saved to $OUTPUT_DIR"
