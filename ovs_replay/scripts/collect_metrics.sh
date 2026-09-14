#!/bin/bash
# Collect Metrics from Replay Logs
#
# This script parses the replay log files and generates summary tables.
#
# Usage: bash collect_metrics.sh [log_dir] [output_file]

set -e

LOG_DIR="${1:-../outputs/logs}"
OUTPUT_FILE="${2:-../outputs/paper_tables/ovs_summary.csv}"

mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "=== Collecting OVS Replay Metrics ==="
echo "Log directory: $LOG_DIR"
echo "Output file: $OUTPUT_FILE"
echo ""

# Check if logs exist
if [ ! -d "$LOG_DIR" ]; then
    echo "Error: Log directory not found: $LOG_DIR"
    exit 1
fi

# Parse logs and generate summary
python3 - "$LOG_DIR" "$OUTPUT_FILE" << 'PYTHON_SCRIPT'
import sys
import os
import csv
from pathlib import Path

log_dir = Path(sys.argv[1])
output_file = sys.argv[2]

# Find all log files
log_files = list(log_dir.glob("*.log"))
if not log_files:
    print(f"No log files found in {log_dir}")
    sys.exit(1)

print(f"Found {len(log_files)} log files")

# Parse each log file
results = []
for log_file in log_files:
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract metrics using simple parsing
    # This is a simplified parser - adjust based on your log format
    scenario = "unknown"
    if "single_targeted" in log_file.name:
        scenario = "Single-targeted"
    elif "two_switch_mixed" in log_file.name:
        scenario = "Two-switch mixed"
    
    # Parse controller actions and metrics
    # (Implementation depends on log format)
    print(f"Parsing {log_file.name}...")

# Write summary CSV
if results:
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSummary written to {output_file}")
else:
    print("\nNo results to write")

PYTHON_SCRIPT

echo ""
echo "Metric collection complete."
