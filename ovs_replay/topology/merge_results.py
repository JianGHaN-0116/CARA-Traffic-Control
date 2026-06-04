#!/usr/bin/env python3
"""
Merge NoControl/Greedy results from first run with CARA-TC results from second run
"""
import pandas as pd
import re
import sys

def extract_summary_from_log(log_path):
    """Extract Summary dict from log file"""
    summaries = []
    with open(log_path, 'r') as f:
        for line in f:
            if 'Summary:' in line:
                # Extract the dict part
                match = re.search(r"Summary: (\{.*\})", line)
                if match:
                    summary_str = match.group(1)
                    summary_dict = eval(summary_str)  # Safe here since we control the source
                    summaries.append(summary_dict)
    return summaries

def main():
    if len(sys.argv) < 4:
        print("Usage: python merge_results.py <first_run_log> <cara_run_log> <output_csv>")
        sys.exit(1)
    
    first_log = sys.argv[1]
    cara_log = sys.argv[2]
    output_csv = sys.argv[3]
    
    print(f"Extracting summaries from {first_log}...")
    first_summaries = extract_summary_from_log(first_log)
    print(f"Found {len(first_summaries)} summaries (NoControl, Greedy)")
    
    print(f"Extracting summaries from {cara_log}...")
    cara_summaries = extract_summary_from_log(cara_log)
    print(f"Found {len(cara_summaries)} summaries (CARA-TC)")
    
    all_summaries = first_summaries + cara_summaries
    df = pd.DataFrame(all_summaries)
    
    df.to_csv(output_csv, index=False)
    print(f"Merged summary saved to {output_csv}")
    print(f"Controllers: {df['controller'].unique().tolist()}")
    print(f"Total rows: {len(df)}")

if __name__ == "__main__":
    main()
