#!/usr/bin/env python3
"""
Run CARA-TC only and merge with existing NoControl/Greedy results
"""
import argparse
import pandas as pd
from enhanced_replay import run_enhanced_replay

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--existing-log", help="Path to existing replay_run.log with NoControl/Greedy summaries")
    args = parser.parse_args()

    plan_df = pd.read_csv(args.plan_csv)
    
    # Filter to CARA-TC only
    cara_plan = plan_df[plan_df['controller'] == 'CARA-TC'].copy()
    cara_plan.to_csv('/tmp/cara_only_plan.csv', index=False)
    
    print(f"Running CARA-TC only ({len(cara_plan)} steps)")
    
    # Run CARA-TC
    run_enhanced_replay('/tmp/cara_only_plan.csv', args.out_dir)
    
    print(f"\nCombine with existing results if needed")

if __name__ == "__main__":
    main()
