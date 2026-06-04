"""
Reproduce OVS/tc Replay Summary Tables (Tables 21-23)

This script reads pre-computed OVS/tc replay data and generates
summary tables for the paper.

Output:
    outputs/paper_tables/ovs_summary.csv
    outputs/paper_tables/ovs_summary.md
    outputs/logs/ovs_summary.log
"""
import os
import sys
import logging
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def setup_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    
    log_path = os.path.join(base_dir, "outputs", "logs", "ovs_summary.log")
    setup_logging(log_path)
    logging.info("Reproducing OVS/tc Replay Summary Tables")
    
    # Find replay data directory
    replay_dir = os.path.join(base_dir, "ovs_replay", "data")
    
    if not os.path.exists(replay_dir):
        logging.error(f"Replay data directory not found: {replay_dir}")
        return
    
    results = []
    
    # Process targeted replay
    targeted_file = os.path.join(replay_dir, "targeted", "closed_loop_summary.csv")
    if os.path.exists(targeted_file):
        logging.info("Loading single-switch targeted replay data...")
        df = pd.read_csv(targeted_file)
        for _, row in df.iterrows():
            results.append({
                "Scenario": "Single-targeted",
                "Controller": row["controller"],
                "Benign TCP (Mbps)": f"{row['benign_tcp_mean_mbps']:.1f}",
                "Benign loss (%)": f"{row['benign_loss_mean_pct']:.1f}",
                "Attack TCP (Mbps)": f"{row['attack_tcp_mean_mbps']:.1f}",
                "Attack loss (%)": f"{row['attack_loss_mean_pct']:.1f}",
            })
    
    # Process mixed replay
    mixed_file = os.path.join(replay_dir, "mixed", "closed_loop_summary.csv")
    if os.path.exists(mixed_file):
        logging.info("Loading two-switch mixed replay data...")
        df = pd.read_csv(mixed_file)
        for _, row in df.iterrows():
            results.append({
                "Scenario": "Two-switch mixed",
                "Controller": row["controller"],
                "Benign TCP (Mbps)": f"{row['benign_tcp_mean_mbps']:.1f}",
                "Benign loss (%)": f"{row['benign_loss_mean_pct']:.1f}",
                "Attack TCP (Mbps)": f"{row['attack_tcp_mean_mbps']:.1f}",
                "Attack loss (%)": f"{row['attack_loss_mean_pct']:.1f}",
            })
    
    if not results:
        logging.warning("No replay data found.")
        return
    
    df = pd.DataFrame(results)
    
    # Save results
    os.makedirs(os.path.join(base_dir, "outputs", "paper_tables"), exist_ok=True)
    csv_path = os.path.join(base_dir, "outputs", "paper_tables", "ovs_summary.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"Results saved to {csv_path}")
    
    # Print markdown table
    print("\n## Table 13: Controller-in-the-Loop OVS/tc Replay: Throughput and Loss\n")
    print("**Source:** pre-computed replay data\n")
    print("| Scenario | Controller | Benign TCP (Mbps) | Benign loss (%) | Attack TCP (Mbps) | Attack loss (%) |")
    print("|----------|------------|-------------------|-----------------|-------------------|-----------------|")
    for _, row in df.iterrows():
        print(f"| {row['Scenario']} | {row['Controller']} | {row['Benign TCP (Mbps)']} | {row['Benign loss (%)']} | {row['Attack TCP (Mbps)']} | {row['Attack loss (%)']} |")
    
    logging.info("OVS summary reproduction complete.")


if __name__ == "__main__":
    main()
