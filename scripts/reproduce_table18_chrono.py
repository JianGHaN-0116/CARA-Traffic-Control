"""
Reproduce Table 18: Symmetric Chronological Split Evaluation

This script reproduces the 2×2 symmetric chronological comparison.

Output:
    outputs/paper_tables/table18.csv
    outputs/paper_tables/table18.md
    outputs/logs/table18.log
"""
import os
import sys
import logging

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
    
    log_path = os.path.join(base_dir, "outputs", "logs", "table18.log")
    setup_logging(log_path)
    logging.info("Reproducing Table 18: Symmetric Chronological Split Evaluation")
    
    # Check if pre-computed results exist
    results_path = os.path.join(base_dir, "data", "precomputed", "chronological_symmetric.csv")
    if not os.path.exists(results_path):
        # Try alternative location
        results_path = os.path.join(base_dir, "..", "DRL-Edge-Traffic-Security", 
                                   "new_experiments", "chronological_symmetric", 
                                   "chronological_symmetric.csv")
    
    if os.path.exists(results_path):
        import pandas as pd
        df = pd.read_csv(results_path)
        logging.info(f"Loaded pre-computed results from {results_path}")
        
        os.makedirs(os.path.join(base_dir, "outputs", "paper_tables"), exist_ok=True)
        csv_path = os.path.join(base_dir, "outputs", "paper_tables", "table18.csv")
        df.to_csv(csv_path, index=False)
        
        print("\n## Table 18: Symmetric Chronological Split Evaluation\n")
        print("| Controller | Tuning source | BenSafe | StrictAtkMit | BenDrop |")
        print("|------------|---------------|---------|--------------|---------|")
        for _, row in df.iterrows():
            print(f"| {row['controller']} | {row['tuning_source']} | {row['bensafe']:.4f} | {row['atkmit']:.4f} | {row['bendrop']:.4f} |")
        print("\n*Note: Rows for ContextualBandit are extra diagnostic rows and are not included in the compact paper Table 18.*")
    else:
        logging.warning("Pre-computed results not found. Please run chronological_symmetric.py first.")
        logging.info("Expected location: data/precomputed/chronological_symmetric.csv")
    
    logging.info("Table 18 reproduction complete.")


if __name__ == "__main__":
    main()
