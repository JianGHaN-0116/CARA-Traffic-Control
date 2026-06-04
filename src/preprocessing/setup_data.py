"""
Setup data symlinks: link actual dataset locations to expected data/ structure.
"""
import os
import sys


def setup_data(base_dir, dataset):
    """Create symlinks or copy references for raw data."""
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Map dataset names to actual data locations (relative to base_dir's parent)
    parent_dir = os.path.dirname(base_dir)

    dataset_paths = {
        "edge_iiotset": os.path.join(parent_dir, "Edge-IIoTset dataset", "Selected dataset for ML and DL"),
        "cicids2017": os.path.join(parent_dir, "CIC-IDS2017"),
        "cicids2018": os.path.join(parent_dir, "CIC-IDS2018", "Processed Traffic Data for ML Algorithms"),
        "nf_uq_nids_v2": os.path.join(parent_dir, "NF-UQ-NIDS-v2", "data"),
    }

    if dataset == "all":
        targets = dataset_paths
    elif dataset in dataset_paths:
        targets = {dataset: dataset_paths[dataset]}
    else:
        print(f"Unknown dataset: {dataset}")
        return

    for name, src_path in targets.items():
        dst_path = os.path.join(raw_dir, name)
        if os.path.exists(dst_path):
            print(f"  Already exists: {dst_path}")
            continue
        if os.path.exists(src_path):
            # Create symlink (junction on Windows)
            try:
                os.symlink(src_path, dst_path)
                print(f"  Linked: {dst_path} -> {src_path}")
            except OSError:
                # On Windows without admin, create a text file with the path
                with open(dst_path + ".txt", "w") as f:
                    f.write(src_path)
                print(f"  Reference saved: {dst_path}.txt -> {src_path}")
                print(f"  (Manually create junction if needed)")
        else:
            print(f"  Warning: Source path not found: {src_path}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(f"Setting up data links for: {dataset}")
    setup_data(base_dir, dataset)


if __name__ == "__main__":
    main()
