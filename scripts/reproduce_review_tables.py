"""
Reproduce Core Review Tables (5, 6, 8, 9, 10)

This script reproduces the core evidence tables that reviewers are
most likely to check. All tables are reproduced from pre-computed data
included in the artifact (no dataset download required).

Tables:
    Table 6  - Feature-Ablation Stress Test
    Table 5  - Main Controller Comparison
    Table 8  - Robustness Summary (Non-Overlap + Chronological)
    Table 9  - Cross-Dataset Calibration on CIC-IDS2017
    Table 10 - Compact OVS/Mininet Replay Summary

Output:
    outputs/paper_tables/review_table6.csv  (and .md)
    outputs/paper_tables/review_table7.csv
    outputs/paper_tables/review_table9.csv
    outputs/paper_tables/review_table12.csv
    outputs/paper_tables/review_table13.csv
    outputs/paper_tables/review_table17.csv
"""
import os
import sys
import csv
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def setup_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "paper_tables")
PRECOMPUTED = os.path.join(BASE_DIR, "data", "precomputed")
FEATURE_ABLATION = os.path.join(
    BASE_DIR, "new_experiments", "feature_ablation_stress", "edge_iiotset"
)


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def df_to_md(df):
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---" for _ in cols]) + "|"
    lines = [header, sep]
    for _, row in df.iterrows():
        vals = [str(v) for v in row.values]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def save_table(df, name, caption):
    csv_path = os.path.join(OUTPUT_DIR, f"review_{name}.csv")
    md_path = os.path.join(OUTPUT_DIR, f"review_{name}.md")
    df.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write(f"## {caption}\n\n")
        f.write(df_to_md(df))
    logging.info(f"{name}: {csv_path}")


# ─── Table 6: Feature-Ablation Stress Test ───────────────────────────

def reproduce_table6():
    """Table 6: Feature-Ablation Stress Test on Edge-IIoTset."""
    import pandas as pd

    csv_path = os.path.join(FEATURE_ABLATION, "summary.csv")
    if not os.path.exists(csv_path):
        logging.error(f"Feature ablation summary not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    out = df[["variant", "num_features", "flow_accuracy", "flow_f1",
               "ra_goodput", "ra_attack_mitigation_rate",
               "ra_benign_drop_rate", "ra_avg_latency"]].copy()
    out.columns = ["Variant", "Features", "Detector Acc.", "Detector F1",
                   "BenSafe", "StrictAtkMit", "BenDrop", "SCCL"]
    out["BenSafe"] = out["BenSafe"].round(4)
    out["StrictAtkMit"] = out["StrictAtkMit"].round(4)
    out["BenDrop"] = out["BenDrop"].round(4)
    out["SCCL"] = out["SCCL"].round(4)
    out["Detector Acc."] = out["Detector Acc."].round(4)
    out["Detector F1"] = out["Detector F1"].round(4)

    save_table(out, "table6", "Table 6: Feature-Ablation Stress Test")
    print("\n### Table 6: Feature-Ablation Stress Test\n")
    print(df_to_md(out))
    return out


# ─── Table 5: Main Controller Comparison ──────────────────────────────

def reproduce_table7():
    """Table 5: Corrected In-Domain Controller Comparison on Edge-IIoTset."""
    import pandas as pd

    csv_path = os.path.join(PRECOMPUTED, "main_comparison.csv")
    df = pd.read_csv(csv_path)

    rows = []
    for _, r in df.iterrows():
        if r["bensafe_std"] > 0:
            bs = f"{r['bensafe']:.4f} ± {r['bensafe_std']:.4f}"
            am = f"{r['atkmit']:.4f} ± {r['atkmit_std']:.4f}"
            bd = f"{r['bendrop']:.4f} ± {r['bendrop_std']:.4f}"
            sc = f"{r['sccel']:.4f} ± {r['sccel_std']:.4f}"
        else:
            bs = f"{r['bensafe']:.4f}"
            am = f"{r['atkmit']:.4f}"
            bd = f"{r['bendrop']:.4f}"
            sc = f"{r['sccel']:.4f}" if pd.notna(r["sccel"]) else "--"
        rows.append({
            "Controller": r["controller"],
            "Tuning": r["tuning"],
            "BenSafe": bs,
            "StrictAtkMit": am,
            "BenDrop": bd,
            "SCCL": sc,
            "Deployable": r["deployable"],
        })

    out = pd.DataFrame(rows)
    save_table(out, "table7", "Table 5: Corrected In-Domain Controller Comparison")
    print("\\n### Table 5: Corrected In-Domain Controller Comparison\\n")
    print(df_to_md(out))
    return out


# ─── Table 8 (Non-Overlap): Non-Overlapping Window Diagnostic ──────────────────────

def reproduce_table9():
    """Table 8 (Non-Overlap Block): Non-Overlapping Window Diagnostic on Edge-IIoTset."""
    import pandas as pd

    csv_path = os.path.join(PRECOMPUTED, "nonoverlap.csv")
    df = pd.read_csv(csv_path)
    df["BenSafe"] = df["bensafe"].apply(lambda x: f"{x:.4f}")
    df["StrictAtkMit"] = df["atkmit"].apply(lambda x: f"{x:.4f}")
    df["BenDrop"] = df["bendrop"].apply(lambda x: f"{x:.4f}")
    df["SCCL"] = df["sccel"].apply(lambda x: f"{x:.4f}")
    out = df[["controller", "BenSafe", "StrictAtkMit", "BenDrop", "SCCL"]].copy()
    out.columns = ["Controller", "BenSafe", "StrictAtkMit", "BenDrop", "SCCL"]

    save_table(out, "table9", "Table 8 (Non-Overlap): Non-Overlapping Window Diagnostic")
    print("\\n### Table 8 (Non-Overlap): Non-Overlapping Window Diagnostic\\n")
    print(df_to_md(out))
    return out


# ─── Table 8 (Chronological): Chronological Split Evaluation ────────────────────────

def reproduce_table12():
    """Table 8 (Chronological Block): Chronological Split Evaluation on Edge-IIoTset."""
    import pandas as pd

    csv_path = os.path.join(PRECOMPUTED, "chronological_split.csv")
    df = pd.read_csv(csv_path)
    df["BenSafe"] = df["bensafe"].apply(lambda x: f"{x:.4f}")
    df["StrictAtkMit"] = df["atkmit"].apply(lambda x: f"{x:.4f}")
    df["BenDrop"] = df["bendrop"].apply(lambda x: f"{x:.4f}")
    df["SCCL"] = df["sccel"].apply(lambda x: f"{x:.4f}")
    out = df[["controller", "BenSafe", "StrictAtkMit", "BenDrop", "SCCL"]].copy()
    out.columns = ["Controller", "BenSafe", "StrictAtkMit", "BenDrop", "SCCL"]

    save_table(out, "table12", "Table 8 (Chronological): Chronological Split Evaluation")
    print("\\n### Table 8 (Chronological): Chronological Split Evaluation\\n")
    print(df_to_md(out))
    return out


# ─── Table 9: Cross-Dataset Calibration on CIC-IDS2017 ────────────────

def reproduce_table13():
    """Table 9: Cross-Dataset Calibration on CIC-IDS2017."""
    import pandas as pd

    csv_path = os.path.join(PRECOMPUTED, "cic_cross_dataset.csv")
    df = pd.read_csv(csv_path)

    rows = []
    for _, r in df.iterrows():
        if r["bensafe_std"] > 0:
            bs = f"{r['bensafe']:.4f} ± {r['bensafe_std']:.4f}"
            am = f"{r['atkmit']:.4f} ± {r['atkmit_std']:.4f}"
            bd = f"{r['bendrop']:.4f} ± {r['bendrop_std']:.4f}"
            sc = f"{r['sccel']:.4f} ± {r['sccel_std']:.4f}"
        else:
            bs = f"{r['bensafe']:.4f}"
            am = f"{r['atkmit']:.4f}"
            bd = f"{r['bendrop']:.4f}"
            sc = f"{r['sccel']:.4f}"
        rows.append({
            "Controller": r["controller"],
            "BenSafe": bs,
            "StrictAtkMit": am,
            "BenDrop": bd,
            "SCCL": sc,
        })

    out = pd.DataFrame(rows)
    save_table(out, "table13", "Table 9: Cross-Dataset Calibration on CIC-IDS2017")
    print("\\n### Table 9: Cross-Dataset Calibration on CIC-IDS2017\\n")
    print(df_to_md(out))
    return out


# ─── Table 10: Compact OVS/Mininet Replay Summary ───────────────────────────

def reproduce_table17():
    """Table 10: Compact OVS/Mininet Replay Summary."""
    import pandas as pd

    enhanced_dir = os.path.join(
        BASE_DIR, "new_experiments", "enhanced_ovs_replay", "results_cara"
    )
    summary_path = os.path.join(enhanced_dir, "enhanced_summary.csv")

    pre_path = os.path.join(PRECOMPUTED, "enhanced_ovs_summary.csv")

    if os.path.exists(pre_path):
        df = pd.read_csv(pre_path)
    elif os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
    else:
        logging.error("Enhanced OVS summary not found")
        return

    # Build the table matching paper format
    rows = []
    controllers = ["NoControl", "Greedy", "CARA-TC", "DQN-TFC", "PPO-TFC"]
    for c in controllers:
        row = df[df["controller"] == c]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        std_row = df[df["controller"] == f"{c}_std"]
        if len(std_row) > 0:
            s = std_row.iloc[0]
            btcp = f"{r['benign_tcp_mbps']:.1f} ± {s['benign_tcp_mbps']:.1f}"
            budp = f"{r['benign_udp_mbps']:.1f} ± {s['benign_udp_mbps']:.1f}"
            atcp = f"{r['attack_tcp_mbps']:.1f} ± {s['attack_tcp_mbps']:.1f}"
            audp = f"{r['attack_udp_mbps']:.1f} ± {s['attack_udp_mbps']:.1f}"
            bl = f"{r['benign_loss_pct']:.1f} ± {s['benign_loss_pct']:.1f}"
            asp = f"{r['attack_suppression_pct']:.1f} ± {s['attack_suppression_pct']:.1f}"
            rl = f"{r['rule_install_latency_ms']:.1f} ± {s['rule_install_latency_ms']:.1f}"
            of = f"{r['mean_ovs_flow_entries']:.1f} ± {s['mean_ovs_flow_entries']:.1f}"
            oc = f"{r['ovs_cpu_pct']:.1f} ± {s['ovs_cpu_pct']:.1f}"
        else:
            btcp = f"{r['benign_tcp_mbps']:.1f}"
            budp = f"{r['benign_udp_mbps']:.1f}"
            atcp = f"{r['attack_tcp_mbps']:.1f}"
            audp = f"{r['attack_udp_mbps']:.1f}"
            bl = f"{r['benign_loss_pct']:.1f}"
            asp = f"{r['attack_suppression_pct']:.1f}"
            rl = f"{r['rule_install_latency_ms']:.1f}"
            of = f"{r['mean_ovs_flow_entries']:.1f}"
            oc = f"{r['ovs_cpu_pct']:.1f}"

        rows.append({
            "Controller": c,
            "Benign TCP (Mbps)": btcp,
            "Benign UDP (Mbps)": budp,
            "Attack TCP (Mbps)": atcp,
            "Attack UDP (Mbps)": audp,
            "Benign Loss (%)": bl,
            "Attack Supp. (%)": asp,
            "Rule-install Lat. (ms)": rl,
            "OVS Flow Entries": of,
            "OVS CPU (%)": oc,
        })

    out = pd.DataFrame(rows)
    save_table(out, "table17", "Table 10: Compact OVS/Mininet Replay Summary")
    print("\\n### Table 10: Compact OVS/Mininet Replay Summary\\n")
    print(df_to_md(out))
    return out


# ─── Main ────────────────────────────────────────────────────────────

def main():
    log_path = os.path.join(BASE_DIR, "outputs", "logs", "review_tables.log")
    setup_logging(log_path)
    logging.info("Reproducing core review tables (6, 7, 9, 12, 13, 17)")
    ensure_output_dir()

    results = {}
    results["table6"] = reproduce_table6()
    results["table7"] = reproduce_table7()
    results["table9"] = reproduce_table9()
    results["table12"] = reproduce_table12()
    results["table13"] = reproduce_table13()
    results["table17"] = reproduce_table17()

    logging.info("All review tables reproduced successfully.")
    logging.info(f"Output directory: {OUTPUT_DIR}")

    return results


if __name__ == "__main__":
    main()
