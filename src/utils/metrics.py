"""
Metrics computation utilities.

Includes:
  - Standard classification metrics (accuracy, F1, FPR, etc.)
  - Benign protection metrics (benign_drop_rate, goodput, etc.)
  - Attack mitigation metrics (attack_mitigation_rate, attack_blocking_rate)
  - Conditional action distributions (actions on benign vs attack traffic)
  - Network performance metrics
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef,
    balanced_accuracy_score,
)


def compute_all_metrics(y_true, y_pred, y_prob=None):
    """Compute comprehensive classification metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

    if y_prob is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auc"] = 0.0

    return metrics


def compute_mitigation_metrics(actions, true_labels, attack_ratios,
                               attack_threshold=0.84):
    """Compute benign protection and attack mitigation metrics.

    Args:
        actions: array of action IDs per step
        true_labels: array of window labels (0=benign, 1=attack)
        attack_ratios: array of attack ratios per window
        attack_threshold: threshold to determine if a window is attack

    Returns:
        dict with benign_drop_rate, attack_mitigation_rate, goodput, etc.
    """
    ACTION_MITIGATION = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    ACTION_NAMES = ["Forward", "Inspect", "Mirror", "Throttle",
                    "Reroute", "Drop", "Isolate"]

    total_benign = 0
    total_attack = 0
    benign_dropped = 0
    benign_throttled = 0
    benign_forwarded = 0
    attack_mitigated = 0
    attack_forwarded = 0

    for act, label, ratio in zip(actions, true_labels, attack_ratios):
        is_attack = (ratio > attack_threshold) if attack_ratios is not None else (label == 1)
        mitigation = ACTION_MITIGATION.get(act, 0)

        if is_attack:
            total_attack += 1
            if mitigation == 2:
                attack_mitigated += 1
            elif mitigation == 0:
                attack_forwarded += 1
        else:
            total_benign += 1
            if mitigation == 2:
                benign_dropped += 1
            elif mitigation == 1:
                benign_throttled += 1
            else:
                benign_forwarded += 1

    return {
        "benign_drop_rate": benign_dropped / max(total_benign, 1),
        "benign_throttle_rate": benign_throttled / max(total_benign, 1),
        "goodput": benign_forwarded / max(total_benign, 1),
        "attack_mitigation_rate": attack_mitigated / max(total_attack, 1),
        "attack_blocking_rate": (attack_mitigated + attack_forwarded) / max(total_attack, 1) if total_attack > 0 else 0.0,
        "total_benign": total_benign,
        "total_attack": total_attack,
        "benign_forwarded": benign_forwarded,
        "benign_dropped": benign_dropped,
        "benign_throttled": benign_throttled,
        "attack_mitigated": attack_mitigated,
        "attack_forwarded": attack_forwarded,
    }


def compute_conditional_action_distribution(actions, true_labels, attack_ratios,
                                            attack_threshold=0.84,
                                            action_names=None):
    """Compute action distribution conditioned on traffic type.

    Returns:
        dict with 'benign' and 'attack' sub-dicts, each mapping action_name -> fraction
    """
    if action_names is None:
        action_names = ["Forward", "Inspect", "Mirror", "Throttle",
                        "Reroute", "Drop", "Isolate"]

    benign_counts = {name: 0 for name in action_names}
    attack_counts = {name: 0 for name in action_names}

    for act, label, ratio in zip(actions, true_labels, attack_ratios):
        is_attack = (ratio > attack_threshold)
        name = action_names[act] if act < len(action_names) else str(act)
        if is_attack:
            attack_counts[name] += 1
        else:
            benign_counts[name] += 1

    benign_total = sum(benign_counts.values())
    attack_total = sum(attack_counts.values())

    benign_dist = {k: v / max(benign_total, 1) for k, v in benign_counts.items()}
    attack_dist = {k: v / max(attack_total, 1) for k, v in attack_counts.items()}

    return {
        "benign": benign_dist,
        "attack": attack_dist,
        "benign_total": benign_total,
        "attack_total": attack_total,
    }


def compute_network_metrics(latencies, queue_lengths, packet_losses,
                            cpu_usages, resource_costs):
    """Compute network performance metrics."""
    return {
        "avg_latency": float(np.mean(latencies)),
        "max_latency": float(np.max(latencies)),
        "avg_queue_length": float(np.mean(queue_lengths)),
        "max_queue_length": float(np.max(queue_lengths)),
        "avg_packet_loss": float(np.mean(packet_losses)),
        "max_packet_loss": float(np.max(packet_losses)),
        "avg_cpu_usage": float(np.mean(cpu_usages)),
        "avg_resource_cost": float(np.mean(resource_costs)),
    }


def compute_detection_delay(true_labels, detection_results, window=10):
    """Estimate detection delay: steps from attack onset to first detection."""
    delays = []
    attack_start = None

    for i, (true, det) in enumerate(zip(true_labels, detection_results)):
        if true == 1 and attack_start is None:
            attack_start = i
        if attack_start is not None and det == 1:
            delays.append(i - attack_start)
            attack_start = None
        if attack_start is not None and (i - attack_start) > window:
            delays.append(window)
            attack_start = None

    return {
        "avg_detection_delay": float(np.mean(delays)) if delays else float(window),
        "max_detection_delay": float(np.max(delays)) if delays else float(window),
        "detection_delay_std": float(np.std(delays)) if delays else 0.0,
    }


# Mitigation level per action: 0=safe, 1=moderate, 2=aggressive
ACTION_MITIGATION = {
    0: 0,  # Forward
    1: 0,  # Inspect
    2: 0,  # Mirror
    3: 1,  # Throttle
    4: 1,  # Reroute
    5: 2,  # Drop
    6: 2,  # Isolate
}


def mean_std_report(results_list):
    """Compute mean +/- std across multiple seed runs."""
    keys = [k for k in results_list[0].keys()
            if isinstance(results_list[0][k], (int, float))]
    report = {}
    for k in keys:
        vals = [r[k] for r in results_list]
        report[f"{k}_mean"] = float(np.mean(vals))
        report[f"{k}_std"] = float(np.std(vals))
    return report


def compute_flow_aware_metrics(actions, n_benign_arr, n_attack_arr):
    """Compute per-flow-level metrics for controlled-ratio windows.

    This is the correct metric for attack-ratio sensitivity: each window
    contains a known number of benign and attack flows, and the agent's
    action applies to ALL of them equally. We aggregate per-flow outcomes
    across all windows to compute flow-level goodput, mitigation, and drop.

    Args:
        actions: array of action IDs per window
        n_benign_arr: array of benign flow counts per window
        n_attack_arr: array of attack flow counts per window

    Returns:
        dict with per-flow benign_goodput, attack_mitigation_rate, benign_drop_rate, etc.
    """
    total_benign = 0
    total_attack = 0
    benign_forwarded = 0
    benign_throttled = 0
    benign_dropped = 0
    attack_mitigated = 0
    attack_forwarded = 0

    for act, n_b, n_a in zip(actions, n_benign_arr, n_attack_arr):
        mitigation = ACTION_MITIGATION.get(int(act), 0)
        total_benign += n_b
        total_attack += n_a

        # Benign flows affected by this action
        if mitigation == 0:
            benign_forwarded += n_b
        elif mitigation == 1:
            benign_throttled += n_b
        else:
            benign_dropped += n_b

        # Attack flows affected by this action
        if mitigation == 2:
            attack_mitigated += n_a
        elif mitigation == 0:
            attack_forwarded += n_a
        else:
            # Moderate action on attack = partial mitigation (count as throttled, not forwarded)
            attack_mitigated += n_a * 0.5

    return {
        "fa_benign_goodput": benign_forwarded / max(total_benign, 1),
        "fa_goodput": benign_forwarded / max(total_benign, 1),
        "fa_benign_drop_rate": benign_dropped / max(total_benign, 1),
        "fa_benign_throttle_rate": benign_throttled / max(total_benign, 1),
        "fa_attack_mitigation_rate": attack_mitigated / max(total_attack, 1),
        "fa_attack_exposure": attack_forwarded / max(total_attack, 1),
        "fa_total_benign_flows": int(total_benign),
        "fa_total_attack_flows": int(total_attack),
    }


def compute_ssu(
    attack_mitigation_rate,
    goodput,
    benign_drop_rate,
    avg_latency,
    avg_resource_cost,
    alpha=0.30,
    beta=0.30,
    gamma=0.20,
    delta=0.10,
    eta=0.10,
):
    """Compute Service-Security Utility (SSU).

    SSU = alpha * AtkMit + beta * BenSafe - gamma * BenDrop
          - delta * LatencyNorm - eta * ResourceCostNorm

    This unified metric captures the trade-off between attack mitigation,
    benign service preservation, and operational cost. Higher is better.

    Args:
        attack_mitigation_rate: fraction of attack traffic mitigated
        goodput: fraction of benign traffic forwarded (BenSafe)
        benign_drop_rate: fraction of benign traffic dropped
        avg_latency: average decision latency
        avg_resource_cost: average resource cost
        alpha: weight for attack mitigation
        beta: weight for benign safety
        gamma: weight for benign drop penalty
        delta: weight for latency penalty
        eta: weight for resource cost penalty

    Returns:
        float: SSU score in approximately [-1, 1]
    """
    latency_norm = 1.0 / (1.0 + float(avg_latency))
    resource_norm = 1.0 / (1.0 + float(avg_resource_cost))

    ssu = (
        alpha * float(attack_mitigation_rate)
        + beta * float(goodput)
        - gamma * float(benign_drop_rate)
        - delta * latency_norm
        - eta * resource_norm
    )
    return float(ssu)


def compute_ssu_from_metrics(metrics, alpha=0.30, beta=0.30, gamma=0.20,
                             delta=0.10, eta=0.10):
    """Compute SSU from a metrics dict (convenience wrapper).

    Expects keys: attack_mitigation_rate, goodput, benign_drop_rate,
    avg_latency, avg_resource_cost.
    """
    return compute_ssu(
        attack_mitigation_rate=metrics.get("attack_mitigation_rate", 0.0),
        goodput=metrics.get("goodput", 0.0),
        benign_drop_rate=metrics.get("benign_drop_rate", 0.0),
        avg_latency=metrics.get("avg_latency", 0.0),
        avg_resource_cost=metrics.get("avg_resource_cost", 0.0),
        alpha=alpha, beta=beta, gamma=gamma, delta=delta, eta=eta,
    )
