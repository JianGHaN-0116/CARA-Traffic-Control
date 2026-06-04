"""
New non-RL baseline controllers requested by reviewer:

1. CostSensitiveClassifier: XGBoost action classifier trained with reward-matrix
   sample weights. Predicts the best action from detector summaries + edge state.

2. ContextualBandit: Online ε-greedy linear contextual bandit trained on
   sequential per-step feedback. Only uses current state, no transition model.

3. DecisionTreePolicy: Shallow decision tree (max_depth=5) predicting actions
   from detector summaries + edge state. Interpretable rule-list baseline.

All three use the SAME state representation as DQN-TFC and CARA-TC,
and are evaluated through the standard evaluate_policy() path.

Usage:
    python -m src.experiments.new_baselines [dataset] [max_eval_steps]
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import yaml
import joblib
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.envs.edge_network_env import EdgeTrafficSecurityEnv, ACTION_NAMES
from src.utils.metrics import (
    compute_all_metrics, compute_mitigation_metrics,
    compute_ssu,
)
from src.utils.path_helpers import (
    resolve_attack_threshold, resolve_detector_model_path, resolve_state_dim,
)

# Reward parameters (mirrors reward_function.py defaults)
BENIGN_FORWARDED_REWARD = 1.0
BENIGN_THROTTLED_PENALTY = -7.0
BENIGN_DROPPED_PENALTY = -15.0
ATTACK_MITIGATED_REWARD = 5.0
ATTACK_DETECTED_REWARD = 2.0
ATTACK_MISSED_PENALTY = -8.0

ACTION_MITIGATION = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}


def oracle_reward(action, is_attack, detection_result):
    """Compute the reward that would be received for a given (action, truth, detection).

    This uses the reward matrix from the paper (Eq. 4) to determine the
    best action for a given window during training. It is used ONLY for
    labeling training data, not during evaluation.
    """
    mitigation = ACTION_MITIGATION.get(action, 0)

    if is_attack:
        if mitigation == 2:
            return ATTACK_MITIGATED_REWARD
        elif detection_result == 1:
            return ATTACK_DETECTED_REWARD
        else:
            return -ATTACK_MISSED_PENALTY
    else:
        if mitigation == 2:
            return BENIGN_DROPPED_PENALTY
        elif mitigation == 1:
            return BENIGN_THROTTLED_PENALTY
        else:
            return BENIGN_FORWARDED_REWARD


def compute_best_action_for_window(window, attack_threshold=0.84):
    """Given a window with ground-truth info, compute the best action.

    This uses the ground-truth attack_ratio (only for training data labeling)
    to compute which action maximizes the immediate reward.
    """
    attack_ratio = window.get("attack_ratio", 0.0)
    detector_confidence = window.get("detector_confidence", 0.5)
    is_attack = attack_ratio > attack_threshold

    best_action = 0
    best_reward = -1e9

    for action in range(7):
        # Simulate detection result based on action-conditioned threshold
        thresholds = {0: 0.5, 1: 0.3, 2: 0.4, 3: 0.5, 4: 0.5, 5: 0.5, 6: 0.5}
        detection_result = 1 if detector_confidence >= thresholds.get(action, 0.5) else 0
        reward = oracle_reward(action, is_attack, detection_result)
        if reward > best_reward:
            best_reward = reward
            best_action = action

    return best_action


def build_labeled_training_data(window_path, state_dim, attack_threshold=0.84):
    """Build (state, best_action) pairs from training windows.

    The state is the full observation vector (including detector summaries
    and edge state). The label is the oracle best action computed from
    ground-truth attack_ratio (used ONLY for training, never for evaluation).
    """
    with open(window_path, "rb") as f:
        windows = pickle.load(f)

    X_list = []
    y_list = []

    # Simulated initial edge state (same as env.reset())
    edge_cpu = 0.3
    edge_memory = 0.3
    queue_length = 0.1
    link_utilization = 0.3
    packet_loss = 0.0

    for window in windows:
        traffic_state = window["state"]
        detector_confidence = window.get("detector_confidence", 0.5)
        detector_estimated_ratio = window.get("detector_estimated_ratio", 0.0)

        edge_state = np.array([
            edge_cpu, edge_memory, queue_length,
            link_utilization, packet_loss,
            detector_estimated_ratio, detector_confidence,
        ], dtype=np.float32)

        state = np.concatenate([traffic_state, edge_state]).astype(np.float32)

        # Pad/truncate to state_dim
        if len(state) < state_dim:
            state = np.pad(state, (0, state_dim - len(state)))
        elif len(state) > state_dim:
            state = state[:state_dim]

        best_action = compute_best_action_for_window(window, attack_threshold)

        X_list.append(state)
        y_list.append(best_action)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=int)


# ═══════════════════════════════════════════════════════════════════════════
# Baseline 1: Cost-Sensitive Action Classifier (XGBoost)
# ═══════════════════════════════════════════════════════════════════════════

class CostSensitiveClassifierPolicy:
    """XGBoost classifier trained to predict the best action from state.

    Training uses the reward matrix to label each window with its oracle
    best action (computed from ground-truth attack_ratio). The classifier
    learns a mapping from detector summaries + edge state → action.

    This answers the reviewer's question: "does this problem need sequential
    RL, or is a cost-sensitive supervised decision problem enough?"
    """

    def __init__(self, model=None, constant_action=None, label_encoder=None):
        self.model = model
        self.constant_action = constant_action
        self.label_encoder = label_encoder

    def predict(self, obs, info=None):
        if self.constant_action is not None:
            return self.constant_action
        if self.model is None:
            return 0
        obs_2d = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        y_enc = int(self.model.predict(obs_2d)[0])
        if self.label_encoder is not None:
            return int(self.label_encoder.inverse_transform([y_enc])[0])
        return y_enc

    @classmethod
    def train(cls, train_window_path, state_dim, attack_threshold=0.84,
              n_estimators=100, max_depth=4):
        """Train the cost-sensitive classifier on labeled window data."""
        from xgboost import XGBClassifier
        from sklearn.preprocessing import LabelEncoder

        X, y = build_labeled_training_data(
            train_window_path, state_dim, attack_threshold)

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            majority = int(unique_classes[0])
            print(f"  CSC: single-class training data (all={majority}), using constant predictor")
            return cls(model=None, constant_action=majority)

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,
            objective='multi:softmax',
            num_class=len(le.classes_),
            tree_method='hist',
            random_state=42,
            n_jobs=4,
        )
        model.fit(X, y_encoded)
        return cls(model=model, label_encoder=le)


# ═══════════════════════════════════════════════════════════════════════════
# Baseline 2: Contextual Bandit (ε-greedy linear)
# ═══════════════════════════════════════════════════════════════════════════

class ContextualBanditPolicy:
    """Online ε-greedy contextual bandit trained on sequential feedback.

    Uses a separate linear model (ridge regression) per action to estimate
    expected reward from state features. At each step, selects the action
    with highest predicted reward (or random with probability ε).

    This bandit models P(r | s, a) independently per action without modeling
    state transitions — a direct test of whether sequential credit assignment
    is needed.
    """

    def __init__(self, n_actions=7, n_features=None, epsilon=0.05,
                 regularization=1.0):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.regularization = regularization

        # Per-action linear models: weights (n_features,), bias
        self.weights = None  # (n_actions, n_features)
        self.biases = np.zeros(n_actions)

        # Per-action covariance matrices for online ridge regression
        self.A = None  # list of (n_features, n_features) matrices
        self.b = None  # list of (n_features,) vectors

        self.rng = np.random.RandomState(42)
        self._fitted = False

    def _init_if_needed(self, obs):
        n_features = len(np.asarray(obs).ravel())
        if self.weights is None or self.weights.shape[1] != n_features:
            self.weights = np.zeros((self.n_actions, n_features))
            self.A = [self.regularization * np.eye(n_features)
                      for _ in range(self.n_actions)]
            self.b = [np.zeros(n_features) for _ in range(self.n_actions)]

    def predict(self, obs, info=None):
        """Select action: ε-greedy with current linear estimates."""
        self._init_if_needed(obs)
        x = np.asarray(obs, dtype=np.float32).ravel()

        if not self._fitted or self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.n_actions)

        scores = self.weights @ x + self.biases
        return int(np.argmax(scores))

    def update(self, obs, action, reward):
        """Online ridge regression update for the chosen action."""
        self._init_if_needed(obs)
        x = np.asarray(obs, dtype=np.float32).ravel()
        a = int(action)

        self.A[a] += np.outer(x, x)
        self.b[a] += reward * x

        try:
            self.weights[a] = np.linalg.solve(self.A[a], self.b[a])
        except np.linalg.LinAlgError:
            pass  # Keep previous weights if singular

        self._fitted = True

    @classmethod
    def train_online(cls, train_window_path, state_dim, env_class,
                     max_train_steps=20000, attack_threshold=0.84,
                     reward_config=None, detector_model_path=None):
        """Train the bandit online on the training environment."""
        policy = cls(n_actions=7, epsilon=0.1, regularization=1.0)

        env = env_class(
            window_path=train_window_path,
            state_dim=state_dim,
            max_steps=min(max_train_steps, 20000),
            reward_config=reward_config or {"attack_threshold": attack_threshold},
            detector_model_path=detector_model_path,
            shuffle_on_reset=True,
        )

        obs, _ = env.reset()
        done = False

        while not done:
            action = policy.predict(obs)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

            policy.update(obs, action, float(reward))

        env.close()
        policy.epsilon = 0.0  # Greedy at test time
        return policy


# ═══════════════════════════════════════════════════════════════════════════
# Baseline 3: Decision Tree / Rule List Policy
# ═══════════════════════════════════════════════════════════════════════════

class DecisionTreePolicy:
    """Shallow decision tree predicting actions from detector summaries + edge state.

    Trained on the same oracle-labeled data as CostSensitiveClassifier.
    Produces an interpretable rule list that can be inspected to understand
    what conditions trigger each action. This serves as a transparent learned
    baseline between CARA-TC (validation-calibrated) and DRL (black-box).
    """

    def __init__(self, model=None, action_names=None):
        self.model = model
        self.action_names = action_names or ACTION_NAMES

    def predict(self, obs, info=None):
        if self.model is None:
            return 0
        obs_2d = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        return int(self.model.predict(obs_2d)[0])

    def export_rules(self, feature_names=None):
        """Export the decision tree as human-readable rules."""
        if self.model is None:
            return "No model trained."

        from sklearn.tree import export_text
        return export_text(
            self.model,
            feature_names=feature_names,
            max_depth=5,
        )

    @classmethod
    def train(cls, train_window_path, state_dim, attack_threshold=0.84,
              max_depth=5, min_samples_leaf=50):
        """Train a shallow decision tree on labeled window data."""
        from sklearn.tree import DecisionTreeClassifier

        X, y = build_labeled_training_data(
            train_window_path, state_dim, attack_threshold)

        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            class_weight='balanced',
        )
        model.fit(X, y)
        return cls(model=model)


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation harness (reuses the standard evaluate_policy pattern)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_policy(env, policy, attack_threshold=0.84):
    """Evaluate any policy and return unified metrics."""
    obs, _ = env.reset()
    true_labels, detection_results, rewards = [], [], []
    latencies, actions_list, attack_ratios = [], [], []
    packet_losses = []
    done = False

    while not done:
        action = policy.predict(obs)
        if isinstance(action, tuple):
            action = action[0]
        action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        true_labels.append(info["true_label"])
        detection_results.append(info["detection_result"])
        rewards.append(float(reward))
        latencies.append(info["latency"])
        actions_list.append(info["action"])
        attack_ratios.append(info["attack_ratio"])
        packet_losses.append(info["packet_loss"])

    y_true = np.array(true_labels)
    y_pred = np.array(detection_results)
    actions_arr = np.array(actions_list)
    ratios_arr = np.array(attack_ratios)

    cls = compute_all_metrics(y_true, y_pred)
    mitigation = compute_mitigation_metrics(
        actions_arr, y_true, ratios_arr, attack_threshold=attack_threshold)

    network = {
        "avg_reward": float(np.mean(rewards)),
        "avg_latency": float(np.mean(latencies)),
        "avg_packet_loss": float(np.mean(packet_losses)),
        "total_steps": len(rewards),
    }

    metrics = {}
    metrics.update(cls)
    metrics.update(mitigation)
    metrics.update(network)

    metrics["ssu"] = compute_ssu(
        attack_mitigation_rate=mitigation["attack_mitigation_rate"],
        goodput=mitigation["goodput"],
        benign_drop_rate=mitigation["benign_drop_rate"],
        avg_latency=network["avg_latency"],
        avg_resource_cost=0.0,
    )

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    dataset = sys.argv[1] if len(sys.argv) > 1 else "edge_iiotset"
    max_eval_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20000

    split_dir = os.path.join(base_dir, "data/processed", dataset)
    results_dir = os.path.join(base_dir, "results/drl_results", dataset)
    train_win = os.path.join(split_dir, "train_windows.pkl")
    test_win = os.path.join(split_dir, "test_windows.pkl")

    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(base_dir, "configs/drl_config.yaml")) as f:
        config = yaml.safe_load(f)

    state_dim = resolve_state_dim(
        base_dir, dataset, config.get("environment", {}).get("state_dim", 48))
    attack_threshold = resolve_attack_threshold(
        base_dir, dataset, config.get("window", {}).get("attack_threshold", 0.84))
    reward_config = dict(config.get("reward", {}))
    reward_config["attack_threshold"] = attack_threshold
    detector_model_path = resolve_detector_model_path(
        base_dir, dataset, config.get("common", {}).get("detector_model_path", ""))

    all_results = []

    # ── Baseline 1: Cost-Sensitive Classifier ──────────────────────────
    print("\n=== Training Cost-Sensitive Action Classifier ===")
    try:
        csc_policy = CostSensitiveClassifierPolicy.train(
            train_win, state_dim, attack_threshold,
            n_estimators=100, max_depth=4,
        )
        # Save model
        csc_path = os.path.join(results_dir, "cost_sensitive_classifier.pkl")
        joblib.dump(csc_policy.model, csc_path)
        print(f"  Model saved: {csc_path}")

        env = EdgeTrafficSecurityEnv(
            window_path=test_win, state_dim=state_dim,
            max_steps=max_eval_steps, reward_config=reward_config,
            detector_model_path=detector_model_path,
        )
        metrics = evaluate_policy(env, csc_policy, attack_threshold)
        metrics["method"] = "CostSensitiveClassifier"
        metrics["controller"] = "CostSensitiveClassifier"
        metrics["algorithm"] = "supervised"
        all_results.append(metrics)
        env.close()

        print(f"  BenSafe: {metrics.get('goodput', 0):.4f} | "
              f"AtkMit: {metrics.get('attack_mitigation_rate', 0):.4f} | "
              f"BenDrop: {metrics.get('benign_drop_rate', 0):.4f} | "
              f"Latency: {metrics.get('avg_latency', 0):.4f} | "
              f"SSU: {metrics.get('ssu', 0):.4f}")
    except Exception as e:
        print(f"  CostSensitiveClassifier failed: {e}")

    # ── Baseline 2: Contextual Bandit ──────────────────────────────────
    print("\n=== Training Contextual Bandit (online) ===")
    try:
        bandit_policy = ContextualBanditPolicy.train_online(
            train_win, state_dim, EdgeTrafficSecurityEnv,
            max_train_steps=20000, attack_threshold=attack_threshold,
            reward_config=reward_config, detector_model_path=detector_model_path,
        )

        env = EdgeTrafficSecurityEnv(
            window_path=test_win, state_dim=state_dim,
            max_steps=max_eval_steps, reward_config=reward_config,
            detector_model_path=detector_model_path,
        )
        metrics = evaluate_policy(env, bandit_policy, attack_threshold)
        metrics["method"] = "ContextualBandit"
        metrics["controller"] = "ContextualBandit"
        metrics["algorithm"] = "bandit"
        all_results.append(metrics)
        env.close()

        print(f"  BenSafe: {metrics.get('goodput', 0):.4f} | "
              f"AtkMit: {metrics.get('attack_mitigation_rate', 0):.4f} | "
              f"BenDrop: {metrics.get('benign_drop_rate', 0):.4f} | "
              f"Latency: {metrics.get('avg_latency', 0):.4f} | "
              f"SSU: {metrics.get('ssu', 0):.4f}")
    except Exception as e:
        print(f"  ContextualBandit failed: {e}")

    # ── Baseline 3: Decision Tree ─────────────────────────────────────
    print("\n=== Training Decision Tree Policy ===")
    try:
        dt_policy = DecisionTreePolicy.train(
            train_win, state_dim, attack_threshold,
            max_depth=5, min_samples_leaf=50,
        )
        # Save model and rules
        dt_path = os.path.join(results_dir, "decision_tree_policy.pkl")
        joblib.dump(dt_policy.model, dt_path)

        # Export rules for paper appendix
        rules = dt_policy.export_rules()
        rules_path = os.path.join(results_dir, "decision_tree_rules.txt")
        with open(rules_path, "w") as f:
            f.write(rules)
        print(f"  Rules saved: {rules_path}")
        print(f"  Model saved: {dt_path}")

        env = EdgeTrafficSecurityEnv(
            window_path=test_win, state_dim=state_dim,
            max_steps=max_eval_steps, reward_config=reward_config,
            detector_model_path=detector_model_path,
        )
        metrics = evaluate_policy(env, dt_policy, attack_threshold)
        metrics["method"] = "DecisionTree"
        metrics["controller"] = "DecisionTree"
        metrics["algorithm"] = "rule_list"
        all_results.append(metrics)
        env.close()

        print(f"  BenSafe: {metrics.get('goodput', 0):.4f} | "
              f"AtkMit: {metrics.get('attack_mitigation_rate', 0):.4f} | "
              f"BenDrop: {metrics.get('benign_drop_rate', 0):.4f} | "
              f"Latency: {metrics.get('avg_latency', 0):.4f} | "
              f"SSU: {metrics.get('ssu', 0):.4f}")
    except Exception as e:
        print(f"  DecisionTree failed: {e}")

    # ── Save results ──────────────────────────────────────────────────
    if all_results:
        df = pd.DataFrame(all_results)
        save_path = os.path.join(results_dir, "new_baselines.csv")
        df.to_csv(save_path, index=False)
        print(f"\nNew baselines saved: {save_path}")

        display_cols = ["method", "f1", "fpr", "recall", "precision",
                        "goodput", "benign_drop_rate", "attack_mitigation_rate",
                        "avg_latency", "ssu"]
        avail_cols = [c for c in display_cols if c in df.columns]
        print(f"\n{'='*110}")
        header = (f"{'Method':<30} {'F1':>8} {'FPR':>8} {'Goodput':>10} "
                  f"{'BenDrop':>10} {'AtkMit':>10} {'Latency':>10} {'SSU':>8}")
        print(header)
        print(f"{'-'*110}")
        for _, row in df.iterrows():
            print(f"{row['method']:<30} "
                  f"{row.get('f1', 0):>8.4f} {row.get('fpr', 0):>8.4f} "
                  f"{row.get('goodput', 0):>10.4f} "
                  f"{row.get('benign_drop_rate', 0):>10.4f} "
                  f"{row.get('attack_mitigation_rate', 0):>10.4f} "
                  f"{row.get('avg_latency', 0):>10.4f} "
                  f"{row.get('ssu', 0):>8.4f}")
    else:
        print("\nNo baseline results generated.")


if __name__ == "__main__":
    main()
