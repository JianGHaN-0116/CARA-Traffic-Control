"""
Edge network traffic security Gymnasium environment.

Two-stage architecture:
  Stage 1: ML detector (XGBoost/LightGBM) provides detector_confidence
  Stage 1.5: Calibration Layer adjusts detector scores for reliability
  Stage 2: DRL agent / CARA-TC chooses traffic control action based on
            calibrated detector output + edge resource state

MDP formulation:
  State: [traffic_window_features, edge_cpu, edge_memory, queue_length,
           link_utilization, packet_loss, detector_estimated_ratio,
           detector_confidence]
  Action: 7 discrete traffic control actions
  Reward: security detection + network cost penalties
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pickle
import os
import joblib
from .reward_function import RewardCalculator
from .calibration_layer import CalibrationLayer


# State feature index mapping (relative to end of traffic features)
# traffic_features (0..traffic_dim-1), then edge state:
#   traffic_dim+0: edge_cpu
#   traffic_dim+1: edge_memory
#   traffic_dim+2: queue_length
#   traffic_dim+3: link_utilization
#   traffic_dim+4: packet_loss
#   traffic_dim+5: detector_estimated_ratio
#   traffic_dim+6: detector_confidence
EDGE_STATE_FEATURES = [
    "edge_cpu", "edge_memory", "queue_length",
    "link_utilization", "packet_loss", "detector_estimated_ratio", "detector_confidence"
]

# Action names for reference
ACTION_NAMES = [
    "Forward", "Inspect", "Mirror", "Throttle", "Reroute", "Drop", "Isolate"
]

# Detection threshold per action (applied to detector_confidence)
# Lower threshold = more sensitive (more likely to flag as attack)
DETECTION_THRESHOLDS = {
    0: 0.50,  # Forward: default threshold
    1: 0.30,  # Inspect: very sensitive (lower threshold -> more detections)
    2: 0.40,  # Mirror: moderately sensitive
    3: 0.50,  # Throttle: default
    4: 0.50,  # Reroute: default
    5: 0.50,  # Drop: default (drop is mitigation, not detection)
    6: 0.50,  # Isolate: default
}

# Base action cost tables: [latency, resource]
ACTION_LATENCY = [0.1, 0.6, 0.4, 0.3, 0.5, 0.2, 0.7]
ACTION_RESOURCE = [0.1, 0.7, 0.5, 0.3, 0.4, 0.2, 0.6]


class EdgeTrafficSecurityEnv(gym.Env):
    """Edge computing network traffic security control environment.

    Integrates a pre-trained ML detector for realistic detection simulation.
    The detector's confidence is included in the state vector so the DRL agent
    can learn to modulate its control actions based on detection certainty.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, window_path, state_dim, max_steps=50000,
                 reward_config=None, allowed_actions=None,
                 detector_model_path=None, remove_state_features=None,
                 resource_scale=1.0, shuffle_on_reset=False,
                 detection_thresholds=None, cost_scale=1.0,
                 use_rho_in_dynamics=True,
                 calibration_config=None):
        """Initialize the edge traffic security environment.

        Args:
            use_rho_in_dynamics: If True (default), ground-truth attack ratio ρ_t
                is used as an exogenous disturbance in latency and next-state
                dynamics. If False, ρ_t is replaced with detector_estimated_ratio
                so that environment dynamics are fully label-free. The flag does
                NOT affect controller-visible state (which never includes ρ_t);
                it only changes hidden simulator dynamics for ablation studies.
            calibration_config: dict passed to CalibrationLayer. Keys:
                method: "platt" | "isotonic" | "temperature" | "none"
                temperature: float (for temperature scaling)
                calibrator_path: str (path to fitted calibrator pickle)
        """
        super().__init__()

        with open(window_path, "rb") as f:
            self.windows = pickle.load(f)

        self.state_dim = state_dim
        self.max_steps = min(max_steps, len(self.windows) - 1)
        self.resource_scale = resource_scale
        self.cost_scale = cost_scale
        self.shuffle_on_reset = shuffle_on_reset
        self.use_rho_in_dynamics = use_rho_in_dynamics
        self.detection_thresholds = dict(DETECTION_THRESHOLDS)
        if detection_thresholds:
            self.detection_thresholds.update(detection_thresholds)

        self.calibration_layer = CalibrationLayer(calibration_config or {})

        # Allowed actions for ablation (default: all 7)
        if allowed_actions is None:
            allowed_actions = list(range(7))
        self.allowed_actions = allowed_actions

        # State features to zero out for ablation
        self.remove_state_features = remove_state_features or []

        self.action_space = spaces.Discrete(len(self.allowed_actions))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(state_dim,), dtype=np.float32,
        )

        self.reward_calc = RewardCalculator(reward_config or {})

        self.current_step = 0
        self.traffic_feature_dim = len(self.windows[0]["state"])

        # Load pre-trained detector model
        self.detector = None
        if detector_model_path and os.path.exists(detector_model_path):
            self.detector = joblib.load(detector_model_path)
            print(f"Loaded detector: {detector_model_path}")

        # Edge state variables (initialized in reset)
        self.edge_cpu = 0.3
        self.edge_memory = 0.3
        self.queue_length = 0.1
        self.link_utilization = 0.3
        self.packet_loss = 0.0

        # Per-episode tracking for metrics
        self._reset_tracking()

    def _reset_tracking(self):
        """Reset per-episode metric trackers."""
        self.tp = 0  # attack traffic, detected as attack
        self.tn = 0  # benign traffic, not flagged
        self.fp = 0  # benign traffic, flagged as attack
        self.fn = 0  # attack traffic, not detected
        self.benign_dropped = 0    # benign traffic with aggressive action
        self.benign_throttled = 0  # benign traffic with moderate action
        self.attack_mitigated = 0  # attack traffic with aggressive action
        self.benign_forwarded = 0  # benign traffic with safe action
        self.total_benign = 0
        self.total_attack = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.edge_cpu = 0.3 * self.resource_scale
        self.edge_memory = 0.3 * self.resource_scale
        self.queue_length = 0.1
        self.link_utilization = 0.3 * self.resource_scale
        self.packet_loss = 0.0
        self._reset_tracking()

        # Shuffle window order each episode to break temporal correlation
        # (data is label-sorted, so without shuffling the agent would learn
        # "early=benign, late=attack" instead of using detector_confidence)
        if self.shuffle_on_reset:
            self.np_random.shuffle(self.windows)

        return self._get_state(), {}

    def step(self, action):
        # Map action index to actual action ID
        real_action = self.allowed_actions[action]

        window = self.windows[self.current_step]
        true_label = window["label"]
        # Ground-truth window composition is retained only as an exogenous
        # simulator disturbance for reward/cost dynamics, not as controller input.
        attack_ratio = window["attack_ratio"]

        # Get detector confidence (real or simulated)
        raw_detector_confidence = self._get_detector_confidence(window)
        detector_confidence = self.calibration_layer.calibrate(raw_detector_confidence)
        detector_estimated_ratio = self._get_detector_estimated_ratio(window)

        # Dynamics disturbance: use ground-truth ρ_t or detector estimate
        # depending on use_rho_in_dynamics flag (for simulator ablation).
        dynamics_ratio = attack_ratio if self.use_rho_in_dynamics else detector_estimated_ratio

        # Determine detection result: detector_confidence + action-based threshold
        detection_result = self._simulate_detection(real_action, detector_confidence)

        latency, resource_cost = self._simulate_cost(real_action, dynamics_ratio)

        reward = self.reward_calc.compute(
            real_action, true_label, detection_result,
            latency, resource_cost, self.packet_loss,
            attack_ratio=attack_ratio,
            detector_confidence=detector_confidence,
        )

        # Track per-step metrics (uses ground-truth attack_ratio for evaluation)
        self._track_step(real_action, true_label, detection_result, attack_ratio)

        self._update_edge_state(real_action, dynamics_ratio)

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            "true_label": true_label,
            "detection_result": detection_result,
            "detector_confidence": detector_confidence,
            "latency": latency,
            "resource_cost": resource_cost,
            "packet_loss": self.packet_loss,
            "edge_cpu": self.edge_cpu,
            "edge_memory": self.edge_memory,
            "queue_length": self.queue_length,
            "link_utilization": self.link_utilization,
            "action": real_action,
            "action_name": ACTION_NAMES[real_action] if real_action < len(ACTION_NAMES) else str(real_action),
            "attack_ratio": attack_ratio,
            "detector_estimated_ratio": detector_estimated_ratio,
        }

        return self._get_state(), reward, terminated, truncated, info

    def _get_state(self):
        traffic_state = self.windows[self.current_step]["state"]
        raw_confidence = self._get_detector_confidence(self.windows[self.current_step])
        detector_confidence = self.calibration_layer.calibrate(raw_confidence)
        detector_estimated_ratio = self._get_detector_estimated_ratio(self.windows[self.current_step])
        edge_state = np.array([
            self.edge_cpu,
            self.edge_memory,
            self.queue_length,
            self.link_utilization,
            self.packet_loss,
            detector_estimated_ratio,
            detector_confidence,
        ], dtype=np.float32)
        state = np.concatenate([traffic_state, edge_state]).astype(np.float32)

        # Zero out removed features for ablation
        if self.remove_state_features:
            traffic_dim = len(traffic_state)
            for feat_name in self.remove_state_features:
                if feat_name in EDGE_STATE_FEATURES:
                    idx = traffic_dim + EDGE_STATE_FEATURES.index(feat_name)
                    if idx < len(state):
                        state[idx] = 0.0

        return state

    def _get_detector_confidence(self, window):
        """Get detector confidence for a window.

        Priority:
        1. Pre-computed per-flow confidence stored in window dict (most accurate)
        2. Real detector model predict_proba on window mean features
        3. Simulated confidence fallback
        """
        # Use pre-computed per-flow detector confidence if available
        if "detector_confidence" in window:
            return window["detector_confidence"]

        # Use real detector model on window mean features
        if self.detector is not None:
            features = window["state"].reshape(1, -1)
            confidence = float(self.detector.predict_proba(features)[0, 1])
            return confidence

        # Fallback: simulate confidence based on attack_ratio
        attack_ratio = window["attack_ratio"]
        noise = self.np_random.normal(0, 0.05) if hasattr(self, 'np_random') else 0.0
        return float(np.clip(0.3 + 0.5 * attack_ratio + noise, 0.0, 1.0))

    def _get_detector_estimated_ratio(self, window):
        """Get a detector-derived attack-ratio estimate for controller state.

        Priority:
        1. Pre-computed per-flow detector-positive ratio stored in the window
        2. Soft fallback to mean detector confidence
        3. Final fallback derived from the simulated confidence path
        """
        if "detector_estimated_ratio" in window:
            return float(window["detector_estimated_ratio"])

        if "detector_confidence" in window:
            return float(np.clip(window["detector_confidence"], 0.0, 1.0))

        return float(np.clip(self._get_detector_confidence(window), 0.0, 1.0))

    def _simulate_detection(self, action, detector_confidence):
        """Determine if traffic is flagged as attack.

        Uses detector_confidence with action-based threshold modulation:
        - Forward: default threshold (0.5)
        - Inspect: lower threshold (0.3) -> higher sensitivity
        - Other actions: default threshold
        """
        threshold = self.detection_thresholds.get(action, 0.5)
        return 1 if detector_confidence >= threshold else 0

    def _track_step(self, action, true_label, detection_result, attack_ratio):
        """Track metrics for computing benign_drop_rate, attack_mitigation_rate, etc."""
        from .reward_function import RewardCalculator
        mitigation = RewardCalculator.ACTION_MITIGATION.get(action, 0)
        is_attack = (attack_ratio > self.reward_calc.attack_threshold)

        if is_attack:
            self.total_attack += 1
            if detection_result == 1:
                self.tp += 1
            else:
                self.fn += 1
            if mitigation == 2:
                self.attack_mitigated += 1
        else:
            self.total_benign += 1
            if detection_result == 1:
                self.fp += 1
            else:
                self.tn += 1
            if mitigation == 2:
                self.benign_dropped += 1
            elif mitigation == 1:
                self.benign_throttled += 1
            else:
                self.benign_forwarded += 1

    def get_episode_metrics(self):
        """Return episode-level metrics for evaluation."""
        total = self.total_benign + self.total_attack
        return {
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "total_benign": self.total_benign,
            "total_attack": self.total_attack,
            "fpr": self.fp / max(self.total_benign, 1),
            "fnr": self.fn / max(self.total_attack, 1),
            "recall": self.tp / max(self.total_attack, 1),
            "precision": self.tp / max(self.tp + self.fp, 1),
            "benign_drop_rate": self.benign_dropped / max(self.total_benign, 1),
            "benign_throttle_rate": self.benign_throttled / max(self.total_benign, 1),
            "attack_mitigation_rate": self.attack_mitigated / max(self.total_attack, 1),
            "goodput": self.benign_forwarded / max(self.total_benign, 1),
            "accuracy": (self.tp + self.tn) / max(total, 1),
        }

    def _simulate_cost(self, action, attack_ratio):
        """Simulate latency and resource cost for the given action.

        Resource-aware:
        - Low CPU makes Inspect slower and more expensive
        - Low bandwidth makes Mirror/Reroute slower and risk packet loss
        - High queue causes nonlinear latency growth
        - Resource_scale affects recovery rate, creating cascading pressure
        """
        stress_multiplier = 1.0 + 3.0 * max(0.0, 1.0 - self.resource_scale)

        # CPU penalty for heavy actions (Inspect, Reroute)
        cpu_penalty = 0.0
        if action in (1, 4):  # Inspect, Reroute
            cpu_penalty = 0.3 * self.edge_cpu * stress_multiplier

        # Bandwidth penalty for network-heavy actions (Mirror, Reroute)
        bw_penalty = 0.0
        if action in (2, 4):  # Mirror, Reroute
            bw_penalty = 0.2 * self.link_utilization * stress_multiplier

        # Queue congestion: nonlinear latency growth
        queue_factor = (self.queue_length ** 1.5) * stress_multiplier

        latency = (1.0 + queue_factor + self.link_utilization
                   + (ACTION_LATENCY[action] * self.cost_scale) + attack_ratio
                   + cpu_penalty + bw_penalty)

        resource_cost = (self.edge_cpu + self.edge_memory
                         + (ACTION_RESOURCE[action] * self.cost_scale)
                         + cpu_penalty * 0.5 + bw_penalty * 0.25)

        return latency, resource_cost

    def _update_edge_state(self, action, attack_ratio):
        """Update simulated edge node state based on action taken.

        Resource-aware dynamics:
        - Recovery rate scales with resource_scale (lower scale = slower recovery)
        - Inspect is expensive under low CPU
        - Mirror/Reroute consume bandwidth
        - Throttle reduces link utilization meaningfully
        - Reroute trades queue for bandwidth
        - Queue overload causes additional packet loss
        """
        stress_multiplier = 1.0 + 3.0 * max(0.0, 1.0 - self.resource_scale)

        # Resource-aware recovery: lower resource_scale = slower recovery
        cpu_recovery = max(0.85, 0.995 - 0.10 * (1.0 - self.resource_scale))
        mem_recovery = max(0.87, 0.995 - 0.08 * (1.0 - self.resource_scale))
        queue_recovery = max(0.90, 0.998 - 0.08 * (1.0 - self.resource_scale))
        bw_recovery = max(0.90, 0.998 - 0.08 * (1.0 - self.resource_scale))

        # Attack traffic increases load
        self.queue_length += 0.05 * attack_ratio * stress_multiplier
        self.link_utilization += 0.05 * attack_ratio * stress_multiplier

        # Action effects on state (resource-aware)
        if action == 1:  # Inspect
            # CPU-intensive: cost scales inversely with available CPU
            cpu_load = 0.05 * stress_multiplier
            self.edge_cpu += cpu_load
            self.edge_memory += 0.03 * stress_multiplier
        elif action == 2:  # Mirror
            # Bandwidth-intensive
            bw_load = 0.04 * (1.0 + 0.5 * (1.0 - self.resource_scale))
            self.link_utilization += bw_load
            self.edge_cpu += 0.02 * stress_multiplier
        elif action == 3:  # Throttle
            # Meaningful: reduces link utilization by limiting traffic
            throttle_reduction = (0.05 + 0.02 * self.link_utilization) / stress_multiplier
            self.link_utilization -= throttle_reduction
            self.edge_cpu += 0.01 * stress_multiplier  # slight CPU cost for rate limiting
        elif action == 4:  # Reroute
            # Trades queue buildup for bandwidth consumption
            queue_reduction = (0.03 + 0.02 * self.queue_length) / stress_multiplier
            self.queue_length -= queue_reduction
            bw_cost = 0.03 * stress_multiplier
            self.link_utilization += bw_cost
            self.edge_cpu += 0.02 * stress_multiplier
        elif action == 5:  # Drop
            self.queue_length -= 0.04 / stress_multiplier
        elif action == 6:  # Isolate
            self.queue_length -= 0.05 / stress_multiplier
            self.link_utilization -= 0.05 / stress_multiplier

        # Natural recovery (resource-aware)
        self.edge_cpu *= cpu_recovery
        self.edge_memory *= mem_recovery
        self.queue_length *= queue_recovery
        self.link_utilization *= bw_recovery

        # Clip to valid range
        self.edge_cpu = float(np.clip(self.edge_cpu, 0.0, 1.0))
        self.edge_memory = float(np.clip(self.edge_memory, 0.0, 1.0))
        self.queue_length = float(np.clip(self.queue_length, 0.0, 1.0))
        self.link_utilization = float(np.clip(self.link_utilization, 0.0, 1.0))

        # Packet loss: congestion-driven (nonlinear at high queue)
        congestion_threshold = max(0.30, 0.7 * self.resource_scale)
        queue_threshold = max(0.40, 0.8 * self.resource_scale)
        congestion = max(0.0, self.link_utilization - congestion_threshold)
        overload = max(0.0, self.queue_length - queue_threshold) ** 1.5
        self.packet_loss = float(np.clip(congestion * 0.5 + overload, 0.0, 1.0))
