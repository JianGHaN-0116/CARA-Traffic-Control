"""
Reward function for the edge traffic security environment.

Two-stage design:
  - Stage 1: ML detector provides attack_probability
  - Stage 2: DRL agent chooses traffic control action

Reward design philosophy:
  - Strongly penalize dropping/isolating benign traffic
  - Reward correctly mitigating attacks
  - Penalize missing attacks
  - Small cost for latency/resources
"""
import numpy as np


class RewardCalculator:
    """Reward calculator for detector-assisted DRL traffic flow control.

    Actions are classified into mitigation levels:
      - Safe (Forward, Inspect, Mirror): no traffic disruption
      - Moderate (Throttle, Reroute): minor traffic disruption
      - Aggressive (Drop, Isolate): major traffic disruption

    Reward depends on: (is_attack, action_mitigation_level)
    """

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

    def __init__(self, reward_config):
        # Core rewards/penalties
        self.attack_mitigated_reward = reward_config.get("attack_mitigated", 5.0)
        self.attack_detected_reward = reward_config.get("attack_detected", 2.0)
        self.benign_forwarded_reward = reward_config.get("benign_forwarded", 1.0)
        self.benign_dropped_penalty = reward_config.get("benign_dropped", 10.0)
        self.benign_throttled_penalty = reward_config.get("benign_throttled", 5.0)
        self.attack_missed_penalty = reward_config.get("attack_missed", 8.0)

        # Cost weights
        self.latency_weight = reward_config.get("latency", 0.5)
        self.resource_weight = reward_config.get("resource_cost", 0.3)
        self.loss_weight = reward_config.get("packet_loss", 1.0)

        # Detection threshold
        self.attack_threshold = reward_config.get("attack_threshold", 0.5)

    def compute(self, action, true_label, detection_result,
                latency, resource_cost, packet_loss,
                attack_ratio=None, detector_confidence=None):
        reward = 0.0

        # Determine ground truth: is this window an attack?
        is_attack = (attack_ratio > self.attack_threshold) if attack_ratio is not None else (true_label == 1)

        # Determine mitigation level of the chosen action
        mitigation = self.ACTION_MITIGATION.get(action, 0)

        if is_attack:
            if mitigation == 2:
                # Attack correctly mitigated (Drop/Isolate on attack traffic)
                reward += self.attack_mitigated_reward
            elif detection_result == 1:
                # Attack detected but only forwarded/inspected/throttled
                reward += self.attack_detected_reward
            else:
                # Attack missed entirely
                reward -= self.attack_missed_penalty
        else:
            if mitigation == 2:
                # Benign traffic dropped/isolated — worst case
                reward -= self.benign_dropped_penalty
            elif mitigation == 1:
                # Benign traffic throttled/rerouted
                reward -= self.benign_throttled_penalty
            else:
                # Benign traffic correctly forwarded
                reward += self.benign_forwarded_reward

        # Network cost penalties (small, secondary)
        reward -= self.latency_weight * latency
        reward -= self.resource_weight * resource_cost
        reward -= self.loss_weight * packet_loss

        return reward
