"""
Calibration Layer for detector-assisted traffic control.

This module sits between the flow-level detector and the window-level
controller, transforming raw detector confidence scores into calibrated
probabilities that account for score drift, artifact sensitivity, and
distribution shift.

Supported calibration methods:
  - "none": pass-through (no calibration)
  - "temperature": temperature scaling (single scalar parameter)
  - "platt": Platt scaling (logistic regression on validation scores)
  - "isotonic": isotonic regression (non-parametric)

Usage:
    layer = CalibrationLayer({"method": "temperature", "temperature": 1.5})
    calibrated = layer.calibrate(raw_confidence)

    # Fit on validation data:
    layer.fit(val_scores, val_labels)
"""
import os
import numpy as np
import pickle


class CalibrationLayer:
    """Calibration layer that adjusts raw detector confidence scores.

    This is a key component of the CARA-TC framework: it ensures that
    detector scores are well-calibrated before being used for control
    decisions, reducing the risk of over-confident or under-confident
    actions that could disrupt benign services or miss attacks.
    """

    def __init__(self, config):
        self.method = config.get("method", "none")
        self.temperature = float(config.get("temperature", 1.0))
        self.calibrator_path = config.get("calibrator_path", None)
        self.calibrator = None

        if self.calibrator_path and os.path.exists(self.calibrator_path):
            with open(self.calibrator_path, "rb") as f:
                self.calibrator = pickle.load(f)

    def calibrate(self, raw_confidence):
        """Apply calibration to a raw detector confidence score.

        Args:
            raw_confidence: float in [0, 1], raw detector output

        Returns:
            float: calibrated confidence in [0, 1]
        """
        if self.method == "none":
            return float(raw_confidence)

        raw = float(np.clip(raw_confidence, 1e-7, 1.0 - 1e-7))

        if self.method == "temperature":
            logit = np.log(raw / (1.0 - raw))
            scaled_logit = logit / max(self.temperature, 1e-7)
            return float(np.clip(1.0 / (1.0 + np.exp(-scaled_logit)), 0.0, 1.0))

        if self.method == "platt":
            if self.calibrator is not None:
                return float(np.clip(self.calibrator.predict_proba([[raw]])[0, 1], 0.0, 1.0))
            return float(raw)

        if self.method == "isotonic":
            if self.calibrator is not None:
                return float(np.clip(self.calibrator.transform([raw])[0], 0.0, 1.0))
            return float(raw)

        return float(raw)

    def calibrate_batch(self, raw_confidences):
        """Apply calibration to an array of raw confidence scores.

        Args:
            raw_confidences: array-like of floats in [0, 1]

        Returns:
            np.ndarray: calibrated confidences
        """
        return np.array([self.calibrate(c) for c in raw_confidences])

    def fit(self, val_scores, val_labels):
        """Fit the calibrator on validation data.

        Args:
            val_scores: array-like of raw detector confidence scores
            val_labels: array-like of ground-truth binary labels
        """
        scores = np.asarray(val_scores, dtype=np.float64)
        labels = np.asarray(val_labels, dtype=np.int32)

        if self.method == "platt":
            from sklearn.linear_model import LogisticRegression
            self.calibrator = LogisticRegression(C=1e10, solver="lbfgs")
            self.calibrator.fit(scores.reshape(-1, 1), labels)

        elif self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(scores, labels.astype(np.float64))

        elif self.method == "temperature":
            self.temperature = self._optimize_temperature(scores, labels)

    def _optimize_temperature(self, scores, labels, n_iter=100, lr=0.01):
        """Optimize temperature parameter via negative log-likelihood minimization.

        Args:
            scores: raw detector confidence scores
            labels: ground-truth binary labels
            n_iter: number of optimization iterations
            lr: learning rate

        Returns:
            float: optimal temperature
        """
        eps = 1e-7
        scores = np.clip(scores, eps, 1.0 - eps)
        logits = np.log(scores / (1.0 - scores))
        labels = labels.astype(np.float64)
        temperature = 1.0

        for _ in range(n_iter):
            scaled_logits = logits / max(temperature, eps)
            probs = 1.0 / (1.0 + np.exp(-scaled_logits))
            probs = np.clip(probs, eps, 1.0 - eps)
            nll = -np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))

            grad = np.mean(
                (probs - labels) * logits / (max(temperature, eps) ** 2)
            )
            temperature = max(0.01, temperature - lr * grad)

        return float(temperature)

    def save(self, path):
        """Save the fitted calibrator to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.calibrator, f)

    def inject_shift(self, raw_confidence, shift_type="overconfident", magnitude=0.1):
        """Simulate calibration drift for robustness experiments.

        This method deliberately distorts raw confidence scores to test
        how CARA-TC and other controllers behave under miscalibration.

        Args:
            raw_confidence: float in [0, 1]
            shift_type: "overconfident" | "underconfident" | "random"
            magnitude: strength of the shift

        Returns:
            float: shifted confidence
        """
        raw = float(raw_confidence)

        if shift_type == "overconfident":
            return float(np.clip(raw + magnitude * (1.0 - raw), 0.0, 1.0))
        elif shift_type == "underconfident":
            return float(np.clip(raw - magnitude * raw, 0.0, 1.0))
        elif shift_type == "random":
            noise = np.random.normal(0, magnitude)
            return float(np.clip(raw + noise, 0.0, 1.0))

        return raw
