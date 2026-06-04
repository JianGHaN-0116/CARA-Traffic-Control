"""
Logging utility for experiment tracking.
"""
import os
import time
import json
import yaml


class ExperimentLogger:
    """Simple experiment logger that saves results to JSON and YAML."""

    def __init__(self, log_dir, experiment_name):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        os.makedirs(log_dir, exist_ok=True)
        self.start_time = time.time()
        self.log_entries = []

    def log(self, message, data=None):
        entry = {
            "timestamp": time.time(),
            "elapsed": time.time() - self.start_time,
            "message": message,
        }
        if data:
            entry.update(data)
        self.log_entries.append(entry)
        print(f"[{entry['elapsed']:.1f}s] {message}")

    def save_results(self, results, filename="results.json"):
        path = os.path.join(self.log_dir, filename)
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        self.log(f"Results saved: {path}")

    def save_config(self, config, filename="config.yaml"):
        path = os.path.join(self.log_dir, filename)
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    def save_log(self, filename="experiment_log.json"):
        path = os.path.join(self.log_dir, filename)
        with open(path, "w") as f:
            json.dump(self.log_entries, f, indent=2, default=str)
