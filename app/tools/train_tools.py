import json
import os
import subprocess
import sys
from typing import Any, Dict, Optional

from app.core.trainer import LoRATrainer


# Global trainer instance tracker for monitoring
_active_trainers: Dict[str, LoRATrainer] = {}


def preprocess_data(dataset_path: str, output_path: str,
                    config: Dict[str, Any], format: str = "alpaca") -> Dict[str, Any]:
    """Preprocess dataset for training."""
    os.makedirs(output_path, exist_ok=True)

    try:
        import json
        import random

        # Load dataset
        samples = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        # Shuffle and split
        random.seed(42)
        random.shuffle(samples)
        split_idx = int(len(samples) * 0.9)
        train_samples = samples[:split_idx]
        eval_samples = samples[split_idx:]

        # Save splits
        train_path = os.path.join(output_path, "train.jsonl")
        eval_path = os.path.join(output_path, "eval.jsonl")

        with open(train_path, "w", encoding="utf-8") as f:
            for s in train_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        with open(eval_path, "w", encoding="utf-8") as f:
            for s in eval_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        return {
            "status": "success",
            "train_path": train_path,
            "eval_path": eval_path,
            "train_samples": len(train_samples),
            "eval_samples": len(eval_samples),
            "format": format,
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


def start_training(config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """Start LoRA training."""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Save config for reference
        config_path = os.path.join(output_dir, "training_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # Initialize trainer
        trainer = LoRATrainer(config)
        trainer.output_dir = output_dir

        # Start training in background thread
        import threading
        _active_trainers[output_dir] = trainer

        def train_worker():
            try:
                trainer.train()
            except Exception as e:
                trainer.training_error = str(e)
                trainer.training_finished = True

        thread = threading.Thread(target=train_worker)
        thread.daemon = True
        thread.start()

        return {
            "status": "success",
            "output_dir": output_dir,
            "message": "Training started",
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


def monitor_training(output_dir: str) -> Dict[str, Any]:
    """Monitor training progress."""
    trainer = _active_trainers.get(output_dir)

    if trainer:
        return trainer.get_status()

    # Fallback: check if training completed by looking at output files
    checkpoint_dir = os.path.join(output_dir, "checkpoint")
    if os.path.exists(checkpoint_dir):
        # Check for completed marker
        completed_file = os.path.join(output_dir, "training_completed.txt")
        if os.path.exists(completed_file):
            return {"status": "completed", "message": "Training completed"}

    # Check log file for progress
    log_file = os.path.join(output_dir, "training_log.jsonl")
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()
            if lines:
                last_log = json.loads(lines[-1])
                return {
                    "status": "running",
                    "current_epoch": last_log.get("epoch", 0),
                    "current_loss": last_log.get("loss", None),
                    "learning_rate": last_log.get("learning_rate", None),
                    "progress": last_log.get("progress", 0),
                }
        except:
            pass

    return {"status": "unknown", "message": "No active training found"}
