"""Training monitoring utilities."""

import json
import os
import time
from typing import Any, Dict, List, Optional


class TrainingMonitor:
    """Monitor training progress from log files."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, "training_log.jsonl")

    def get_latest_status(self) -> Dict[str, Any]:
        """Get the latest training status from logs."""
        if not os.path.exists(self.log_file):
            # Check for completion marker
            completed_file = os.path.join(self.output_dir, "training_completed.txt")
            if os.path.exists(completed_file):
                return {"status": "completed", "message": "Training completed"}
            return {"status": "unknown", "message": "No log file found"}

        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if not lines:
                return {"status": "starting", "message": "Training starting"}

            last_line = json.loads(lines[-1])

            # Check if completed
            completed_file = os.path.join(self.output_dir, "training_completed.txt")
            if os.path.exists(completed_file):
                last_line["status"] = "completed"
            else:
                last_line["status"] = "running"

            return last_line

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_loss_history(self) -> List[Dict[str, Any]]:
        """Get full loss history for plotting."""
        if not os.path.exists(self.log_file):
            return []

        history = []
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if entry.get("loss") is not None:
                            history.append({
                                "step": entry.get("step", 0),
                                "epoch": entry.get("epoch", 0),
                                "loss": entry["loss"],
                                "learning_rate": entry.get("learning_rate"),
                            })
        except:
            pass

        return history

    def wait_for_completion(self, timeout: Optional[float] = None,
                            poll_interval: float = 5.0) -> Dict[str, Any]:
        """Poll until training completes or timeout."""
        start_time = time.time()

        while True:
            status = self.get_latest_status()

            if status.get("status") in ("completed", "failed", "error"):
                return status

            if timeout and (time.time() - start_time) > timeout:
                return {"status": "timeout", "message": f"Timed out after {timeout}s"}

            time.sleep(poll_interval)
