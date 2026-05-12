import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import load_dataset


def analyze_dataset(dataset_path: str) -> Dict[str, Any]:
    """Analyze dataset format and quality."""
    if not os.path.exists(dataset_path):
        return {"status": "error", "message": f"Dataset not found: {dataset_path}"}

    # Detect format
    format_type = "unknown"
    if dataset_path.endswith(".jsonl"):
        format_type = "jsonl"
    elif dataset_path.endswith(".json"):
        format_type = "json"
    elif dataset_path.endswith(".csv"):
        format_type = "csv"
    elif dataset_path.endswith(".parquet"):
        format_type = "parquet"

    # Load and analyze
    try:
        if format_type == "jsonl":
            with open(dataset_path, "r", encoding="utf-8") as f:
                first_line = json.loads(f.readline())
            # Detect sub-format
            if "instruction" in first_line and "output" in first_line:
                detected_format = "alpaca"
            elif "messages" in first_line:
                detected_format = "sharegpt"
            elif "conversations" in first_line:
                detected_format = "sharegpt"
            elif "text" in first_line:
                detected_format = "raw"
            else:
                detected_format = "unknown"

            # Count samples
            with open(dataset_path, "r", encoding="utf-8") as f:
                num_samples = sum(1 for _ in f)

        elif format_type == "csv":
            df = pd.read_csv(dataset_path)
            first_line = df.iloc[0].to_dict()
            num_samples = len(df)
            detected_format = "alpaca" if "instruction" in df.columns else "raw" if "text" in df.columns else "unknown"

        else:
            detected_format = format_type
            num_samples = 0
            first_line = {}

        # Analyze columns
        columns = list(first_line.keys()) if isinstance(first_line, dict) else []

        return {
            "status": "success",
            "format": detected_format,
            "file_format": format_type,
            "num_samples": num_samples,
            "columns": columns,
            "sample": first_line,
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


def validate_config(config_path: Optional[str] = None,
                    model_name: Optional[str] = None,
                    dataset_info: Optional[Dict] = None) -> Dict[str, Any]:
    """Validate or generate LoRA config."""
    import json

    # Load existing config if provided
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        # Load default Qwen config
        default_config_path = "configs/qwen2.5_3b_lora.json"
        if model_name and "7B" in model_name:
            default_config_path = "configs/qwen2.5_7b_lora.json"

        if os.path.exists(default_config_path):
            with open(default_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {}

    # Override with provided values
    if model_name:
        config["model_name"] = model_name

    if dataset_info:
        data_config = config.get("data_config", {})
        if "dataset_path" in dataset_info:
            data_config["train_file"] = dataset_info["dataset_path"]
        if "format" in dataset_info:
            data_config["format"] = dataset_info["format"]
        config["data_config"] = data_config

    # Validate required fields
    required = ["model_name", "lora_config", "training_config"]
    missing = [r for r in required if r not in config]

    if missing:
        return {"status": "error", "message": f"Missing config fields: {missing}", "config": config}

    return {"status": "success", "config": config}
