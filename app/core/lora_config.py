"""LoRA configuration builder for different models."""

import json
from typing import Any, Dict, List, Optional


class LoRAConfigBuilder:
    """Build LoRA configs for various models."""

    # Target modules for different model families
    TARGET_MODULES = {
        "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
        "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }

    @classmethod
    def detect_model_family(cls, model_name: str) -> str:
        """Detect model family from model name."""
        name_lower = model_name.lower()
        for family in cls.TARGET_MODULES:
            if family in name_lower:
                return family
        return "qwen"  # Default fallback

    @classmethod
    def get_target_modules(cls, model_name: str) -> List[str]:
        """Get target modules for a model."""
        family = cls.detect_model_family(model_name)
        return cls.TARGET_MODULES.get(family, cls.TARGET_MODULES["qwen"])

    @classmethod
    def build_config(
        cls,
        model_name: str,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        use_rslora: bool = False,
        output_dir: str = "./outputs/lora",
        **kwargs
    ) -> Dict[str, Any]:
        """Build a complete LoRA configuration."""
        target_modules = cls.get_target_modules(model_name)

        # Adjust batch size and accumulation based on model size
        model_size = cls._estimate_model_size(model_name)
        if model_size <= 3:
            per_device_batch = 2
            grad_accum = 4
        elif model_size <= 7:
            per_device_batch = 1
            grad_accum = 8
        else:
            per_device_batch = 1
            grad_accum = 16

        config = {
            "model_name": model_name,
            "lora_config": {
                "r": r,
                "lora_alpha": lora_alpha,
                "target_modules": target_modules,
                "lora_dropout": lora_dropout,
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "use_rslora": use_rslora,
            },
            "training_config": {
                "output_dir": output_dir,
                "num_train_epochs": kwargs.get("num_train_epochs", 3),
                "per_device_train_batch_size": per_device_batch,
                "per_device_eval_batch_size": per_device_batch,
                "gradient_accumulation_steps": grad_accum,
                "learning_rate": kwargs.get("learning_rate", 2e-4),
                "max_grad_norm": 0.3,
                "warmup_ratio": 0.03,
                "lr_scheduler_type": "cosine",
                "logging_steps": 10,
                "save_strategy": "epoch",
                "evaluation_strategy": "epoch",
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "bf16": False,
                "fp16": True,
                "optim": "paged_adamw_8bit",
                "group_by_length": True,
                "max_seq_length": kwargs.get("max_seq_length", 2048),
                "packing": False,
                "max_steps": -1,
                "report_to": "none",
            },
            "quantization_config": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
            },
            "data_config": {
                "format": kwargs.get("data_format", "alpaca"),
                "max_seq_length": kwargs.get("max_seq_length", 2048),
            },
        }
        return config

    @classmethod
    def _estimate_model_size(cls, model_name: str) -> int:
        """Estimate model size in billions from name."""
        import re
        match = re.search(r'(\d+\.?\d*)[bB]', model_name)
        if match:
            return int(float(match.group(1)))
        return 3  # Default

    @classmethod
    def save_config(cls, config: Dict[str, Any], path: str):
        """Save config to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
