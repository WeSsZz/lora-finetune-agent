"""LoRA training engine using PEFT and TRL."""

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
)
from trl import SFTTrainer


class TrainingMonitorCallback(TrainerCallback):
    """Custom callback to monitor training progress."""

    def __init__(self, log_file: str):
        self.log_file = log_file
        self.current_epoch = 0
        self.current_loss = None
        self.learning_rate = None
        self.progress = 0.0
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.current_epoch = state.epoch or 0
            self.current_loss = logs.get("loss", logs.get("train_loss", None))
            self.learning_rate = logs.get("learning_rate", None)
            self.progress = state.global_step / state.max_steps if state.max_steps > 0 else 0

            # Write to log file
            with open(self.log_file, "a", encoding="utf-8") as f:
                log_entry = {
                    "epoch": self.current_epoch,
                    "step": state.global_step,
                    "loss": self.current_loss,
                    "learning_rate": self.learning_rate,
                    "progress": self.progress,
                    "timestamp": time.time(),
                }
                f.write(json.dumps(log_entry) + "\n")

    def on_epoch_end(self, args, state, control, **kwargs):
        self.current_epoch = state.epoch or 0


class LoRATrainer:
    """LoRA fine-tuning trainer."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config["model_name"]
        self.lora_config_dict = config["lora_config"]
        self.training_config = config["training_config"]
        self.quantization_config = config.get("quantization_config", {})
        self.data_config = config.get("data_config", {})

        self.output_dir = self.training_config.get("output_dir", "./outputs")
        self.model = None
        self.tokenizer = None
        self.trainer = None

        # Training state
        self.training_started = False
        self.training_finished = False
        self.training_error = None
        self.monitor_callback = None

    def _load_model_and_tokenizer(self):
        """Load base model and tokenizer with quantization."""
        print(f"Loading model: {self.model_name}")

        # Quantization config
        bnb_config = None
        if self.quantization_config.get("load_in_4bit", True):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.quantization_config.get("bnb_4bit_compute_dtype", "float16")),
                bnb_4bit_use_double_quant=self.quantization_config.get("bnb_4bit_use_double_quant", True),
                bnb_4bit_quant_type=self.quantization_config.get("bnb_4bit_quant_type", "nf4"),
            )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Prepare model for k-bit training
        if bnb_config:
            self.model = prepare_model_for_kbit_training(self.model)

        # Apply LoRA
        lora_config = LoraConfig(
            r=self.lora_config_dict["r"],
            lora_alpha=self.lora_config_dict["lora_alpha"],
            target_modules=self.lora_config_dict["target_modules"],
            lora_dropout=self.lora_config_dict["lora_dropout"],
            bias=self.lora_config_dict.get("bias", "none"),
            task_type=self.lora_config_dict.get("task_type", "CAUSAL_LM"),
            use_rslora=self.lora_config_dict.get("use_rslora", False),
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def _load_dataset(self) -> tuple:
        """Load and preprocess training and eval datasets."""
        train_file = self.data_config.get("train_file")
        eval_file = self.data_config.get("eval_file")
        data_format = self.data_config.get("format", "alpaca")

        if not train_file or not os.path.exists(train_file):
            raise ValueError(f"Training file not found: {train_file}")

        train_dataset = self._load_and_format(train_file, data_format)
        eval_dataset = None
        if eval_file and os.path.exists(eval_file):
            eval_dataset = self._load_and_format(eval_file, data_format)

        return train_dataset, eval_dataset

    def _load_and_format(self, file_path: str, fmt: str) -> Dataset:
        """Load dataset file and format for training."""
        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        # Format samples based on format type
        formatted = []
        for sample in samples:
            text = self._format_sample(sample, fmt)
            if text:
                formatted.append({"text": text})

        return Dataset.from_list(formatted)

    def _format_sample(self, sample: Dict, fmt: str) -> str:
        """Format a single sample based on format."""
        if fmt == "alpaca":
            instruction = sample.get("instruction", "")
            input_text = sample.get("input", "")
            output = sample.get("output", "")

            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            return prompt

        elif fmt == "sharegpt":
            messages = sample.get("messages", sample.get("conversations", []))
            if not messages:
                return ""
            # Use chat template if available
            if hasattr(self.tokenizer, "apply_chat_template"):
                try:
                    return self.tokenizer.apply_chat_template(messages, tokenize=False)
                except:
                    pass
            # Fallback
            texts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                texts.append(f"{role}: {content}")
            return "\n".join(texts)

        elif fmt == "raw":
            return sample.get("text", "")

        return str(sample)

    def train(self):
        """Execute training."""
        try:
            self.training_started = True
            self._load_model_and_tokenizer()
            train_dataset, eval_dataset = self._load_dataset()

            # Setup training arguments
            training_args_dict = self.training_config.copy()
            training_args_dict["output_dir"] = self.output_dir

            # Remove keys not in TrainingArguments
            valid_keys = TrainingArguments.__dataclass_fields__.keys()
            filtered_args = {k: v for k, v in training_args_dict.items() if k in valid_keys}

            training_args = TrainingArguments(**filtered_args)

            # Setup monitor callback
            log_file = os.path.join(self.output_dir, "training_log.jsonl")
            self.monitor_callback = TrainingMonitorCallback(log_file)

            # Create SFT trainer
            max_seq_length = self.data_config.get("max_seq_length", 2048)

            self.trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=training_args,
                max_seq_length=max_seq_length,
                dataset_text_field="text",
                callbacks=[self.monitor_callback],
            )

            # Train
            print("Starting training...")
            self.trainer.train()

            # Save final model
            self.trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)

            # Mark as completed
            with open(os.path.join(self.output_dir, "training_completed.txt"), "w") as f:
                f.write("Training completed successfully\n")

            self.training_finished = True
            print(f"Training completed. Model saved to {self.output_dir}")

        except Exception as e:
            self.training_error = str(e)
            self.training_finished = True
            raise

    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        if self.training_error:
            return {
                "status": "failed",
                "message": self.training_error,
            }

        if not self.training_started:
            return {
                "status": "pending",
                "message": "Training not started",
            }

        if self.training_finished:
            return {
                "status": "completed",
                "message": "Training completed",
            }

        if self.monitor_callback:
            return {
                "status": "running",
                "current_epoch": self.monitor_callback.current_epoch,
                "current_loss": self.monitor_callback.current_loss,
                "learning_rate": self.monitor_callback.learning_rate,
                "progress": self.monitor_callback.progress,
            }

        return {
            "status": "running",
            "message": "Training in progress",
        }
