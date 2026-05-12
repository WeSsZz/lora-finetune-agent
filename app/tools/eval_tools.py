import json
import os
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate_model(model_path: str, eval_dataset_path: Optional[str],
                   config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate fine-tuned model."""
    try:
        model_name = config.get("model_name", "Qwen/Qwen2.5-3B-Instruct")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load base model with PEFT adapter
        quantization_config = config.get("quantization_config", {})
        load_in_4bit = quantization_config.get("load_in_4bit", True)

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

        # Load adapter
        adapter_path = os.path.join(model_path, "final_adapter")
        if not os.path.exists(adapter_path):
            adapter_path = model_path

        if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            model = PeftModel.from_pretrained(model, adapter_path)

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load eval dataset
        if eval_dataset_path and os.path.exists(eval_dataset_path):
            eval_samples = []
            with open(eval_dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        eval_samples.append(json.loads(line))
        else:
            return {"status": "error", "message": "Eval dataset not found"}

        # Calculate perplexity on a subset
        max_eval_samples = min(100, len(eval_samples))
        total_loss = 0.0
        total_tokens = 0

        model.eval()
        with torch.no_grad():
            for sample in eval_samples[:max_eval_samples]:
                # Format text based on format type
                text = _format_sample(sample, config)
                if not text:
                    continue

                inputs = tokenizer(text, return_tensors="pt", truncation=True,
                                   max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

                num_tokens = inputs["input_ids"].shape[1]
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        # Clean up
        del model
        torch.cuda.empty_cache()

        return {
            "status": "success",
            "perplexity": round(perplexity, 4),
            "eval_loss": round(avg_loss, 4),
            "eval_samples": max_eval_samples,
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


def _format_sample(sample: Dict, config: Dict) -> str:
    """Format a sample into text for evaluation."""
    data_config = config.get("data_config", {})
    fmt = data_config.get("format", "alpaca")

    if fmt == "alpaca":
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output = sample.get("output", "")
        if input_text:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    elif fmt == "raw":
        return sample.get("text", "")
    else:
        return str(sample)
