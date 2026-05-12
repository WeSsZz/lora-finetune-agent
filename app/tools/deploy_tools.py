import json
import os
import shutil
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def export_adapter(model_path: str, output_path: str,
                   merge_base: bool = False) -> Dict[str, Any]:
    """Export LoRA adapter, optionally merged with base model."""
    try:
        os.makedirs(output_path, exist_ok=True)

        # Load config to get base model name
        config_path = os.path.join(model_path, "training_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            model_name = config.get("model_name", "Qwen/Qwen2.5-3B-Instruct")
        else:
            model_name = "Qwen/Qwen2.5-3B-Instruct"

        # Check if adapter exists
        adapter_path = model_path
        if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            # Try to find latest checkpoint
            checkpoints = [d for d in os.listdir(model_path)
                           if d.startswith("checkpoint")]
            if checkpoints:
                checkpoints.sort()
                adapter_path = os.path.join(model_path, checkpoints[-1])

        if merge_base:
            # Merge adapter with base model
            print(f"Loading base model: {model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()

            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            # Save merged model
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)

            return {
                "status": "success",
                "model_path": output_path,
                "type": "merged",
                "message": f"Merged model saved to {output_path}",
            }
        else:
            # Copy adapter only
            adapter_files = ["adapter_config.json", "adapter_model.safetensors",
                             "adapter_model.bin"]
            for f in adapter_files:
                src = os.path.join(adapter_path, f)
                if os.path.exists(src):
                    shutil.copy2(src, output_path)

            # Also copy tokenizer if needed
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            tokenizer.save_pretrained(output_path)

            return {
                "status": "success",
                "adapter_path": output_path,
                "type": "adapter_only",
                "message": f"Adapter saved to {output_path}",
            }

    except Exception as e:
        return {"status": "error", "message": str(e)}


def test_inference(model_path: str, test_prompts: List[str]) -> Dict[str, Any]:
    """Test model inference with sample prompts."""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Detect if it's an adapter or merged model
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            # Need to load base model + adapter
            with open(os.path.join(model_path, "adapter_config.json")) as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2.5-3B-Instruct")

            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(model, model_path)
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.eval()
        results = []

        device = next(model.parameters()).device

        for prompt in test_prompts:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )

            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            results.append({"prompt": prompt, "response": response})

        del model
        torch.cuda.empty_cache()

        return {
            "status": "success",
            "results": results,
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
