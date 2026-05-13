# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Start the service
```bash
bash scripts/start.sh
# Or directly:
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Run training via CLI
```bash
python scripts/train_cli.py "用 data/examples/alpaca_example.jsonl 微调 Qwen2.5-3B" --wait
```

### Health check
```bash
curl http://localhost:8000/health
```

## Architecture

### Agent execution model
`app/agent/core.py::LoRAFinetuneAgent` uses a **deterministic execution plan**, not a dynamic LLM planner. The plan is always:

`parse_intent → analyze_dataset → validate_config → preprocess_data → start_training → monitor_training → evaluate_model → export_adapter`

The agent is initialized lazily in `app/api/routes.py::get_agent()`. Tools are registered at module import time via `_register_tools()`.

### Training engine
`app/core/trainer.py::LoRATrainer` is the single entry point for training. It:

1. Loads the base model with `BitsAndBytesConfig` (4-bit NF4 quantization)
2. Applies LoRA via `get_peft_model()`
3. Uses `trl.SFTTrainer` for supervised fine-tuning
4. Writes progress to `training_log.jsonl` via `TrainingMonitorCallback`

**Critical detail**: `train()` runs synchronously. The background thread is created in `app/tools/train_tools.py::start_training()`, which stores the `LoRATrainer` instance in `_active_trainers[output_dir]`. `monitor_training()` first checks `_active_trainers`, then falls back to reading the log file.

### Config schema
Every config JSON is a flat dictionary with four top-level keys:
- `model_name` — HuggingFace model ID
- `lora_config` — passed to `peft.LoraConfig`
- `training_config` — **filtered** to `TrainingArguments.__dataclass_fields__` in `trainer.py` before construction; keys not present in `TrainingArguments` are silently dropped
- `quantization_config` — passed to `transformers.BitsAndBytesConfig`
- `data_config` — consumed by `_load_dataset()` for format selection and `max_seq_length`

`app/core/lora_config.py::LoRAConfigBuilder` auto-detects model family from the name string and adjusts `per_device_train_batch_size` / `gradient_accumulation_steps` by estimated model size.

### Data formatting
`LoRATrainer._format_sample()` formats samples before `Dataset.from_list()`. Supported formats:
- `alpaca`: `instruction` + optional `input` + `output` → `### Instruction:\n...\n\n### Response:\n...`
- `sharegpt`: uses `tokenizer.apply_chat_template()` if available; falls back to concatenating `role: content`
- `raw`: passes through `text` field verbatim

### Pydantic models
`app/agent/models.py` uses Pydantic v2. `FinetuneTask.model_name` triggers a namespace warning because `model_` is a protected prefix. This is non-fatal; if you want to silence it, add `model_config = ConfigDict(protected_namespaces=())` to the model.

## Deployment notes

- On AutoDL (or other China-hosted environments), set `HF_ENDPOINT=https://hf-mirror.com` before running to avoid slow or blocked downloads from huggingface.co.
- The first training run downloads the base model; this can take a long time and produces no log output until the download completes.
- Training runs in a `daemon=True` thread. If the main process exits, the training thread is killed.
