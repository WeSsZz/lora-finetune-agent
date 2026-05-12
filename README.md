# LoRA Fine-tune Agent

基于 Agent 的 LoRA 大模型微调系统。支持 Qwen、LLaMA 等主流模型，提供 REST API 和命令行两种交互方式。

## 特性

- **Agent 自动化工作流**：数据验证 → 配置生成 → 训练执行 → 评估 → 导出，一键完成
- **多种交互方式**：REST API、命令行、Python SDK
- **内置 Qwen 支持**：预置 Qwen2.5-3B / 7B 的 LoRA 配置模板
- **4-bit 量化训练**：16GB 显存可跑 7B 模型微调
- **实时监控**：训练进度、loss 曲线、学习率实时追踪

## 环境要求

- Python 3.10+
- NVIDIA GPU（推荐 16GB+ 显存，支持 5070Ti）
- CUDA 12.1+

## 快速开始

### 1. 安装依赖

```bash
cd lora-finetune-agent

# 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动服务

```bash
# 方式一：一键启动（推荐）
bash scripts/start.sh

# 方式二：直接启动
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

访问 http://localhost:8000/docs 查看 API 文档。

### 3. 命令行使用

```bash
# 使用示例数据集进行微调
python scripts/train_cli.py "用 data/examples/alpaca_example.jsonl 微调 Qwen2.5-3B" --wait

# 指定自定义配置
python scripts/train_cli.py "微调模型" --dataset data/my_data.jsonl --model Qwen/Qwen2.5-3B-Instruct --config configs/qwen2.5_3b_lora.json --wait
```

### 4. API 使用

#### 创建任务

```bash
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "用 data/examples/alpaca_example.jsonl 微调 Qwen2.5-3B",
    "dataset_path": "data/examples/alpaca_example.jsonl",
    "model_name": "Qwen/Qwen2.5-3B-Instruct"
  }'
```

返回：`{"task_id": "abc12345", "status": "pending", "message": "..."}`

#### 启动任务

```bash
curl -X POST http://localhost:8000/api/v1/tasks/abc12345/start
```

#### 查询状态

```bash
curl http://localhost:8000/api/v1/tasks/abc12345/status
```

#### 上传数据集

```bash
curl -X POST http://localhost:8000/api/v1/datasets/upload \
  -F "file=@my_data.jsonl" \
  -F "format=alpaca"
```

#### 推理测试

```bash
curl -X POST http://localhost:8000/api/v1/models/outputs/qwen2.5-3b-lora/test \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["什么是机器学习？", "解释深度学习"]
  }'
```

## 项目结构

```
lora-finetune-agent/
├── app/
│   ├── main.py              # FastAPI 入口
│   ├── agent/
│   │   ├── core.py          # ReAct Agent 核心逻辑
│   │   ├── prompts.py       # System Prompt
│   │   └── models.py        # Pydantic 数据模型
│   ├── tools/
│   │   ├── data_tools.py    # 数据集分析、验证
│   │   ├── train_tools.py   # 训练启动、监控
│   │   ├── eval_tools.py    # 模型评估
│   │   └── deploy_tools.py  # 导出、推理测试
│   ├── core/
│   │   ├── lora_config.py   # LoRA 配置构建器
│   │   ├── trainer.py       # 训练引擎（PEFT + TRL）
│   │   └── monitor.py       # 训练监控
│   └── api/
│       └── routes.py        # REST API 路由
├── configs/
│   ├── qwen2.5_3b_lora.json   # Qwen2.5-3B 默认配置
│   └── qwen2.5_7b_lora.json   # Qwen2.5-7B 默认配置
├── data/
│   └── examples/
│       └── alpaca_example.jsonl  # 示例数据集
├── scripts/
│   ├── start.sh             # 启动脚本
│   └── train_cli.py         # 命令行工具
├── requirements.txt
└── README.md
```

## 数据格式

### Alpaca 格式（推荐）

```jsonl
{"instruction": "解释什么是机器学习", "input": "", "output": "机器学习是..."}
{"instruction": "翻译句子", "input": "Hello world", "output": "你好世界"}
```

### ShareGPT 格式

```jsonl
{"messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]}
```

### Raw 格式

```jsonl
{"text": "### Instruction:\n问题\n\n### Response:\n答案"}
```

## 配置说明

### LoRA 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| r | 16 | LoRA 秩，控制参数量 |
| lora_alpha | 32 | 缩放因子，通常 = 2*r |
| lora_dropout | 0.05 | Dropout 概率 |
| target_modules | 自动检测 | 目标模块（q_proj, v_proj 等） |

### 训练参数

| 参数 | 3B 默认值 | 7B 默认值 | 说明 |
|------|-----------|-----------|------|
| per_device_train_batch_size | 2 | 1 | 每设备 batch size |
| gradient_accumulation_steps | 4 | 8 | 梯度累积步数 |
| learning_rate | 2e-4 | 2e-4 | 学习率 |
| num_train_epochs | 3 | 3 | 训练轮数 |
| max_seq_length | 2048 | 2048 | 最大序列长度 |

## 显存占用参考

| 模型 | 量化 | LoRA r=16 | 预估显存 |
|------|------|-----------|----------|
| Qwen2.5-3B | 4-bit | ✅ | ~8GB |
| Qwen2.5-7B | 4-bit | ✅ | ~12GB |
| Qwen2.5-14B | 4-bit | ✅ | ~20GB |

5070Ti 移动端 16GB 显存可以流畅跑 7B 模型的 LoRA 微调。

## 常见问题

### 1. CUDA out of memory

- 减小 `per_device_train_batch_size` 到 1
- 增大 `gradient_accumulation_steps`
- 减小 `max_seq_length`
- 使用更小的 `r` 值（如 8）

### 2. 训练 loss 为 NaN

- 降低 `learning_rate`（如 1e-4 或 5e-5）
- 增大 `warmup_ratio`
- 检查数据是否有异常样本

### 3. 下载模型速度慢

设置镜像源：
```bash
# Linux/Mac
export HF_ENDPOINT=https://hf-mirror.com

# Windows
set HF_ENDPOINT=https://hf-mirror.com
```

## 路线图

- [ ] 支持更多模型（DeepSeek、Phi、Gemma）
- [ ] 支持 DPO/ORPO 对齐训练
- [ ] Web UI 界面
- [ ] 分布式多卡训练
- [ ] Wandb 集成
- [ ] Java SDK / Spring Boot Starter

## 技术栈

- **Agent 框架**：自研 ReAct（简化版）
- **训练引擎**：PEFT + Transformers + TRL
- **量化**：BitsAndBytes 4-bit
- **API 框架**：FastAPI
- **数据**：Datasets + Pandas

## License

MIT
