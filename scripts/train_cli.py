#!/usr/bin/env python3
"""
CLI interface for LoRA fine-tuning agent.
Usage: python train_cli.py "用 data/examples/alpaca_example.jsonl 微调 Qwen2.5-3B"
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent.core import LoRAFinetuneAgent
from app.tools.data_tools import analyze_dataset, validate_config
from app.tools.train_tools import preprocess_data, start_training, monitor_training
from app.tools.eval_tools import evaluate_model
from app.tools.deploy_tools import export_adapter


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning Agent CLI")
    parser.add_argument("prompt", help="自然语言指令，例如：用data/train.jsonl微调Qwen2.5-3B")
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--dataset", help="数据集路径")
    parser.add_argument("--model", help="模型名称")
    parser.add_argument("--wait", action="store_true", help="等待训练完成")

    args = parser.parse_args()

    # Initialize agent
    agent = LoRAFinetuneAgent()
    agent.register_tool("analyze_dataset", analyze_dataset)
    agent.register_tool("validate_config", validate_config)
    agent.register_tool("preprocess_data", preprocess_data)
    agent.register_tool("start_training", start_training)
    agent.register_tool("monitor_training", monitor_training)
    agent.register_tool("evaluate_model", evaluate_model)
    agent.register_tool("export_adapter", export_adapter)

    # Create task
    task_id = agent.create_task(
        user_prompt=args.prompt,
        config_path=args.config,
        dataset_path=args.dataset,
        model_name=args.model,
    )

    print(f"[系统] 任务已创建: {task_id}")
    print(f"[系统] 用户指令: {args.prompt}")
    print("=" * 50)

    # Run task
    if args.wait:
        try:
            task = agent.run_task(task_id)
            print("=" * 50)
            print(f"[系统] 任务完成!")
            print(f"[结果] 输出目录: {task.result.get('output_dir')}")
            print(f"[结果] 评估指标: {task.result.get('eval_metrics')}")
        except Exception as e:
            print(f"[错误] 任务失败: {e}")
            sys.exit(1)
    else:
        agent.run_task_async(task_id)
        print(f"[系统] 任务已在后台启动")
        print(f"[提示] 使用以下命令查看状态:")
        print(f"  curl http://localhost:8000/api/v1/tasks/{task_id}/status")


if __name__ == "__main__":
    main()
