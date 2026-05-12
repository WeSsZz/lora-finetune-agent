"""
ReAct Agent core for LoRA fine-tuning.
Simplified implementation without external LLM dependency for MVP.
Uses rule-based planning with direct tool execution.
"""

import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Callable

from .models import FinetuneTask, TaskPlan, ToolCall, TaskStatus
from .prompts import SYSTEM_PROMPT, FINAL_ANSWER_PROMPT


class ToolRegistry:
    """Registry for agent tools."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable):
        self._tools[name] = func

    def get(self, name: str) -> Optional[Callable]:
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        schemas = []
        for name, func in self._tools.items():
            schema = {"name": name, "description": func.__doc__ or ""}
            schemas.append(schema)
        return schemas


class LoRAFinetuneAgent:
    """
    Agent that orchestrates LoRA fine-tuning workflow.
    Uses a deterministic execution plan based on task type.
    """

    def __init__(self):
        self.tools = ToolRegistry()
        self.tasks: Dict[str, FinetuneTask] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register tools - will be populated by external modules."""
        pass

    def register_tool(self, name: str, func: Callable):
        self.tools.register(name, func)

    def create_task(self, user_prompt: str, config_path: Optional[str] = None,
                    dataset_path: Optional[str] = None,
                    model_name: Optional[str] = None) -> str:
        task_id = str(uuid.uuid4())[:8]
        task = FinetuneTask(
            id=task_id,
            user_prompt=user_prompt,
            config_path=config_path,
            dataset_path=dataset_path,
            model_name=model_name,
        )
        self.tasks[task_id] = task
        return task_id

    def get_task(self, task_id: str) -> Optional[FinetuneTask]:
        return self.tasks.get(task_id)

    def list_tasks(self) -> List[FinetuneTask]:
        return list(self.tasks.values())

    def _parse_user_intent(self, task: FinetuneTask) -> Dict[str, Any]:
        """Extract model and dataset info from user prompt."""
        prompt = task.user_prompt.lower()
        intent = {
            "model_name": task.model_name,
            "dataset_path": task.dataset_path,
            "config_path": task.config_path,
        }

        # Try to extract model name from prompt
        if not intent["model_name"]:
            models = {
                "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
                "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
                "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
                "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
                "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
            }
            for key, model_id in models.items():
                if key in prompt:
                    intent["model_name"] = model_id
                    break

        # Default model
        if not intent["model_name"]:
            intent["model_name"] = "Qwen/Qwen2.5-3B-Instruct"

        # Try to extract dataset path
        if not intent["dataset_path"]:
            patterns = [
                r'[\s"\']([^\s"\']+\.jsonl?)[\s"\']',
                r'[\s"\']([^\s"\']+\.csv)[\s"\']',
                r'[\s"\']([^\s"\']+\.parquet)[\s"\']',
            ]
            for pattern in patterns:
                match = re.search(pattern, task.user_prompt)
                if match:
                    path = match.group(1)
                    if not path.startswith(("http", "https")):
                        intent["dataset_path"] = path
                        break

        return intent

    def _create_execution_plan(self, task: FinetuneTask) -> TaskPlan:
        """Create a deterministic execution plan."""
        steps = [
            "parse_intent",
            "analyze_dataset",
            "validate_config",
            "preprocess_data",
            "start_training",
            "monitor_training",
            "evaluate_model",
            "export_adapter",
        ]
        return TaskPlan(steps=steps)

    def _execute_tool(self, task: FinetuneTask, tool_name: str,
                      arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool and record the call."""
        tool_call = ToolCall(tool_name=tool_name, arguments=arguments)
        task.tool_calls.append(tool_call)

        func = self.tools.get(tool_name)
        if not func:
            tool_call.status = "error"
            tool_call.message = f"Tool '{tool_name}' not found"
            return {"status": "error", "message": tool_call.message}

        tool_call.status = "running"
        try:
            result = func(**arguments)
            tool_call.result = result
            tool_call.status = "success"
            return {"status": "success", "data": result}
        except Exception as e:
            tool_call.status = "error"
            tool_call.message = str(e)
            return {"status": "error", "message": str(e)}

    def run_task(self, task_id: str) -> FinetuneTask:
        """Execute a fine-tuning task with full workflow."""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        task.status = TaskStatus.PLANNING
        intent = self._parse_user_intent(task)

        # Update task with parsed intent
        task.model_name = intent.get("model_name")
        task.dataset_path = intent.get("dataset_path")
        task.config_path = intent.get("config_path")

        task.plan = self._create_execution_plan(task)
        print(f"[{task_id}] 执行计划: {task.plan.steps}")

        try:
            # Step 1: Analyze dataset
            if task.dataset_path:
                task.status = TaskStatus.DATA_PREPARING
                result = self._execute_tool(task, "analyze_dataset", {
                    "dataset_path": task.dataset_path
                })
                dataset_info = result.get("data", {}) if result.get("status") == "success" else {}
                print(f"[{task_id}] 数据集分析完成: {dataset_info}")
            else:
                dataset_info = {}
                print(f"[{task_id}] 警告: 未指定数据集路径")

            # Step 2: Validate/generate config
            config_result = self._execute_tool(task, "validate_config", {
                "config_path": task.config_path,
                "model_name": task.model_name,
                "dataset_info": dataset_info,
            })
            config = config_result.get("data", {}) if config_result.get("status") == "success" else {}
            print(f"[{task_id}] 配置验证完成")

            # Step 3: Preprocess data
            if task.dataset_path:
                preprocess_result = self._execute_tool(task, "preprocess_data", {
                    "dataset_path": task.dataset_path,
                    "output_path": f"./data/processed/{task_id}",
                    "config": config,
                    "format": dataset_info.get("format", "alpaca"),
                })
                processed_info = preprocess_result.get("data", {}) if preprocess_result.get("status") == "success" else {}
                print(f"[{task_id}] 数据预处理完成")
            else:
                processed_info = {}

            # Step 4: Start training
            task.status = TaskStatus.TRAINING
            output_dir = config.get("training_config", {}).get("output_dir", f"./outputs/{task_id}")
            task.output_dir = output_dir

            train_result = self._execute_tool(task, "start_training", {
                "config": config,
                "output_dir": output_dir,
            })

            if train_result.get("status") != "success":
                raise RuntimeError(f"训练启动失败: {train_result.get('message')}")

            print(f"[{task_id}] 训练已启动: {output_dir}")

            # Step 5: Monitor training
            training_complete = False
            max_checks = 3600  # Max 1 hour of monitoring
            checks = 0

            while not training_complete and checks < max_checks:
                time.sleep(5)
                monitor_result = self._execute_tool(task, "monitor_training", {
                    "output_dir": output_dir,
                })
                status = monitor_result.get("data", {})

                if status.get("status") == "completed":
                    training_complete = True
                    print(f"[{task_id}] 训练完成!")
                elif status.get("status") == "failed":
                    raise RuntimeError(f"训练失败: {status.get('message')}")
                elif status.get("status") == "running":
                    current = status.get("current_epoch", 0)
                    total = status.get("total_epochs", 0)
                    loss = status.get("current_loss", "N/A")
                    if checks % 12 == 0:  # Print every minute
                        print(f"[{task_id}] 训练中... Epoch {current}/{total}, Loss: {loss}")
                checks += 1

            # Step 6: Evaluate model
            task.status = TaskStatus.EVALUATING
            eval_result = self._execute_tool(task, "evaluate_model", {
                "model_path": output_dir,
                "eval_dataset_path": processed_info.get("eval_path") or task.dataset_path,
                "config": config,
            })
            eval_metrics = eval_result.get("data", {}) if eval_result.get("status") == "success" else {}
            print(f"[{task_id}] 评估完成: {eval_metrics}")

            # Step 7: Export adapter
            task.status = TaskStatus.EXPORTING
            export_result = self._execute_tool(task, "export_adapter", {
                "model_path": output_dir,
                "output_path": f"{output_dir}/final_adapter",
                "merge_base": False,
            })
            export_info = export_result.get("data", {}) if export_result.get("status") == "success" else {}
            print(f"[{task_id}] 导出完成: {export_info}")

            # Complete
            task.status = TaskStatus.COMPLETED
            task.result = {
                "model_name": task.model_name,
                "output_dir": output_dir,
                "adapter_path": export_info.get("adapter_path"),
                "eval_metrics": eval_metrics,
                "training_status": status if 'status' in dir() else {},
            }
            print(f"[{task_id}] 任务完成!")

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            print(f"[{task_id}] 任务失败: {e}")
            raise

        return task

    def run_task_async(self, task_id: str):
        """Run task in background (for API usage)."""
        import threading
        thread = threading.Thread(target=self.run_task, args=(task_id,))
        thread.daemon = True
        thread.start()
        return task_id
