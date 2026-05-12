"""FastAPI routes for LoRA fine-tuning agent."""

import os
import shutil
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.agent.core import LoRAFinetuneAgent
from app.agent.models import (
    DatasetInfo,
    FinetuneTask,
    TaskCreateRequest,
    TaskResponse,
    TaskStatus,
)
from app.tools.data_tools import analyze_dataset
from app.tools.deploy_tools import export_adapter, test_inference

router = APIRouter()

# Global agent instance
_agent: Optional[LoRAFinetuneAgent] = None


def get_agent() -> LoRAFinetuneAgent:
    """Get or initialize the agent."""
    global _agent
    if _agent is None:
        _agent = LoRAFinetuneAgent()
        _register_tools()
    return _agent


def _register_tools():
    """Register all tools with the agent."""
    from app.tools.data_tools import analyze_dataset, validate_config
    from app.tools.train_tools import preprocess_data, start_training, monitor_training
    from app.tools.eval_tools import evaluate_model
    from app.tools.deploy_tools import export_adapter

    agent = get_agent()
    agent.register_tool("analyze_dataset", analyze_dataset)
    agent.register_tool("validate_config", validate_config)
    agent.register_tool("preprocess_data", preprocess_data)
    agent.register_tool("start_training", start_training)
    agent.register_tool("monitor_training", monitor_training)
    agent.register_tool("evaluate_model", evaluate_model)
    agent.register_tool("export_adapter", export_adapter)


@router.post("/tasks", response_model=TaskResponse)
async def create_task(request: TaskCreateRequest):
    """Create a new fine-tuning task."""
    agent = get_agent()
    task_id = agent.create_task(
        user_prompt=request.prompt,
        config_path=request.config_path,
        dataset_path=request.dataset_path,
        model_name=request.model_name,
    )

    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message=f"Task created. Use GET /tasks/{task_id}/status to monitor.",
    )


@router.post("/tasks/{task_id}/start")
async def start_task(task_id: str):
    """Start executing a fine-tuning task."""
    agent = get_agent()
    task = agent.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != TaskStatus.PENDING:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Task already {task.status}"},
        )

    # Start in background
    agent.run_task_async(task_id)

    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.PLANNING,
        message="Task started in background",
    )


@router.get("/tasks/{task_id}/status")
async def get_task_status(task_id: str):
    """Get task status and progress."""
    agent = get_agent()
    task = agent.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task.id,
        "status": task.status,
        "model_name": task.model_name,
        "dataset_path": task.dataset_path,
        "output_dir": task.output_dir,
        "plan": task.plan.dict() if task.plan else None,
        "tool_calls": [
            {
                "tool": tc.tool_name,
                "status": tc.status,
                "message": tc.message,
            }
            for tc in task.tool_calls
        ],
        "result": task.result,
        "error_message": task.error_message,
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
    }


@router.get("/tasks", response_model=List[Dict[str, Any]])
async def list_tasks():
    """List all tasks."""
    agent = get_agent()
    tasks = agent.list_tasks()
    return [
        {
            "task_id": t.id,
            "status": t.status,
            "model_name": t.model_name,
            "created_at": t.created_at.isoformat(),
        }
        for t in sorted(tasks, key=lambda x: x.created_at, reverse=True)
    ]


@router.post("/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    format: str = Form(default="alpaca"),
):
    """Upload a dataset file."""
    # Validate file extension
    allowed_extensions = [".jsonl", ".json", ".csv", ".parquet"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_extensions}",
        )

    # Save file
    upload_dir = "data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f"{uuid.uuid4().hex}{ext}")

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Analyze dataset
    info = analyze_dataset(file_path)

    return {
        "status": "success",
        "file_path": file_path,
        "original_name": file.filename,
        "analysis": info,
    }


@router.get("/datasets/{dataset_path:path}/info")
async def get_dataset_info(dataset_path: str):
    """Get dataset information."""
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    info = analyze_dataset(dataset_path)
    return info


@router.post("/models/{model_path:path}/test")
async def test_model(
    model_path: str,
    prompts: List[str],
):
    """Test model inference."""
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")

    result = test_inference(model_path, prompts)
    return result


@router.post("/models/{model_path:path}/export")
async def export_model(
    model_path: str,
    output_path: Optional[str] = None,
    merge_base: bool = False,
):
    """Export LoRA adapter or merged model."""
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")

    if not output_path:
        output_path = f"{model_path}_exported"

    result = export_adapter(model_path, output_path, merge_base)
    return result


@router.get("/configs")
async def list_configs():
    """List available config templates."""
    configs_dir = "configs"
    if not os.path.exists(configs_dir):
        return {"configs": []}

    configs = []
    for f in os.listdir(configs_dir):
        if f.endswith(".json"):
            configs.append({
                "name": f.replace(".json", ""),
                "path": os.path.join(configs_dir, f),
            })

    return {"configs": configs}
