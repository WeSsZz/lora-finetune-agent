from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime


class TaskStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    DATA_PREPARING = "data_preparing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"


class ToolCall(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    status: str = "pending"  # pending, running, success, error
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class TaskPlan(BaseModel):
    steps: List[str]
    current_step: int = 0


class FinetuneTask(BaseModel):
    id: str
    user_prompt: str
    config_path: Optional[str] = None
    dataset_path: Optional[str] = None
    model_name: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    plan: Optional[TaskPlan] = None
    tool_calls: List[ToolCall] = []
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    output_dir: Optional[str] = None


class TaskCreateRequest(BaseModel):
    prompt: str = Field(..., description="用户自然语言指令，例如：用data/alpaca.jsonl微调Qwen2.5-3B")
    config_path: Optional[str] = Field(None, description="自定义配置文件路径")
    dataset_path: Optional[str] = Field(None, description="数据集文件路径")
    model_name: Optional[str] = Field(None, description="指定模型名称")


class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: str
    result: Optional[Dict[str, Any]] = None


class DatasetUploadRequest(BaseModel):
    format: str = Field(default="alpaca", description="数据格式: alpaca, sharegpt, raw")


class DatasetInfo(BaseModel):
    path: str
    format: str
    num_samples: int
    columns: List[str]
    avg_length: float
