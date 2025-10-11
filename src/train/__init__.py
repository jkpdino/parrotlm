"""Training module for GPT models."""

from .checkpoints import CheckpointManager, CheckpointMetadata
from .config import (
    CheckpointConfig,
    DataConfig,
    ModelConfig,
    MonitoringConfig,
    TrainingConfig,
    TrainingRunConfig,
    load_training_config,
    save_training_config,
)
from .inference import InferenceMonitor, generate_sample, generate_multiple_samples
from .metrics import MetricsTracker, MetricPoint
from .model import GPT, create_default_gpt, create_mqa_gpt
from .muon import Muon, create_hybrid_optimizer, HybridOptimizer
from .pipeline import DocumentBatchLoader, visualize_attention_mask

__all__ = [
    # Config
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "CheckpointConfig",
    "MonitoringConfig",
    "TrainingRunConfig",
    "load_training_config",
    "save_training_config",
    # Checkpoints
    "CheckpointManager",
    "CheckpointMetadata",
    # Metrics
    "MetricsTracker",
    "MetricPoint",
    # Model
    "GPT",
    "create_default_gpt",
    "create_mqa_gpt",
    # Optimizer
    "Muon",
    "create_hybrid_optimizer",
    "HybridOptimizer",
    # Pipeline
    "DocumentBatchLoader",
    "visualize_attention_mask",
    # Inference
    "InferenceMonitor",
    "generate_sample",
    "generate_multiple_samples",
]
