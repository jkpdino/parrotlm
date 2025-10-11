"""Configuration parsing and validation for training runs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import yaml

from ..exceptions import ConfigError, ValidationError


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    vocab_size: int = 16384
    dim: int = 96
    depth: int = 16
    n_heads: int = 6
    n_kv_heads: Optional[int] = 2
    context_length: int = 512
    dropout: float = 0.0
    ffn_hidden_mult: int = 4
    pad_token_id: int = 0
    rope_scaling_factor: float = 4.0
    use_gradient_checkpointing: bool = False

    # MoE (Mixture of Experts) configuration
    use_moe: bool = False
    num_experts: int = 8
    num_experts_active: int = 2
    moe_layers: Optional[list[int]] = None  # Which layers use MoE (None = all layers)
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001

    def __post_init__(self):
        """Validate model configuration."""
        if self.vocab_size <= 0:
            raise ValidationError("vocab_size must be positive")
        if self.dim <= 0:
            raise ValidationError("dim must be positive")
        if self.depth <= 0:
            raise ValidationError("depth must be positive")
        if self.n_heads <= 0:
            raise ValidationError("n_heads must be positive")
        if self.dim % self.n_heads != 0:
            raise ValidationError(f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})")
        if self.n_kv_heads is not None:
            if self.n_kv_heads <= 0:
                raise ValidationError("n_kv_heads must be positive")
            if self.n_heads % self.n_kv_heads != 0:
                raise ValidationError(
                    f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
                )
        if self.context_length <= 0:
            raise ValidationError("context_length must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValidationError("dropout must be in range [0, 1)")
        # MoE validation
        if self.use_moe:
            if self.num_experts <= 0:
                raise ValidationError("num_experts must be positive")
            if self.num_experts_active <= 0:
                raise ValidationError("num_experts_active must be positive")
            if self.num_experts_active > self.num_experts:
                raise ValidationError(
                    f"num_experts_active ({self.num_experts_active}) cannot exceed num_experts ({self.num_experts})"
                )
            if self.moe_layers is not None:
                if not all(0 <= layer < self.depth for layer in self.moe_layers):
                    raise ValidationError(
                        f"moe_layers must contain valid layer indices in range [0, {self.depth})"
                    )
            if self.router_aux_loss_coef < 0:
                raise ValidationError("router_aux_loss_coef must be non-negative")
            if self.router_z_loss_coef < 0:
                raise ValidationError("router_z_loss_coef must be non-negative")


@dataclass
class DataConfig:
    """Configuration for data loading."""

    dataset_path: str
    tokenizer_path: str
    batch_size: int = 8
    shuffle: bool = True
    num_workers: int = 0

    def __post_init__(self):
        """Validate data configuration."""
        if not self.dataset_path:
            raise ValidationError("dataset_path is required")
        if not self.tokenizer_path:
            raise ValidationError("tokenizer_path is required")
        if self.batch_size <= 0:
            raise ValidationError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValidationError("num_workers must be non-negative")


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    learning_rate: float = 0.001
    max_batches: Optional[int] = None
    max_tokens: Optional[int] = None
    optimizer: Literal["muon", "adamw"] = "muon"
    warmup_batches: int = 0
    weight_decay: float = 0.1
    grad_clip: Optional[float] = 1.0
    lr_schedule: Literal["constant", "cosine", "linear"] = "constant"
    min_lr_ratio: float = 0.1

    def __post_init__(self):
        """Validate training configuration."""
        if self.learning_rate <= 0:
            raise ValidationError("learning_rate must be positive")
        if self.max_batches is None and self.max_tokens is None:
            raise ValidationError("Either max_batches or max_tokens must be specified")
        if self.max_batches is not None and self.max_batches <= 0:
            raise ValidationError("max_batches must be positive")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValidationError("max_tokens must be positive")
        if self.optimizer not in ["muon", "adamw"]:
            raise ValidationError("optimizer must be 'muon' or 'adamw'")
        if self.warmup_batches < 0:
            raise ValidationError("warmup_batches must be non-negative")
        if self.weight_decay < 0:
            raise ValidationError("weight_decay must be non-negative")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValidationError("grad_clip must be positive")
        if not 0.0 < self.min_lr_ratio <= 1.0:
            raise ValidationError("min_lr_ratio must be in range (0, 1]")


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing."""

    save_every_n_batches: int = 100
    keep_last_n: int = 10
    output_dir: str = "training_runs"

    def __post_init__(self):
        """Validate checkpoint configuration."""
        if self.save_every_n_batches <= 0:
            raise ValidationError("save_every_n_batches must be positive")
        if self.keep_last_n < 0:
            raise ValidationError("keep_last_n must be non-negative")
        if not self.output_dir:
            raise ValidationError("output_dir is required")


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and validation."""

    validate_every_n_batches: int = 100
    validation_samples: int = 100
    inference_every_n_batches: int = 100
    inference_prompt: str = "Once upon a time"
    inference_max_tokens: int = 50
    inference_temperature: float = 1.0
    graph_window_size: int = 1000

    def __post_init__(self):
        """Validate monitoring configuration."""
        if self.validate_every_n_batches <= 0:
            raise ValidationError("validate_every_n_batches must be positive")
        if self.validation_samples <= 0:
            raise ValidationError("validation_samples must be positive")
        if self.inference_every_n_batches <= 0:
            raise ValidationError("inference_every_n_batches must be positive")
        if self.inference_max_tokens <= 0:
            raise ValidationError("inference_max_tokens must be positive")
        if self.inference_temperature <= 0:
            raise ValidationError("inference_temperature must be positive")
        if self.graph_window_size <= 0:
            raise ValidationError("graph_window_size must be positive")


@dataclass
class TrainingRunConfig:
    """Top-level configuration for a training run."""

    name: str
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    checkpointing: CheckpointConfig
    monitoring: MonitoringConfig
    random_seed: int = 42

    def __post_init__(self):
        """Validate training run configuration."""
        if not self.name:
            raise ValidationError("name is required")


def load_training_config(config_path: str | Path) -> TrainingRunConfig:
    """
    Load and validate a training configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        TrainingRunConfig object with validated configuration

    Raises:
        ConfigError: If config file doesn't exist or is invalid
        ValidationError: If configuration values are invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    if not data:
        raise ConfigError("Config file is empty")

    # Parse model config
    model_data = data.get('model', {})
    model_config = ModelConfig(
        vocab_size=model_data.get('vocab_size', 16384),
        dim=model_data.get('dim', 96),
        depth=model_data.get('depth', 16),
        n_heads=model_data.get('n_heads', 6),
        n_kv_heads=model_data.get('n_kv_heads', 2),
        context_length=model_data.get('context_length', 512),
        dropout=model_data.get('dropout', 0.0),
        ffn_hidden_mult=model_data.get('ffn_hidden_mult', 4),
        pad_token_id=model_data.get('pad_token_id', 0),
        rope_scaling_factor=model_data.get('rope_scaling_factor', 4.0),
        use_gradient_checkpointing=model_data.get('use_gradient_checkpointing', False),
        # MoE parameters
        use_moe=model_data.get('use_moe', False),
        num_experts=model_data.get('num_experts', 8),
        num_experts_active=model_data.get('num_experts_active', 2),
        moe_layers=model_data.get('moe_layers', None),
        router_aux_loss_coef=model_data.get('router_aux_loss_coef', 0.01),
        router_z_loss_coef=model_data.get('router_z_loss_coef', 0.001)
    )

    # Parse data config
    data_config_data = data.get('data', {})
    data_config = DataConfig(
        dataset_path=data_config_data.get('dataset_path'),
        tokenizer_path=data_config_data.get('tokenizer_path'),
        batch_size=data_config_data.get('batch_size', 8),
        shuffle=data_config_data.get('shuffle', True),
        num_workers=data_config_data.get('num_workers', 0)
    )

    # Parse training config
    training_data = data.get('training', {})
    training_config = TrainingConfig(
        learning_rate=training_data.get('learning_rate', 0.001),
        max_batches=training_data.get('max_batches'),
        max_tokens=training_data.get('max_tokens'),
        optimizer=training_data.get('optimizer', 'muon'),
        warmup_batches=training_data.get('warmup_batches', 0),
        weight_decay=training_data.get('weight_decay', 0.1),
        grad_clip=training_data.get('grad_clip', 1.0),
        lr_schedule=training_data.get('lr_schedule', 'constant'),
        min_lr_ratio=training_data.get('min_lr_ratio', 0.1)
    )

    # Parse checkpoint config
    checkpoint_data = data.get('checkpointing', {})
    checkpoint_config = CheckpointConfig(
        save_every_n_batches=checkpoint_data.get('save_every_n_batches', 100),
        keep_last_n=checkpoint_data.get('keep_last_n', 10),
        output_dir=checkpoint_data.get('output_dir', 'training_runs')
    )

    # Parse monitoring config
    monitoring_data = data.get('monitoring', {})
    monitoring_config = MonitoringConfig(
        validate_every_n_batches=monitoring_data.get('validate_every_n_batches', 100),
        validation_samples=monitoring_data.get('validation_samples', 100),
        inference_every_n_batches=monitoring_data.get('inference_every_n_batches', 100),
        inference_prompt=monitoring_data.get('inference_prompt', 'Once upon a time'),
        inference_max_tokens=monitoring_data.get('inference_max_tokens', 50),
        inference_temperature=monitoring_data.get('inference_temperature', 1.0),
        graph_window_size=monitoring_data.get('graph_window_size', 1000)
    )

    # Create training run config
    training_run_config = TrainingRunConfig(
        name=data.get('name'),
        model=model_config,
        data=data_config,
        training=training_config,
        checkpointing=checkpoint_config,
        monitoring=monitoring_config,
        random_seed=data.get('random_seed', 42)
    )

    return training_run_config


def save_training_config(config: TrainingRunConfig, output_path: str | Path) -> None:
    """
    Save a training configuration to a YAML file.

    Args:
        config: TrainingRunConfig to save
        output_path: Path to save the YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict
    config_dict = {
        'name': config.name,
        'random_seed': config.random_seed,
        'model': {
            'vocab_size': config.model.vocab_size,
            'dim': config.model.dim,
            'depth': config.model.depth,
            'n_heads': config.model.n_heads,
            'n_kv_heads': config.model.n_kv_heads,
            'context_length': config.model.context_length,
            'dropout': config.model.dropout,
            'ffn_hidden_mult': config.model.ffn_hidden_mult,
            'pad_token_id': config.model.pad_token_id,
            'rope_scaling_factor': config.model.rope_scaling_factor,
            'use_gradient_checkpointing': config.model.use_gradient_checkpointing,
            # MoE parameters
            'use_moe': config.model.use_moe,
            'num_experts': config.model.num_experts,
            'num_experts_active': config.model.num_experts_active,
            'moe_layers': config.model.moe_layers,
            'router_aux_loss_coef': config.model.router_aux_loss_coef,
            'router_z_loss_coef': config.model.router_z_loss_coef
        },
        'data': {
            'dataset_path': config.data.dataset_path,
            'tokenizer_path': config.data.tokenizer_path,
            'batch_size': config.data.batch_size,
            'shuffle': config.data.shuffle,
            'num_workers': config.data.num_workers
        },
        'training': {
            'learning_rate': config.training.learning_rate,
            'max_batches': config.training.max_batches,
            'max_tokens': config.training.max_tokens,
            'optimizer': config.training.optimizer,
            'warmup_batches': config.training.warmup_batches,
            'weight_decay': config.training.weight_decay,
            'grad_clip': config.training.grad_clip,
            'lr_schedule': config.training.lr_schedule,
            'min_lr_ratio': config.training.min_lr_ratio
        },
        'checkpointing': {
            'save_every_n_batches': config.checkpointing.save_every_n_batches,
            'keep_last_n': config.checkpointing.keep_last_n,
            'output_dir': config.checkpointing.output_dir
        },
        'monitoring': {
            'validate_every_n_batches': config.monitoring.validate_every_n_batches,
            'validation_samples': config.monitoring.validation_samples,
            'inference_every_n_batches': config.monitoring.inference_every_n_batches,
            'inference_prompt': config.monitoring.inference_prompt,
            'inference_max_tokens': config.monitoring.inference_max_tokens,
            'inference_temperature': config.monitoring.inference_temperature,
            'graph_window_size': config.monitoring.graph_window_size
        }
    }

    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
