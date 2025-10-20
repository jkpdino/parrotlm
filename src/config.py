"""Configuration parsing and validation for dataset definitions."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import yaml

from .exceptions import ConfigError, ValidationError


@dataclass
class SliceConfig:
    """Configuration for a single dataset slice."""

    key: str
    huggingface: Optional[str] = None
    local: Optional[str] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    sampling: Literal["random", "sequential"] = "sequential"

    def __post_init__(self):
        """Validate slice configuration."""
        if not self.huggingface and not self.local:
            raise ValidationError(f"Slice '{self.key}' must have either 'huggingface' or 'local' property")

        if self.huggingface and self.local:
            raise ValidationError(f"Slice '{self.key}' cannot have both 'huggingface' and 'local' properties")

        if self.limit is not None and self.limit <= 0:
            raise ValidationError(f"Slice '{self.key}' limit must be positive")

        if self.offset is not None and self.offset < 0:
            raise ValidationError(f"Slice '{self.key}' offset must be non-negative")

        if self.sampling not in ["random", "sequential"]:
            raise ValidationError(f"Slice '{self.key}' sampling must be 'random' or 'sequential'")


@dataclass
class DatasetConfig:
    """Configuration for a complete dataset."""

    key: str
    name: str
    slices: List[SliceConfig]
    train_split: float = 0.8
    validation_split: float = 0.2
    split_strategy: Literal["shuffle", "sequential"] = "shuffle"
    random_seed: int = 42

    def __post_init__(self):
        """Validate dataset configuration."""
        if not self.key:
            raise ValidationError("Dataset 'key' is required")

        if not self.name:
            raise ValidationError("Dataset 'name' is required")

        if not self.slices:
            raise ValidationError("Dataset must have at least one slice")

        if not (0 < self.train_split < 1):
            raise ValidationError("train_split must be between 0 and 1")

        if not (0 < self.validation_split < 1):
            raise ValidationError("validation_split must be between 0 and 1")

        if abs(self.train_split + self.validation_split - 1.0) > 1e-6:
            raise ValidationError("train_split and validation_split must sum to 1.0")
        if self.split_strategy not in ["shuffle", "sequential"]:
            raise ValidationError("split_strategy must be 'shuffle' or 'sequential'")


def load_config(config_path: str | Path) -> DatasetConfig:
    """
    Load and validate a dataset configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        DatasetConfig object with validated configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    if not data:
        raise ConfigError("Config file is empty")

    # Parse slices
    slices_data = data.get('slices', [])
    slices = []

    for slice_data in slices_data:
        slice_config = SliceConfig(
            key=slice_data.get('key'),
            huggingface=slice_data.get('huggingface'),
            local=slice_data.get('local'),
            limit=slice_data.get('limit'),
            offset=slice_data.get('offset'),
            sampling=slice_data.get('sampling', 'sequential')
        )
        slices.append(slice_config)

    # Create dataset config
    dataset_config = DatasetConfig(
        key=data.get('key'),
        name=data.get('name'),
        slices=slices,
        train_split=data.get('train_split', 0.8),
        validation_split=data.get('validation_split', 0.2),
        random_seed=data.get('random_seed', 42),
        split_strategy=data.get('split_strategy', 'shuffle')
    )

    return dataset_config
