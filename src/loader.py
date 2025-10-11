"""Data loading from various sources (HuggingFace, local files)."""

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from .config import SliceConfig


def load_slice_data(slice_config: SliceConfig) -> List[Dict]:
    """
    Load data for a single slice from HuggingFace or local source.

    Args:
        slice_config: Configuration for the slice to load

    Returns:
        List of dictionaries representing the dataset records

    Raises:
        ValueError: If data source is invalid or cannot be loaded
    """
    if slice_config.huggingface:
        return _load_huggingface_slice(slice_config)
    elif slice_config.local:
        return _load_local_slice(slice_config)
    else:
        raise ValueError(f"Slice '{slice_config.key}' has no valid data source")


def _load_huggingface_slice(slice_config: SliceConfig) -> List[Dict]:
    """
    Load data from HuggingFace datasets.

    Args:
        slice_config: Configuration with huggingface dataset path

    Returns:
        List of dictionaries with the loaded data
    """
    print(f"Loading HuggingFace dataset: {slice_config.huggingface}")

    # Parse dataset path (format: "dataset_name" or "dataset_name:split")
    parts = slice_config.huggingface.split(':')
    dataset_name = parts[0]
    split = parts[1] if len(parts) > 1 else 'train'

    try:
        # Load dataset from HuggingFace
        dataset = load_dataset(dataset_name, split=split)

        # Apply offset
        offset = slice_config.offset or 0
        if offset > 0:
            dataset = dataset.select(range(offset, len(dataset)))

        # Apply limit
        if slice_config.limit is not None:
            limit = min(slice_config.limit, len(dataset))
            dataset = dataset.select(range(limit))

        # Apply sampling strategy
        if slice_config.sampling == "random":
            dataset = dataset.shuffle(seed=42)

        # Convert to list of dicts
        data = []
        for item in tqdm(dataset, desc=f"Loading {slice_config.key}"):
            data.append(dict(item))

        print(f"Loaded {len(data)} records from {slice_config.huggingface}")
        return data

    except Exception as e:
        raise ValueError(f"Failed to load HuggingFace dataset '{slice_config.huggingface}': {str(e)}")


def _load_local_slice(slice_config: SliceConfig) -> List[Dict]:
    """
    Load data from local files (JSONL, JSON, CSV, Parquet).

    Args:
        slice_config: Configuration with local file path

    Returns:
        List of dictionaries with the loaded data
    """
    local_path = Path(slice_config.local)

    if not local_path.exists():
        raise ValueError(f"Local file not found: {local_path}")

    print(f"Loading local file: {local_path}")

    try:
        # Determine file type and load accordingly
        if local_path.suffix == '.jsonl':
            data = _load_jsonl(local_path)
        elif local_path.suffix == '.json':
            data = _load_json(local_path)
        elif local_path.suffix == '.csv':
            data = _load_csv(local_path)
        elif local_path.suffix in ['.parquet', '.pq']:
            data = _load_parquet(local_path)
        else:
            raise ValueError(f"Unsupported file format: {local_path.suffix}")

        # Apply offset
        offset = slice_config.offset or 0
        if offset > 0:
            data = data[offset:]

        # Apply limit
        if slice_config.limit is not None:
            data = data[:slice_config.limit]

        # Apply sampling strategy
        if slice_config.sampling == "random":
            import random
            random.seed(42)
            data = random.sample(data, len(data))

        print(f"Loaded {len(data)} records from {local_path}")
        return data

    except Exception as e:
        raise ValueError(f"Failed to load local file '{local_path}': {str(e)}")


def _load_jsonl(path: Path) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def _load_json(path: Path) -> List[Dict]:
    """Load data from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle both list and single object
    if isinstance(data, list):
        return data
    else:
        return [data]


def _load_csv(path: Path) -> List[Dict]:
    """Load data from CSV file."""
    df = pd.read_csv(path)
    return df.to_dict('records')


def _load_parquet(path: Path) -> List[Dict]:
    """Load data from Parquet file."""
    df = pd.read_parquet(path)
    return df.to_dict('records')
