"""Data processing, splitting, and output generation with streaming support."""

import itertools
import json
import random
from pathlib import Path
from typing import Dict, Iterator, List

from .config import DatasetConfig


def merge_slices_streaming(slice_iterators: List[Iterator[Dict]]) -> Iterator[Dict]:
    """
    Merge multiple dataset slices into a single stream.

    Memory-efficient: Streams data without loading everything into memory.

    Args:
        slice_iterators: List of iterators yielding records

    Yields:
        Records from all slices combined
    """
    # Using itertools.chain for cleaner, more Pythonic code
    yield from itertools.chain(*slice_iterators)


def create_train_validation_split_streaming(
    data_iterator: Iterator[Dict],
    train_split: float = 0.8,
    random_seed: int = 42
) -> tuple[List[Dict], List[Dict]]:
    """
    Split streaming data into training and validation sets.

    Note: This requires loading all data into memory for shuffling.
    For datasets too large for memory, consider pre-shuffling or
    sequential splitting.

    Args:
        data_iterator: Iterator yielding data records
        train_split: Fraction of data for training (0 to 1)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, validation_data)
    """
    # Materialize the iterator
    data = list(data_iterator)

    # Shuffle data using Random instance (thread-safe)
    rng = random.Random(random_seed)
    rng.shuffle(data)

    # Calculate split point
    split_idx = int(len(data) * train_split)

    train_data = data[:split_idx]
    validation_data = data[split_idx:]

    return train_data, validation_data


def save_dataset(
    dataset_config: DatasetConfig,
    train_data: List[Dict],
    validation_data: List[Dict],
    output_dir: str | Path = "datasets"
) -> Path:
    """
    Save train and validation data to disk in JSONL format.

    Creates the directory structure:
    output_dir/
        {dataset_name}/
            config.yml (copy of original config)
            train/
                data.jsonl
            validation/
                data.jsonl

    Args:
        dataset_config: Dataset configuration
        train_data: Training data records
        validation_data: Validation data records
        output_dir: Base directory for saving datasets

    Returns:
        Path to the created dataset directory
    """
    output_dir = Path(output_dir)
    dataset_dir = output_dir / dataset_config.name

    # Create directory structure
    train_dir = dataset_dir / "train"
    validation_dir = dataset_dir / "validation"

    train_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    # Save train data
    train_path = train_dir / "data.jsonl"
    _save_jsonl(train_data, train_path)
    print(f"Saved {len(train_data)} training records to {train_path}")

    # Save validation data
    validation_path = validation_dir / "data.jsonl"
    _save_jsonl(validation_data, validation_path)
    print(f"Saved {len(validation_data)} validation records to {validation_path}")

    # Save dataset statistics
    stats = {
        "dataset_name": dataset_config.name,
        "dataset_key": dataset_config.key,
        "total_records": len(train_data) + len(validation_data),
        "train_records": len(train_data),
        "validation_records": len(validation_data),
        "train_split": dataset_config.train_split,
        "validation_split": dataset_config.validation_split,
        "num_slices": len(dataset_config.slices),
    }

    stats_path = dataset_dir / "stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved dataset statistics to {stats_path}")

    return dataset_dir


def _save_jsonl(data: List[Dict], path: Path) -> None:
    """
    Save data to a JSONL file.

    Args:
        data: List of records to save
        path: Output file path
    """
    with open(path, 'w', encoding='utf-8') as f:
        for record in data:
            json_line = json.dumps(record, ensure_ascii=False)
            f.write(json_line + '\n')


def preprocess_data(data: List[Dict], preprocessing_fn=None) -> List[Dict]:
    """
    Apply preprocessing function to data.

    Args:
        data: List of data records
        preprocessing_fn: Optional function to apply to each record

    Returns:
        Preprocessed data
    """
    if preprocessing_fn is None:
        return data

    preprocessed = []
    for record in data:
        processed_record = preprocessing_fn(record)
        if processed_record is not None:
            preprocessed.append(processed_record)

    return preprocessed
