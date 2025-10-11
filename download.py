#!/usr/bin/env python3
"""
Dataset download and processing script with streaming support.

This script downloads datasets from various sources (HuggingFace, local files),
applies sampling strategies, creates train/validation splits, and saves the
processed data in JSONL format.

Uses SOLID principles with a factory pattern for extensible data loaders.

Usage:
    python download.py <config_file>
    uv run download.py <config_file>

Example:
    python download.py configs/my_dataset.yml
"""

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from src.config import load_config
from src.loaders.factory import LoaderFactory
from src.processor import (
    create_train_validation_split_streaming,
    merge_slices_streaming,
    save_dataset,
)


def main():
    """Main entry point for the dataset download script."""
    parser = argparse.ArgumentParser(
        description="Download and process datasets from various sources"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the dataset configuration YAML file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets",
        help="Output directory for processed datasets (default: datasets)"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        print(f"\n{'='*60}")
        print(f"Loading configuration from: {args.config}")
        print(f"{'='*60}\n")

        config = load_config(args.config)

        print(f"Dataset: {config.name}")
        print(f"Key: {config.key}")
        print(f"Number of slices: {len(config.slices)}")
        print(f"Train/Validation split: {config.train_split:.0%}/{config.validation_split:.0%}")
        print()

        # Create loaders for all slices using the Factory pattern
        print(f"{'='*60}")
        print("Creating data loaders (SOLID architecture)")
        print(f"{'='*60}\n")

        loaders = []
        for i, slice_config in enumerate(config.slices, 1):
            print(f"[{i}/{len(config.slices)}] Creating loader for slice: {slice_config.key}")

            # Factory creates the appropriate loader based on configuration
            loader = LoaderFactory.create_loader(slice_config)
            print(f"  {loader.get_description()}")

            loaders.append(loader)

        print(f"\nSuccessfully created {len(loaders)} loaders")

        # Load all slices with streaming
        print(f"\n{'='*60}")
        print("Streaming dataset slices")
        print(f"{'='*60}\n")

        slice_iterators = []
        for i, loader in enumerate(loaders, 1):
            print(f"\n[{i}/{len(loaders)}] Loading slice with streaming...")
            # Each loader returns an iterator for memory efficiency
            iterator = loader.load()
            slice_iterators.append(iterator)

        # Merge slices into single stream
        print(f"\n{'='*60}")
        print("Merging slices (streaming)")
        print(f"{'='*60}\n")

        merged_stream = merge_slices_streaming(slice_iterators)
        print("Slices merged into unified stream")

        # Create train/validation split
        # Note: This step requires materializing data for shuffling
        print(f"\n{'='*60}")
        print("Creating train/validation split")
        print(f"{'='*60}\n")

        print("Materializing stream for train/val split (required for shuffling)...")
        train_data, validation_data = create_train_validation_split_streaming(
            merged_stream,
            train_split=config.train_split,
            random_seed=config.random_seed
        )

        print(f"Training records: {len(train_data)}")
        print(f"Validation records: {len(validation_data)}")

        # Save dataset
        print(f"\n{'='*60}")
        print("Saving dataset")
        print(f"{'='*60}\n")

        dataset_dir = save_dataset(
            config,
            train_data,
            validation_data,
            output_dir=args.output_dir
        )

        # Success summary
        print(f"\n{'='*60}")
        print("SUCCESS")
        print(f"{'='*60}\n")

        print(f"Dataset '{config.name}' has been successfully created!")
        print(f"Location: {dataset_dir.absolute()}")
        print(f"\nDirectory structure:")
        print(f"  {dataset_dir}/")
        print(f"    ├── train/")
        print(f"    │   └── data.jsonl ({len(train_data)} records)")
        print(f"    ├── validation/")
        print(f"    │   └── data.jsonl ({len(validation_data)} records)")
        print(f"    └── stats.json")
        print()

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
