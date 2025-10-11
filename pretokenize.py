#!/usr/bin/env python3
"""
Dataset pretokenization script.

This script pretokenizes datasets for faster training by converting text to
token IDs ahead of time and saving them in a binary format.

Usage:
    python pretokenize.py <config_file> --tokenizer <tokenizer_name>
    uv run pretokenize.py <config_file> --tokenizer <tokenizer_name>

Example:
    python pretokenize.py configs/tinystories_100k.yml --tokenizer gpt2
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from src.config import load_config
from src.exceptions import ConfigError, LoaderError
from src.tokenizer import (
    load_tokenizer,
    save_document_tokens,
    save_metadata,
    tokenize_jsonl,
)


def get_tokenizer_short_name(tokenizer_name: str) -> str:
    """
    Get a short name for the tokenizer for directory naming.

    Args:
        tokenizer_name: Full tokenizer name (e.g., "meta-llama/Llama-2-7b-hf")

    Returns:
        Short name (e.g., "Llama-2-7b-hf")
    """
    # If it has a slash, take the part after it
    if '/' in tokenizer_name:
        short_name = tokenizer_name.split('/')[-1]
    else:
        short_name = tokenizer_name

    # Strip .json extension if present
    if short_name.endswith('.json'):
        short_name = short_name[:-5]

    return short_name


def main():
    """Main entry point for the pretokenization script."""
    parser = argparse.ArgumentParser(
        description="Pretokenize datasets for faster training"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the dataset configuration YAML file"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="HuggingFace tokenizer name (e.g., 'gpt2', 'meta-llama/Llama-2-7b-hf')"
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Name of the text field in JSON objects (default: text)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets",
        help="Base directory containing datasets (default: datasets)"
    )
    parser.add_argument(
        "--no-eos",
        action="store_true",
        help="Do not add EOS token after each document"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retokenization even if output files exist"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        print(f"\n{'='*60}")
        print(f"Loading configuration from: {args.config}")
        print(f"{'='*60}\n")

        config = load_config(args.config)

        print(f"Dataset: {config.name}")
        print(f"Tokenizer: {args.tokenizer}")
        print(f"Text field: {args.text_field}")
        print()

        # Determine paths
        output_dir = Path(args.output_dir)
        dataset_dir = output_dir / config.name

        # Check if dataset exists
        train_jsonl = dataset_dir / "train" / "data.jsonl"
        validation_jsonl = dataset_dir / "validation" / "data.jsonl"

        if not train_jsonl.exists():
            print(f"Error: Training data not found at {train_jsonl}", file=sys.stderr)
            print(f"Please run download.py first to create the dataset.", file=sys.stderr)
            sys.exit(1)

        if not validation_jsonl.exists():
            print(f"Error: Validation data not found at {validation_jsonl}", file=sys.stderr)
            print(f"Please run download.py first to create the dataset.", file=sys.stderr)
            sys.exit(1)

        # Prepare output directory
        tokenizer_short_name = get_tokenizer_short_name(args.tokenizer)
        pretokenized_dir = dataset_dir / "pretokenized" / tokenizer_short_name

        train_output = pretokenized_dir / "train.npy"
        train_offsets_output = pretokenized_dir / "train_offsets.npy"
        validation_output = pretokenized_dir / "validation.npy"
        validation_offsets_output = pretokenized_dir / "validation_offsets.npy"
        metadata_output = pretokenized_dir / "metadata.json"

        # Check if output already exists
        if (train_output.exists() and train_offsets_output.exists() and
            validation_output.exists() and validation_offsets_output.exists() and
            not args.force):
            print(f"{'='*60}")
            print("ALREADY EXISTS")
            print(f"{'='*60}\n")
            print(f"Pretokenized data already exists at:")
            print(f"  {pretokenized_dir}/")
            print(f"    ├── train.npy")
            print(f"    ├── train_offsets.npy")
            print(f"    ├── validation.npy")
            print(f"    ├── validation_offsets.npy")
            print(f"    └── metadata.json")
            print()
            print("Use --force to retokenize anyway.")
            return

        # Load tokenizer
        print(f"{'='*60}")
        print("Loading tokenizer")
        print(f"{'='*60}\n")

        tokenizer = load_tokenizer(args.tokenizer)

        print(f"Tokenizer loaded successfully")
        vocab_size = tokenizer.get_vocab_size()
        print(f"Vocabulary size: {vocab_size}")

        # Try to find EOS token
        for eos_token in ["<|endoftext|>", "</s>", "<eos>", "[EOS]"]:
            token_id = tokenizer.token_to_id(eos_token)
            if token_id is not None:
                print(f"EOS token: {eos_token} (ID: {token_id})")
                break
        print()

        # Tokenize training data
        print(f"{'='*60}")
        print("Tokenizing training data")
        print(f"{'='*60}\n")

        train_tokens, train_offsets = tokenize_jsonl(
            train_jsonl,
            tokenizer,
            text_field=args.text_field,
            add_eos=not args.no_eos
        )

        num_train_docs = len(train_offsets) - 1
        print(f"\nTraining documents: {num_train_docs:,}")
        print(f"Training tokens: {len(train_tokens):,}")

        # Tokenize validation data
        print(f"\n{'='*60}")
        print("Tokenizing validation data")
        print(f"{'='*60}\n")

        validation_tokens, validation_offsets = tokenize_jsonl(
            validation_jsonl,
            tokenizer,
            text_field=args.text_field,
            add_eos=not args.no_eos
        )

        num_val_docs = len(validation_offsets) - 1
        print(f"\nValidation documents: {num_val_docs:,}")
        print(f"Validation tokens: {len(validation_tokens):,}")

        # Save tokenized data
        print(f"\n{'='*60}")
        print("Saving pretokenized data")
        print(f"{'='*60}\n")

        save_document_tokens(train_tokens, train_offsets, train_output, train_offsets_output)
        print(f"Saved training tokens to {train_output}")
        print(f"Saved training offsets to {train_offsets_output}")

        save_document_tokens(validation_tokens, validation_offsets, validation_output, validation_offsets_output)
        print(f"Saved validation tokens to {validation_output}")
        print(f"Saved validation offsets to {validation_offsets_output}")

        # Save metadata
        metadata = {
            "tokenizer_name": args.tokenizer,
            "vocab_size": vocab_size,
            "train_tokens": int(len(train_tokens)),
            "train_documents": num_train_docs,
            "validation_tokens": int(len(validation_tokens)),
            "validation_documents": num_val_docs,
            "total_tokens": int(len(train_tokens) + len(validation_tokens)),
            "total_documents": num_train_docs + num_val_docs,
            "text_field": args.text_field,
            "add_eos": not args.no_eos,
            "dtype": str(train_tokens.dtype),
            "offset_dtype": str(train_offsets.dtype),
            "created_at": datetime.now().isoformat(),
            "dataset_name": config.name,
            "dataset_key": config.key,
        }

        save_metadata(metadata, metadata_output)
        print(f"Saved metadata to {metadata_output}")

        # Success summary
        print(f"\n{'='*60}")
        print("SUCCESS")
        print(f"{'='*60}\n")

        print(f"Dataset '{config.name}' has been pretokenized with '{args.tokenizer}'!")
        print(f"Location: {pretokenized_dir.absolute()}")
        print(f"\nDirectory structure:")
        print(f"  {pretokenized_dir}/")
        print(f"    ├── train.npy ({len(train_tokens):,} tokens, {num_train_docs:,} docs)")
        print(f"    ├── train_offsets.npy")
        print(f"    ├── validation.npy ({len(validation_tokens):,} tokens, {num_val_docs:,} docs)")
        print(f"    ├── validation_offsets.npy")
        print(f"    └── metadata.json")
        print()

    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except LoaderError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
