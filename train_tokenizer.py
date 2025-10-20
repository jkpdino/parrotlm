#!/usr/bin/env python3
"""
Tokenizer training script.

This script trains custom tokenizers using BPE, WordPiece, or Unigram algorithms.

Usage:
    # Train from dataset
    python train_tokenizer.py --dataset tinystories_100k --name tinystories_bpe

    # Train from files
    python train_tokenizer.py --files data/*.txt --name custom_tokenizer

    # Train from config
    python train_tokenizer.py --config configs/my_tokenizer.yml

Example:
    python train_tokenizer.py --dataset tinystories_100k --name tiny_bpe_8k --vocab-size 32768
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import List

from src.exceptions import ConfigError, LoaderError
from src.tokenizer_training import (
    configure_normalizer,
    configure_pre_tokenizer,
    create_tokenizer_model,
    create_trainer,
    extract_text_from_jsonl,
    get_vocabulary_stats,
    save_training_metadata,
    show_tokenization_samples,
)


def load_config_file(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise ConfigError(f"Failed to load config file '{config_path}': {str(e)}")


def get_sample_texts(file_paths: List[Path], num_samples: int = 5) -> List[str]:
    """Get sample texts for verification after training."""
    samples = []

    for file_path in file_paths:
        if file_path.suffix == '.jsonl':
            # Read from JSONL
            try:
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= num_samples:
                            break
                        if line.strip():
                            record = json.loads(line)
                            # Try common text field names
                            for field in ['text', 'content', 'body']:
                                if field in record:
                                    samples.append(record[field])
                                    break
            except Exception:
                pass
        else:
            # Read from text file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= num_samples:
                            break
                        if line.strip():
                            samples.append(line.strip())
            except Exception:
                pass

        if len(samples) >= num_samples:
            break

    return samples[:num_samples]


def main():
    """Main entry point for the tokenizer training script."""
    parser = argparse.ArgumentParser(
        description="Train custom tokenizers using BPE, WordPiece, or Unigram algorithms"
    )

    # Input sources
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--dataset",
        type=str,
        help="Dataset name (reads from datasets/<name>/train/data.jsonl)"
    )
    input_group.add_argument(
        "--files",
        type=str,
        nargs="+",
        help="Text or JSONL files to train on"
    )
    input_group.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file"
    )

    # Required arguments (unless using config)
    parser.add_argument(
        "--name",
        type=str,
        help="Name for the trained tokenizer (required unless using --config)"
    )

    # Tokenizer configuration
    parser.add_argument(
        "--type",
        type=str,
        default="bpe",
        choices=["bpe", "wordpiece", "unigram"],
        help="Tokenizer type (default: bpe)"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=30000,
        help="Target vocabulary size (default: 30000)"
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=0,
        help="Minimum frequency for tokens (default: 0)"
    )
    parser.add_argument(
        "--special-tokens",
        type=str,
        nargs="+",
        help="Special tokens to add (e.g., '<pad>' '<unk>' '<s>' '</s>')"
    )

    # Text processing
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Text field name in JSONL files (default: text)"
    )
    parser.add_argument(
        "--normalizer",
        type=str,
        default="nfc",
        choices=["nfc", "nfd", "nfkc", "nfkd", "lowercase", "nfc_lowercase", "none"],
        help="Text normalization strategy (default: nfc)"
    )
    parser.add_argument(
        "--pre-tokenizer",
        type=str,
        default="whitespace",
        choices=["whitespace", "byte_level", "digits", "whitespace_digits"],
        help="Pre-tokenization strategy (default: whitespace)"
    )

    # Dataset options
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default="datasets",
        help="Base directory for datasets (default: datasets)"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tokenizers",
        help="Output directory for trained tokenizers (default: tokenizers)"
    )

    args = parser.parse_args()

    try:
        # Load configuration from file if provided
        if args.config:
            print(f"\n{'='*60}")
            print(f"Loading configuration from: {args.config}")
            print(f"{'='*60}\n")

            config = load_config_file(args.config)

            # Extract configuration
            name = config.get("name")
            tokenizer_type = config.get("type", "bpe")
            vocab_size = config.get("vocab_size", 30000)
            min_frequency = config.get("min_frequency", 0)
            special_tokens = config.get("special_tokens", None)
            text_field = config.get("text_field", "text")
            normalizer = config.get("normalizer", "nfc")
            pre_tokenizer = config.get("pre_tokenizer", "whitespace")

            # Determine input source
            if "dataset" in config:
                dataset_name = config["dataset"]
                split = config.get("split", "train")
                datasets_dir = config.get("datasets_dir", "datasets")
                dataset_path = Path(datasets_dir) / dataset_name / split / "data.jsonl"
                input_files = [dataset_path]
            elif "files" in config:
                input_files = [Path(f) for f in config["files"]]
            else:
                raise ConfigError("Config must specify either 'dataset' or 'files'")

            output_dir = Path(config.get("output_dir", "tokenizers"))
        else:
            # Use command-line arguments
            if not args.name:
                print("Error: --name is required when not using --config", file=sys.stderr)
                sys.exit(1)

            name = args.name
            tokenizer_type = args.type
            vocab_size = args.vocab_size
            min_frequency = args.min_frequency
            special_tokens = args.special_tokens
            text_field = args.text_field
            normalizer = args.normalizer
            pre_tokenizer = args.pre_tokenizer
            output_dir = Path(args.output_dir)

            # Determine input source
            if args.dataset:
                dataset_path = Path(args.datasets_dir) / args.dataset / args.split / "data.jsonl"
                input_files = [dataset_path]
            else:
                input_files = [Path(f) for f in args.files]

        # Validate input files
        print(f"\n{'='*60}")
        print("VALIDATING INPUT FILES")
        print(f"{'='*60}\n")

        for file_path in input_files:
            if not file_path.exists():
                print(f"Error: Input file not found: {file_path}", file=sys.stderr)
                sys.exit(1)
            print(f"  ✓ {file_path}")

        print()

        # Print configuration
        print(f"{'='*60}")
        print("TRAINING CONFIGURATION")
        print(f"{'='*60}\n")

        print(f"Tokenizer name: {name}")
        print(f"Tokenizer type: {tokenizer_type}")
        print(f"Vocabulary size: {vocab_size:,}")
        print(f"Min frequency: {min_frequency}")
        print(f"Special tokens: {special_tokens or 'None'}")
        print(f"Normalizer: {normalizer}")
        print(f"Pre-tokenizer: {pre_tokenizer}")
        print(f"Text field: {text_field}")
        print()

        # Create output directory
        tokenizer_dir = output_dir / name
        tokenizer_dir.mkdir(parents=True, exist_ok=True)

        # Create tokenizer model
        print(f"{'='*60}")
        print("CREATING TOKENIZER MODEL")
        print(f"{'='*60}\n")

        tokenizer = create_tokenizer_model(tokenizer_type)
        print(f"Created {tokenizer_type.upper()} tokenizer model")

        # Configure normalizer
        configure_normalizer(tokenizer, normalizer)
        print(f"Configured normalizer: {normalizer}")

        # Configure pre-tokenizer
        configure_pre_tokenizer(tokenizer, pre_tokenizer)
        print(f"Configured pre-tokenizer: {pre_tokenizer}")
        print()

        # Create trainer
        print(f"{'='*60}")
        print("TRAINING TOKENIZER")
        print(f"{'='*60}\n")

        trainer = create_trainer(
            tokenizer_type,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens or [],
        )

        # Train the tokenizer
        # For JSONL files, we need to use train_from_iterator
        # For text files, we can use train with file paths
        all_jsonl = all(f.suffix == '.jsonl' for f in input_files)

        if all_jsonl:
            # Use iterator for JSONL files
            def text_iterator():
                for file_path in input_files:
                    yield from extract_text_from_jsonl(file_path, text_field)

            tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
        else:
            # Use file paths for text files
            tokenizer.train([str(f) for f in input_files], trainer=trainer)

        print(f"\n✓ Training complete!")
        print()

        # Get vocabulary statistics
        stats = get_vocabulary_stats(tokenizer)

        print(f"{'='*60}")
        print("VOCABULARY STATISTICS")
        print(f"{'='*60}\n")

        print(f"Vocabulary size: {stats['vocab_size']:,}")
        print(f"Total tokens: {stats['total_tokens']:,}")
        print()

        # Save tokenizer
        print(f"{'='*60}")
        print("SAVING TOKENIZER")
        print(f"{'='*60}\n")

        tokenizer_path = tokenizer_dir / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))
        print(f"✓ Saved tokenizer to {tokenizer_path}")

        # Save training configuration
        training_config = {
            "tokenizer_type": tokenizer_type,
            "vocab_size": vocab_size,
            "min_frequency": min_frequency,
            "special_tokens": special_tokens,
            "normalizer": normalizer,
            "pre_tokenizer": pre_tokenizer,
            "text_field": text_field,
            "input_files": [str(f) for f in input_files],
        }

        metadata_path = tokenizer_dir / "metadata.json"
        save_training_metadata(
            metadata_path,
            tokenizer_type,
            vocab_size,
            training_config,
            stats
        )
        print(f"✓ Saved metadata to {metadata_path}")

        # Save config for reproducibility
        if args.config:
            config_copy_path = tokenizer_dir / "config.yml"
            with open(config_copy_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"✓ Saved config to {config_copy_path}")

        print()

        # Show tokenization samples
        sample_texts = get_sample_texts(input_files)
        if sample_texts:
            show_tokenization_samples(tokenizer, sample_texts)

        # Success summary
        print(f"{'='*60}")
        print("SUCCESS")
        print(f"{'='*60}\n")

        print(f"Tokenizer '{name}' has been trained successfully!")
        print(f"Location: {tokenizer_dir.absolute()}")
        print(f"\nDirectory structure:")
        print(f"  {tokenizer_dir}/")
        print(f"    ├── tokenizer.json")
        print(f"    └── metadata.json")
        if args.config:
            print(f"    └── config.yml")
        print()
        print(f"To use with pretokenization:")
        print(f"  uv run pretokenize.py <config> --tokenizer {tokenizer_path.absolute()}")
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
