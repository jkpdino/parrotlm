"""Tokenizer training utilities for creating custom tokenizers."""

import json
from pathlib import Path
from typing import Iterator, List, Optional

from tokenizers import (
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)
from tqdm import tqdm

from .exceptions import LoaderError


def create_tokenizer_model(tokenizer_type: str) -> Tokenizer:
    """
    Create a tokenizer with the specified model type.

    Args:
        tokenizer_type: Type of tokenizer ("bpe", "wordpiece", or "unigram")

    Returns:
        Tokenizer instance with the specified model

    Raises:
        LoaderError: If tokenizer_type is invalid
    """
    tokenizer_type = tokenizer_type.lower()

    if tokenizer_type == "bpe":
        model = models.BPE()
    elif tokenizer_type == "wordpiece":
        model = models.WordPiece()
    elif tokenizer_type == "unigram":
        model = models.Unigram()
    else:
        raise LoaderError(
            f"Invalid tokenizer type '{tokenizer_type}'. "
            f"Must be one of: bpe, wordpiece, unigram"
        )

    return Tokenizer(model)


def create_trainer(
    tokenizer_type: str,
    vocab_size: int = 30000,
    min_frequency: int = 0,
    special_tokens: Optional[List[str]] = None,
    **kwargs
) -> trainers.Trainer:
    """
    Create a trainer for the specified tokenizer type.

    Args:
        tokenizer_type: Type of tokenizer ("bpe", "wordpiece", or "unigram")
        vocab_size: Target vocabulary size
        min_frequency: Minimum frequency for tokens
        special_tokens: List of special tokens to add
        **kwargs: Additional trainer-specific arguments

    Returns:
        Trainer instance

    Raises:
        LoaderError: If tokenizer_type is invalid
    """
    tokenizer_type = tokenizer_type.lower()

    if special_tokens is None:
        special_tokens = []

    if tokenizer_type == "bpe":
        return trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=kwargs.get("show_progress", True),
        )
    elif tokenizer_type == "wordpiece":
        return trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            continuing_subword_prefix=kwargs.get("continuing_subword_prefix", "##"),
        )
    elif tokenizer_type == "unigram":
        return trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            show_progress=kwargs.get("show_progress", True),
        )
    else:
        raise LoaderError(
            f"Invalid tokenizer type '{tokenizer_type}'. "
            f"Must be one of: bpe, wordpiece, unigram"
        )


def configure_normalizer(tokenizer: Tokenizer, normalizer_type: Optional[str] = "nfc"):
    """
    Configure text normalization for the tokenizer.

    Args:
        tokenizer: Tokenizer to configure
        normalizer_type: Type of normalization ("nfc", "nfd", "nfkc", "nfkd",
                        "lowercase", "nfc_lowercase", or None)
    """
    if normalizer_type is None or normalizer_type == "none":
        return

    normalizer_type = normalizer_type.lower()

    if normalizer_type == "nfc":
        tokenizer.normalizer = normalizers.NFC()
    elif normalizer_type == "nfd":
        tokenizer.normalizer = normalizers.NFD()
    elif normalizer_type == "nfkc":
        tokenizer.normalizer = normalizers.NFKC()
    elif normalizer_type == "nfkd":
        tokenizer.normalizer = normalizers.NFKD()
    elif normalizer_type == "lowercase":
        tokenizer.normalizer = normalizers.Lowercase()
    elif normalizer_type == "nfc_lowercase":
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFC(),
            normalizers.Lowercase(),
        ])
    else:
        raise LoaderError(
            f"Invalid normalizer type '{normalizer_type}'. "
            f"Must be one of: nfc, nfd, nfkc, nfkd, lowercase, nfc_lowercase, none"
        )


def configure_pre_tokenizer(tokenizer: Tokenizer, pre_tokenizer_type: str = "whitespace"):
    """
    Configure pre-tokenization (splitting strategy) for the tokenizer.

    Args:
        tokenizer: Tokenizer to configure
        pre_tokenizer_type: Type of pre-tokenization ("whitespace", "byte_level",
                           "digits", or "whitespace_digits")
    """
    pre_tokenizer_type = pre_tokenizer_type.lower()

    if pre_tokenizer_type == "whitespace":
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    elif pre_tokenizer_type == "byte_level":
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    elif pre_tokenizer_type == "digits":
        tokenizer.pre_tokenizer = pre_tokenizers.Digits()
    elif pre_tokenizer_type == "whitespace_digits":
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Whitespace(),
            pre_tokenizers.Digits(individual_digits=True),
        ])
    else:
        raise LoaderError(
            f"Invalid pre-tokenizer type '{pre_tokenizer_type}'. "
            f"Must be one of: whitespace, byte_level, digits, whitespace_digits"
        )


def extract_text_from_jsonl(
    jsonl_path: Path,
    text_field: str = "text"
) -> Iterator[str]:
    """
    Extract text from a JSONL file for tokenizer training.

    Args:
        jsonl_path: Path to JSONL file
        text_field: Name of the text field in JSON objects

    Yields:
        Text strings from the JSONL file

    Raises:
        LoaderError: If file cannot be read
    """
    if not jsonl_path.exists():
        raise LoaderError(f"JSONL file not found: {jsonl_path}")

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                record = json.loads(line)

                if text_field not in record:
                    raise LoaderError(
                        f"Text field '{text_field}' not found in record. "
                        f"Available fields: {list(record.keys())}"
                    )

                yield record[text_field]

    except json.JSONDecodeError as e:
        raise LoaderError(f"Invalid JSON in file '{jsonl_path}': {str(e)}")
    except Exception as e:
        raise LoaderError(f"Failed to read file '{jsonl_path}': {str(e)}")


def show_tokenization_samples(tokenizer: Tokenizer, texts: List[str], num_samples: int = 5):
    """
    Display tokenization samples for verification.

    Args:
        tokenizer: Trained tokenizer
        texts: Sample texts to tokenize
        num_samples: Number of samples to show
    """
    print(f"\n{'='*60}")
    print("TOKENIZATION SAMPLES")
    print(f"{'='*60}\n")

    for i, text in enumerate(texts[:num_samples], 1):
        # Truncate long texts for display
        display_text = text[:100] + "..." if len(text) > 100 else text

        encoding = tokenizer.encode(text)
        tokens = encoding.tokens
        token_ids = encoding.ids

        print(f"Sample {i}:")
        print(f"  Text: {display_text}")
        print(f"  Tokens ({len(tokens)}): {tokens[:20]}")
        if len(tokens) > 20:
            print(f"  ... ({len(tokens) - 20} more tokens)")
        print(f"  IDs: {token_ids[:20]}")
        if len(token_ids) > 20:
            print(f"  ... ({len(token_ids) - 20} more IDs)")
        print()


def get_vocabulary_stats(tokenizer: Tokenizer) -> dict:
    """
    Get statistics about the trained vocabulary.

    Args:
        tokenizer: Trained tokenizer

    Returns:
        Dictionary with vocabulary statistics
    """
    vocab = tokenizer.get_vocab()

    stats = {
        "vocab_size": len(vocab),
        "total_tokens": tokenizer.get_vocab_size(),
    }

    return stats


def save_training_metadata(
    output_path: Path,
    tokenizer_type: str,
    vocab_size: int,
    training_config: dict,
    stats: dict
):
    """
    Save tokenizer training metadata.

    Args:
        output_path: Path to save metadata JSON
        tokenizer_type: Type of tokenizer
        vocab_size: Target vocabulary size
        training_config: Full training configuration
        stats: Vocabulary statistics
    """
    from datetime import datetime

    metadata = {
        "tokenizer_type": tokenizer_type,
        "vocab_size": vocab_size,
        "actual_vocab_size": stats["vocab_size"],
        "created_at": datetime.now().isoformat(),
        "training_config": training_config,
        "stats": stats,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
