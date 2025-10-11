"""Tokenization utilities for pretokenizing datasets."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

from .exceptions import LoaderError


def load_tokenizer(tokenizer_name: str):
    """
    Load a tokenizer from a local file or HuggingFace Hub.

    Args:
        tokenizer_name: Name or path of the tokenizer
                       - Local file: "path/to/tokenizer.json"
                       - HuggingFace Hub: "gpt2", "meta-llama/Llama-2-7b-hf"

    Returns:
        Loaded tokenizer instance

    Raises:
        LoaderError: If tokenizer cannot be loaded
    """
    try:
        # Check if it's a local file path
        tokenizer_path = Path(tokenizer_name)
        if tokenizer_path.exists() and tokenizer_path.is_file():
            # Load from local file
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            return tokenizer

        # Otherwise, try loading from HuggingFace Hub
        tokenizer = Tokenizer.from_pretrained(tokenizer_name)
        return tokenizer
    except Exception as e:
        raise LoaderError(f"Failed to load tokenizer '{tokenizer_name}': {str(e)}")


def tokenize_jsonl(
    jsonl_path: Path,
    tokenizer,
    text_field: str = "text",
    add_eos: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tokenize all text from a JSONL file using the fast tokenizers library.

    Args:
        jsonl_path: Path to JSONL file
        tokenizer: HuggingFace tokenizer instance (from tokenizers library)
        text_field: Name of the text field in the JSON objects
        add_eos: Whether to add EOS token after each document

    Returns:
        Tuple of (tokens, offsets):
            - tokens: 1D numpy array of all token IDs
            - offsets: 1D numpy array of document start indices (length = num_docs + 1)
                      Each document's tokens are tokens[offsets[i]:offsets[i+1]]

    Raises:
        LoaderError: If file cannot be read or tokenized
    """
    if not jsonl_path.exists():
        raise LoaderError(f"JSONL file not found: {jsonl_path}")

    all_tokens = []
    offsets = [0]  # Track document boundaries (start index of each document)

    # Try to get EOS token ID if needed
    eos_token_id = None
    if add_eos:
        # Try common EOS token strings
        for eos_token in ["<|endoftext|>", "</s>", "<eos>", "[EOS]"]:
            token_id = tokenizer.token_to_id(eos_token)
            if token_id is not None:
                eos_token_id = token_id
                break

    try:
        # First pass: count lines for progress bar
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            num_lines = sum(1 for _ in f)

        # Second pass: tokenize
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=num_lines, desc=f"Tokenizing {jsonl_path.name}"):
                if not line.strip():
                    continue

                record = json.loads(line)

                if text_field not in record:
                    raise LoaderError(
                        f"Text field '{text_field}' not found in record. " +
                        f"Available fields: {list(record.keys())}"
                    )

                text = record[text_field]

                # Tokenize using the fast tokenizers library
                encoding = tokenizer.encode(text)
                tokens = encoding.ids

                # Add EOS token if requested and found
                if eos_token_id is not None:
                    tokens = list(tokens) + [eos_token_id]

                all_tokens.extend(tokens)
                # Track the end position of this document (= start of next document)
                offsets.append(len(all_tokens))

        # Convert to numpy array with appropriate dtype
        vocab_size = tokenizer.get_vocab_size()
        if vocab_size < 2**16:
            dtype = np.uint16
        else:
            dtype = np.uint32

        tokens_array = np.array(all_tokens, dtype=dtype)

        # Convert offsets to numpy array (use uint32 or uint64 based on total tokens)
        if len(all_tokens) < 2**32:
            offset_dtype = np.uint32
        else:
            offset_dtype = np.uint64
        offsets_array = np.array(offsets, dtype=offset_dtype)

        return tokens_array, offsets_array

    except json.JSONDecodeError as e:
        raise LoaderError(f"Invalid JSON in file '{jsonl_path}': {str(e)}")
    except Exception as e:
        raise LoaderError(f"Failed to tokenize file '{jsonl_path}': {str(e)}")


def save_tokens(tokens: np.ndarray, output_path: Path, compressed: bool = False) -> None:
    """
    Save tokenized data in NumPy format.

    Args:
        tokens: Numpy array of token IDs
        output_path: Path to save the file (.npy or .npz)
        compressed: If True, save as compressed NPZ format

    Raises:
        LoaderError: If file cannot be saved
    """
    try:
        # Create parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if compressed:
            # Save as compressed NPZ (slower to load but saves disk space)
            np.savez_compressed(output_path, tokens=tokens)
        else:
            # Save as NPY (fast, memory-mappable, includes metadata)
            np.save(output_path, tokens)

    except Exception as e:
        raise LoaderError(f"Failed to save tokens to '{output_path}': {str(e)}")


def save_document_tokens(
    tokens: np.ndarray,
    offsets: np.ndarray,
    tokens_path: Path,
    offsets_path: Path
) -> None:
    """
    Save tokenized data with document boundaries.

    Args:
        tokens: Numpy array of all token IDs (1D)
        offsets: Numpy array of document start indices (1D, length = num_docs + 1)
        tokens_path: Path to save tokens file (.npy)
        offsets_path: Path to save offsets file (.npy)

    Raises:
        LoaderError: If files cannot be saved
    """
    try:
        # Create parent directory if it doesn't exist
        tokens_path.parent.mkdir(parents=True, exist_ok=True)
        offsets_path.parent.mkdir(parents=True, exist_ok=True)

        # Save both arrays as NPY format (memory-mappable)
        np.save(tokens_path, tokens)
        np.save(offsets_path, offsets)

    except Exception as e:
        raise LoaderError(
            f"Failed to save document tokens to '{tokens_path}' and '{offsets_path}': {str(e)}"
        )


def save_metadata(
    metadata: Dict,
    output_path: Path
) -> None:
    """
    Save tokenization metadata as JSON.

    Args:
        metadata: Dictionary containing metadata
        output_path: Path to save the JSON file

    Raises:
        LoaderError: If file cannot be saved
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

    except Exception as e:
        raise LoaderError(f"Failed to save metadata to '{output_path}': {str(e)}")


def load_tokens(token_path: Path, mmap_mode: str = None) -> np.ndarray:
    """
    Load tokenized data from NumPy format.

    Args:
        token_path: Path to the token file (.npy or .npz)
        mmap_mode: Memory-map mode ('r' for read-only, 'r+' for read-write, None to load into memory)
                   Memory-mapping is efficient for large files during training.
                   Not supported for NPZ files.

    Returns:
        Numpy array of token IDs

    Raises:
        LoaderError: If file cannot be loaded
    """
    try:
        if not token_path.exists():
            raise LoaderError(f"Token file not found: {token_path}")

        # Check if it's a compressed NPZ file
        if token_path.suffix == '.npz':
            # NPZ files cannot be memory-mapped
            with np.load(token_path) as data:
                tokens = data['tokens']
            return tokens
        else:
            # NPY files can be memory-mapped for efficient loading
            tokens = np.load(token_path, mmap_mode=mmap_mode)
            return tokens

    except Exception as e:
        raise LoaderError(f"Failed to load tokens from '{token_path}': {str(e)}")


def load_document_tokens(
    tokens_path: Path,
    offsets_path: Path,
    doc_index: int = None,
    mmap_mode: str = None
) -> np.ndarray:
    """
    Load document tokens with optional document-level access.

    Args:
        tokens_path: Path to the tokens file (.npy)
        offsets_path: Path to the offsets file (.npy)
        doc_index: Optional document index to load. If None, returns all tokens.
                  If provided, returns tokens for that specific document.
        mmap_mode: Memory-map mode for tokens ('r' recommended for large datasets)

    Returns:
        Numpy array of token IDs (either all tokens or single document)

    Raises:
        LoaderError: If files cannot be loaded or doc_index is out of range

    Examples:
        # Load all tokens
        tokens = load_document_tokens(tokens_path, offsets_path)

        # Load specific document
        doc_5_tokens = load_document_tokens(tokens_path, offsets_path, doc_index=5)

        # Memory-map for efficient access
        tokens = load_document_tokens(tokens_path, offsets_path, mmap_mode='r')
    """
    try:
        if not tokens_path.exists():
            raise LoaderError(f"Tokens file not found: {tokens_path}")
        if not offsets_path.exists():
            raise LoaderError(f"Offsets file not found: {offsets_path}")

        # Load offsets (always load into memory, it's small)
        offsets = np.load(offsets_path)

        # Load tokens (optionally memory-mapped)
        tokens = np.load(tokens_path, mmap_mode=mmap_mode)

        # If doc_index specified, extract that document
        if doc_index is not None:
            num_docs = len(offsets) - 1
            if doc_index < 0 or doc_index >= num_docs:
                raise LoaderError(
                    f"Document index {doc_index} out of range [0, {num_docs})"
                )

            start = offsets[doc_index]
            end = offsets[doc_index + 1]
            return tokens[start:end]

        # Otherwise return all tokens
        return tokens

    except LoaderError:
        raise
    except Exception as e:
        raise LoaderError(
            f"Failed to load document tokens from '{tokens_path}' and '{offsets_path}': {str(e)}"
        )
