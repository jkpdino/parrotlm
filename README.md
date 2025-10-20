# Datasets

Datasets is a utility to create Datasets from various sources. It then creates train and validation splits for the datasets. The utility also provides a way to preprocess the data.

## Features

- Load datasets from HuggingFace Hub or local files (JSONL, JSON, CSV, Parquet)
- Combine multiple dataset slices into one unified dataset
- Apply sampling strategies (random or sequential)
- Limit and offset support for each slice
- Automatic train/validation splitting
- Reproducible results with random seeds
- Progress tracking during downloads
- Dataset statistics generation
- **Train custom tokenizers** using BPE, WordPiece, or Unigram algorithms
- **Pretokenize datasets** for faster training with NumPy format

## Installation

This project uses [UV](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies
uv sync
```

## Dataset Configuration

Datasets are composed of a list of slices. Each slice represents a subset of the dataset. Each slice can have a HuggingFace dataset or a local dataset. Slices are configured using a YAML file.

### Configuration Schema

```yaml
key: unique_dataset_id # Unique identifier for this dataset
name: dataset_name # Human-readable name

# Optional: Split ratios (must sum to 1.0)
train_split: 0.8 # Default: 0.8
validation_split: 0.2 # Default: 0.2

# Optional: Random seed for reproducibility
random_seed: 42 # Default: 42

slices:
  - key: slice_identifier # Unique identifier for this slice
    huggingface: "dataset:split" # HuggingFace dataset (format: "name:split")
    # OR
    local: "/path/to/file" # Local file (JSONL, JSON, CSV, or Parquet)

    # Optional parameters:
    limit: 1000 # Maximum number of records to load
    offset: 100 # Skip first N records
    sampling: random # "random" or "sequential" (default: sequential)
```

### Example Configuration

See `configs/example_dataset.yml` for a comprehensive example:

```yaml
key: example_dataset_v1
name: example_dataset

train_split: 0.8
validation_split: 0.2
random_seed: 42

slices:
  # Load from HuggingFace
  - key: hf_sentiment
    huggingface: "imdb:train"
    limit: 1000
    sampling: random

  # Load from local file
  - key: local_data
    local: "/path/to/data.jsonl"
    limit: 500
    sampling: sequential
```

## Usage

### Basic Usage

```bash
# Using UV
uv run download.py configs/example_dataset.yml

# Or with Python directly
python download.py configs/example_dataset.yml
```

### Custom Output Directory

```bash
uv run download.py configs/example_dataset.yml --output-dir my_datasets
```

### Quick Test

Try the simple example configuration:

```bash
uv run download.py configs/simple_example.yml
```

## Output Structure

Datasets are stored in a directory structure:

```
datasets/
  dataset_name/
    train/
      data.jsonl              # Training data
    validation/
      data.jsonl              # Validation data
    stats.json                # Dataset statistics
    pretokenized/             # Pretokenized data (optional)
      tokenizer_name/
        train.npy             # Tokenized training data (NumPy format)
        train_offsets.npy     # Document boundary offsets for training
        validation.npy        # Tokenized validation data (NumPy format)
        validation_offsets.npy # Document boundary offsets for validation
        metadata.json         # Tokenization metadata
```

## Supported Data Sources

### HuggingFace Datasets

Format: `"dataset_name:split"`

Examples:

- `"imdb:train"` - IMDB sentiment dataset, training split
- `"yelp_review_full:test"` - Yelp reviews, test split
- `"squad:validation"` - SQuAD dataset, validation split

### Local Files

Supported formats:

- **JSONL** (.jsonl) - JSON Lines format (recommended)
- **JSON** (.json) - Standard JSON (array of objects or single object)
- **CSV** (.csv) - Comma-separated values
- **Parquet** (.parquet, .pq) - Apache Parquet format

## Pretokenization

Pretokenization converts text to token IDs ahead of time, significantly speeding up training by eliminating tokenization overhead during the training loop.

### Benefits

- **Faster Training**: No tokenization overhead during training
- **Reduced I/O**: Binary format is more efficient than JSONL
- **Memory Efficient**: Tokens stored as uint16/uint32 arrays
- **Reproducible**: Same tokens every time
- **Multiple Tokenizers**: Support different tokenizers for the same dataset

### Basic Usage

```bash
# Using UV
uv run pretokenize.py configs/tinystories_100k.yml --tokenizer gpt2

# Or with Python directly
python pretokenize.py configs/tinystories_100k.yml --tokenizer gpt2
```

### Tokenizer Options

You can use any HuggingFace tokenizer:

```bash
# GPT-2 tokenizer
uv run pretokenize.py configs/my_dataset.yml --tokenizer gpt2

# LLaMA tokenizer
uv run pretokenize.py configs/my_dataset.yml --tokenizer meta-llama/Llama-2-7b-hf

# Custom tokenizer
uv run pretokenize.py configs/my_dataset.yml --tokenizer /path/to/tokenizer
```

### Advanced Options

```bash
# Specify text field name (default: "text")
uv run pretokenize.py configs/my_dataset.yml --tokenizer gpt2 --text-field content

# Don't add EOS token after each document
uv run pretokenize.py configs/my_dataset.yml --tokenizer gpt2 --no-eos

# Force retokenization even if files exist
uv run pretokenize.py configs/my_dataset.yml --tokenizer gpt2 --force

# Custom output directory
uv run pretokenize.py configs/my_dataset.yml --tokenizer gpt2 --output-dir my_datasets
```

### Output Format

The pretokenization process creates:

1. **train.npy** - NumPy array file containing all token IDs for training data (1D array)
2. **train_offsets.npy** - NumPy array of document start indices (1D array, length = num_docs + 1)
3. **validation.npy** - NumPy array file containing all token IDs for validation data (1D array)
4. **validation_offsets.npy** - NumPy array of document start indices (1D array, length = num_docs + 1)
5. **metadata.json** - Metadata about the tokenization:
   - Tokenizer name and vocabulary size
   - Number of tokens and documents in each split
   - Text field name used
   - Data types for tokens and offsets
   - Creation timestamp

The offset arrays preserve document boundaries, allowing you to:

- Extract individual documents: `tokens[offsets[i]:offsets[i+1]]`
- Batch by complete documents
- Avoid mixing tokens from different documents
- Track which tokens belong to which document

The NPY format includes dtype and shape metadata, making it self-describing and easy to use. Files can also be memory-mapped for efficient training on large datasets.

### Loading Pretokenized Data

You can load the pretokenized data for training:

```python
import numpy as np
from pathlib import Path
from src.tokenizer import load_document_tokens

# Method 1: Using the helper function to load all tokens
tokens = load_document_tokens(
    Path('datasets/my_dataset/pretokenized/gpt2/train.npy'),
    Path('datasets/my_dataset/pretokenized/gpt2/train_offsets.npy')
)

# Method 2: Load a specific document by index
doc_5_tokens = load_document_tokens(
    Path('datasets/my_dataset/pretokenized/gpt2/train.npy'),
    Path('datasets/my_dataset/pretokenized/gpt2/train_offsets.npy'),
    doc_index=5
)

# Method 3: Memory-map for efficient access (recommended for large datasets)
tokens = load_document_tokens(
    Path('datasets/my_dataset/pretokenized/gpt2/train.npy'),
    Path('datasets/my_dataset/pretokenized/gpt2/train_offsets.npy'),
    mmap_mode='r'
)

# Method 4: Manual loading with NumPy
train_tokens = np.load('datasets/my_dataset/pretokenized/gpt2/train.npy', mmap_mode='r')
train_offsets = np.load('datasets/my_dataset/pretokenized/gpt2/train_offsets.npy')

# Extract individual documents
doc_0 = train_tokens[train_offsets[0]:train_offsets[1]]
doc_1 = train_tokens[train_offsets[1]:train_offsets[2]]

# Iterate through all documents
num_docs = len(train_offsets) - 1
for i in range(num_docs):
    doc_tokens = train_tokens[train_offsets[i]:train_offsets[i+1]]
    # Process document...

# Use with PyTorch DataLoader, etc.
```

### Example Workflow

```bash
# 1. Download and process dataset
uv run download.py configs/tinystories_100k.yml

# 2. Pretokenize with GPT-2
uv run pretokenize.py configs/tinystories_100k.yml --tokenizer gpt2

# 3. Pretokenize with different tokenizer (optional)
uv run pretokenize.py configs/tinystories_100k.yml --tokenizer meta-llama/Llama-2-7b-hf
```

This creates:

```
datasets/
  tinystories_100k/
    train/data.jsonl
    validation/data.jsonl
    stats.json
    pretokenized/
      gpt2/
        train.npy
        train_offsets.npy
        validation.npy
        validation_offsets.npy
        metadata.json
      Llama-2-7b-hf/
        train.npy
        train_offsets.npy
        validation.npy
        validation_offsets.npy
        metadata.json
```

## Training Custom Tokenizers

You can train custom tokenizers on your datasets using BPE (Byte-Pair Encoding), WordPiece, or Unigram algorithms. This is useful for creating domain-specific tokenizers optimized for your text data.

### Why Train Custom Tokenizers?

- **Better compression**: Tokenizers trained on your specific domain can achieve better compression ratios
- **Domain-specific vocabulary**: Capture terminology and patterns specific to your data
- **Control over vocabulary size**: Choose the optimal vocabulary size for your model
- **Special token customization**: Define custom special tokens for your use case

### Tokenizer Types

- **BPE (Byte-Pair Encoding)**: Used by GPT-2, GPT-3, RoBERTa, LLaMA. Iteratively merges most common token pairs.
- **WordPiece**: Used by BERT, DistilBERT. Similar to BPE but uses likelihood-based merging.
- **Unigram**: Used by T5, ALBERT. Probabilistic approach starting with large vocabulary and pruning.

### Basic Usage

```bash
# Train from an existing dataset
uv run train_tokenizer.py --dataset tinystories_100k --name tinystories_bpe --vocab-size 32768

# Train from text files
uv run train_tokenizer.py --files data/*.txt --name custom_tokenizer --vocab-size 30000

# Train from YAML config
uv run train_tokenizer.py --config configs/example_tokenizer.yml
```

### Command-Line Options

```bash
# Input sources (choose one)
--dataset DATASET              # Train from existing dataset
--files FILE [FILE ...]        # Train from text/JSONL files
--config CONFIG                # Train from YAML config

# Required (unless using --config)
--name NAME                    # Name for the trained tokenizer

# Tokenizer configuration
--type {bpe,wordpiece,unigram} # Tokenizer type (default: bpe)
--vocab-size SIZE              # Target vocabulary size (default: 30000)
--min-frequency FREQ           # Minimum token frequency (default: 0)
--special-tokens TOKEN [...]   # Special tokens to add

# Text processing
--normalizer TYPE              # Text normalization: nfc, nfd, lowercase, etc.
--pre-tokenizer TYPE           # Pre-tokenization: whitespace, byte_level, etc.
--text-field FIELD             # Text field in JSONL (default: "text")

# Dataset options
--split SPLIT                  # Dataset split to use (default: "train")
--datasets-dir DIR             # Datasets directory (default: "datasets")

# Output
--output-dir DIR               # Output directory (default: "tokenizers")
```

### YAML Configuration

Create a YAML configuration file for reproducible tokenizer training:

```yaml
# configs/my_tokenizer.yml
name: my_custom_tokenizer
type: bpe
vocab_size: 32768
min_frequency: 2

# Special tokens
special_tokens:
  - "<|pad|>"
  - "<|unk|>"
  - "<|bos|>"
  - "<|eos|>"

# Input source
dataset: tinystories_100k
split: train
text_field: text

# Text processing
normalizer: nfc
pre_tokenizer: whitespace

# Output
output_dir: tokenizers
```

Then train:

```bash
uv run train_tokenizer.py --config configs/my_tokenizer.yml
```

### Output Structure

Trained tokenizers are saved in the following structure:

```
tokenizers/
  tokenizer_name/
    tokenizer.json      # Full tokenizer configuration (loadable with Tokenizer.from_file())
    metadata.json       # Training metadata and statistics
    config.yml          # Copy of training config (if trained from YAML)
```

### Using Trained Tokenizers

Once trained, you can use your custom tokenizer with `pretokenize.py`:

```bash
# Use custom tokenizer for pretokenization
uv run pretokenize.py configs/dataset.yml --tokenizer tokenizers/my_tokenizer/tokenizer.json
```

You can also load it programmatically:

```python
from tokenizers import Tokenizer

# Load trained tokenizer
tokenizer = Tokenizer.from_file("tokenizers/my_tokenizer/tokenizer.json")

# Tokenize text
encoding = tokenizer.encode("Hello, world!")
print(encoding.tokens)  # ['Hello', ',', 'world', '!']
print(encoding.ids)     # [245, 11, 234, 8]
```

### Example Workflow

```bash
# 1. Download dataset
uv run download.py configs/tinystories_100k.yml

# 2. Train custom tokenizer
uv run train_tokenizer.py --dataset tinystories_100k --name tinystories_bpe_8k --vocab-size 32768

# 3. Pretokenize dataset with custom tokenizer
uv run pretokenize.py configs/tinystories_100k.yml --tokenizer tokenizers/tinystories_bpe_8k/tokenizer.json
```

This creates:

```
tokenizers/
  tinystories_bpe_8k/
    tokenizer.json
    metadata.json

datasets/
  tinystories_100k/
    train/data.jsonl
    validation/data.jsonl
    stats.json
    pretokenized/
      tinystories_bpe_8k/
        train.npy
        train_offsets.npy
        validation.npy
        validation_offsets.npy
        metadata.json
```

### Advanced Options

**Text Normalization:**

- `nfc` - Unicode NFC normalization (default)
- `nfd` - Unicode NFD normalization
- `lowercase` - Convert to lowercase
- `nfc_lowercase` - NFC + lowercase
- `none` - No normalization

**Pre-tokenization:**

- `whitespace` - Split on whitespace (default)
- `byte_level` - Byte-level pre-tokenization (like GPT-2)
- `digits` - Split individual digits
- `whitespace_digits` - Whitespace + split digits

**Special Tokens Examples:**

```bash
# GPT-style
--special-tokens "<|endoftext|>" "<|pad|>"

# BERT-style
--special-tokens "[PAD]" "[UNK]" "[CLS]" "[SEP]" "[MASK]"

# Custom
--special-tokens "<|pad|>" "<|unk|>" "<|bos|>" "<|eos|>" "<|user|>" "<|assistant|>"
```

## Project Structure

```
.
├── download.py              # Dataset download and processing script
├── pretokenize.py           # Pretokenization script
├── train_tokenizer.py       # Tokenizer training script
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration parsing and validation
│   ├── exceptions.py       # Custom exceptions
│   ├── tokenizer.py        # Tokenization utilities
│   ├── tokenizer_training.py  # Tokenizer training utilities
│   ├── processor.py        # Train/val splitting and output
│   └── loaders/            # Data loader modules (SOLID architecture)
│       ├── __init__.py
│       ├── base.py         # Abstract base loader
│       ├── factory.py      # Loader factory
│       ├── sampling.py     # Sampling strategies
│       ├── huggingface_loader.py
│       └── local_loaders.py
├── configs/                # Example configurations
│   ├── example_dataset.yml
│   ├── example_tokenizer.yml
│   ├── simple_example.yml
│   └── tinystories_100k.yml
├── pyproject.toml          # Project dependencies
└── README.md
```

## Development

### Adding Custom Preprocessing

You can extend the `processor.py` module to add custom preprocessing functions:

```python
from src.processor import preprocess_data

def my_preprocessing_fn(record):
    # Your custom logic here
    record['text'] = record['text'].lower()
    return record

# Use in your workflow
processed_data = preprocess_data(data, my_preprocessing_fn)
```

## Dependencies

- Python 3.9+
- PyYAML - YAML configuration parsing
- datasets - HuggingFace datasets library
- pandas - Data manipulation
- tqdm - Progress bars
- tokenizers - Fast Rust-based tokenization library (for pretokenization)
- numpy - Numerical operations (for pretokenization)

## License

MIT
