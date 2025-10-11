"""Document-aware batch loader for pretokenized datasets with attention masking."""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Union
import random
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.tokenizer import load_tokenizer

import torch


class DocumentBatchLoader:
    """
    PyTorch batch loader that packs multiple documents into context windows with proper attention masks.

    Features:
    - Packs multiple short documents into single context windows
    - Creates attention masks preventing cross-document attention (additive format: 0/-inf)
    - Returns PyTorch tensors ready for training
    - Automatic label generation for next-token prediction
    - Memory-efficient with optional memory mapping
    - GPU placement support

    Example:
        loader = DocumentBatchLoader(
            tokens_path="datasets/tinystories/pretokenized/tokenizer/train.npy",
            offsets_path="datasets/tinystories/pretokenized/tokenizer/train_offsets.npy",
            context_length=512,
            batch_size=8,
            return_labels=True,
            device="cuda"
        )

        for batch_tokens, batch_masks, batch_labels in loader:
            # batch_tokens: (8, 512) - PyTorch tensor of token IDs
            # batch_masks: (8, 512, 512) - PyTorch tensor, additive masks (0/-inf)
            # batch_labels: (8, 512) - PyTorch tensor for next-token prediction
            # All tensors already on GPU if device="cuda"
    """

    def __init__(
        self,
        tokens_path: str,
        offsets_path: str,
        context_length: int = 512,
        batch_size: int = 8,
        pad_token_id: int = 0,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        shuffle: bool = False,
        drop_last: bool = False,
        mmap_mode: Optional[str] = 'r',
        max_docs_per_window: Optional[int] = None,
        return_labels: bool = False,
        label_ignore_index: int = 0,
        device: Optional[Union[str, torch.device]] = None,
        return_masks: bool = True
    ):
        """
        Initialize the document batch loader.

        Args:
            tokens_path: Path to the tokens .npy file
            offsets_path: Path to the offsets .npy file
            context_length: Maximum sequence length for each sample
            batch_size: Number of samples per batch
            pad_token_id: Token ID to use for padding
            bos_token_id: Token ID for beginning of sequence (wraps each document, optional)
            eos_token_id: Token ID for end of sequence (wraps each document, optional)
            shuffle: Whether to shuffle documents before batching
            drop_last: Whether to drop the last incomplete batch
            mmap_mode: Memory map mode ('r' for read-only, None to load into memory)
            max_docs_per_window: Maximum number of documents to pack per window (None = unlimited)
            return_labels: Whether to return labels for next-token prediction
            label_ignore_index: Value to use for positions to ignore in loss (0 for PyTorch)
            device: Device to place tensors on (e.g., "cuda", "cpu", or torch.device)
            return_masks: Whether to return attention masks (set to False to save memory if not using document boundaries)
        """
        self.tokens_path = Path(tokens_path)
        self.offsets_path = Path(offsets_path)
        self.context_length = context_length
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.max_docs_per_window = max_docs_per_window
        self.return_labels = return_labels
        self.label_ignore_index = label_ignore_index
        self.device = device
        self.return_masks = return_masks

        # Validate paths
        if not self.tokens_path.exists():
            raise FileNotFoundError(f"Tokens file not found: {self.tokens_path}")
        if not self.offsets_path.exists():
            raise FileNotFoundError(f"Offsets file not found: {self.offsets_path}")

        # Load data
        self.tokens = np.load(self.tokens_path, mmap_mode=mmap_mode)
        self.offsets = np.load(self.offsets_path)

        self.num_documents = len(self.offsets) - 1

        # Pre-pack documents into context windows
        self.packed_windows = self._pack_documents()
        self.num_windows = len(self.packed_windows)

        # Calculate number of batches
        if self.drop_last:
            self.num_batches = self.num_windows // self.batch_size
        else:
            self.num_batches = (self.num_windows + self.batch_size - 1) // self.batch_size

        # Initialize iteration
        self.current_batch = 0
        self.window_indices = list(range(self.num_windows))
        if self.shuffle:
            random.shuffle(self.window_indices)

    def _pack_documents(self) -> List[List[int]]:
        """
        Pack documents into context windows greedily.

        Returns:
            List of packed windows, where each window is a list of document indices
        """
        packed_windows = []
        current_window = []
        current_length = 0

        # Calculate overhead from special tokens
        special_token_overhead = 0
        if self.bos_token_id is not None:
            special_token_overhead += 1
        if self.eos_token_id is not None:
            special_token_overhead += 1

        for doc_idx in range(self.num_documents):
            doc_start = self.offsets[doc_idx]
            doc_end = self.offsets[doc_idx + 1]
            doc_length = doc_end - doc_start

            # Add special token overhead to get effective length
            effective_doc_length = doc_length + special_token_overhead

            # Skip empty documents
            if doc_length == 0:
                continue

            # If document is too long, truncate it to fit in one window
            if effective_doc_length > self.context_length:
                # Create a window with just this truncated document
                if current_window:
                    packed_windows.append(current_window)
                    current_window = []
                    current_length = 0
                packed_windows.append([doc_idx])
                continue

            # Check if adding this document would exceed context length
            if current_length + effective_doc_length > self.context_length:
                # Start a new window
                packed_windows.append(current_window)
                current_window = [doc_idx]
                current_length = effective_doc_length
            else:
                # Check max_docs_per_window constraint
                if self.max_docs_per_window is not None and len(current_window) >= self.max_docs_per_window:
                    packed_windows.append(current_window)
                    current_window = [doc_idx]
                    current_length = effective_doc_length
                else:
                    # Add to current window
                    current_window.append(doc_idx)
                    current_length += effective_doc_length

        # Add the last window if not empty
        if current_window:
            packed_windows.append(current_window)

        return packed_windows

    def _create_sample_from_window(self, doc_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a single sample (tokens and mask) from a packed window of documents.

        Args:
            doc_indices: List of document indices to pack into this window

        Returns:
            Tuple of (tokens, attention_mask)
            - tokens: (context_length,) array of token IDs
            - attention_mask: (context_length, context_length) boolean mask
        """
        # Initialize output arrays
        tokens = np.full(self.context_length, self.pad_token_id, dtype=self.tokens.dtype)

        # Create document ID array to track which document each token belongs to
        # -1 means padding, 0+ means document index within this window
        doc_ids = np.full(self.context_length, -1, dtype=np.int32)

        # Fill tokens and document IDs
        current_pos = 0
        for local_doc_idx, global_doc_idx in enumerate(doc_indices):
            doc_start = self.offsets[global_doc_idx]
            doc_end = self.offsets[global_doc_idx + 1]
            doc_length = doc_end - doc_start

            # Calculate how much space we have left
            remaining_space = self.context_length - current_pos

            # Calculate total space needed for this document (with special tokens)
            total_needed = doc_length
            if self.bos_token_id is not None:
                total_needed += 1
            if self.eos_token_id is not None:
                total_needed += 1

            if total_needed > remaining_space:
                # Not enough space for this document, stop packing
                break

            # Add BOS token if specified
            if self.bos_token_id is not None:
                tokens[current_pos] = self.bos_token_id
                doc_ids[current_pos] = local_doc_idx
                current_pos += 1

            # Copy document tokens
            tokens[current_pos:current_pos + doc_length] = self.tokens[doc_start:doc_start + doc_length]
            doc_ids[current_pos:current_pos + doc_length] = local_doc_idx
            current_pos += doc_length

            # Add EOS token if specified
            if self.eos_token_id is not None:
                tokens[current_pos] = self.eos_token_id
                doc_ids[current_pos] = local_doc_idx
                current_pos += 1

            if current_pos >= self.context_length:
                break

        # Create attention mask
        # Shape: (context_length, context_length)
        # mask[i, j] = True if token i can attend to token j
        attention_mask = self._create_attention_mask(doc_ids)

        return tokens, attention_mask

    def _create_attention_mask(self, doc_ids: np.ndarray) -> np.ndarray:
        """
        Create attention mask that enforces document boundaries.

        Creates square blocks where tokens within the same document have value 1.
        The causal (triangular) mask should be applied separately during training.

        Args:
            doc_ids: (seq_len,) array where each element is document ID (-1 for padding)

        Returns:
            (seq_len, seq_len) boolean mask where True = same document, False = different documents
        """
        seq_len = len(doc_ids)

        # Create document boundary mask (square blocks for each document)
        # mask[i, j] = True if doc_ids[i] == doc_ids[j] and both are not padding
        doc_ids_expanded = doc_ids[:, None]  # (seq_len, 1)
        attention_mask = (doc_ids_expanded == doc_ids[None, :]) & (doc_ids_expanded >= 0)

        return attention_mask

    def __len__(self) -> int:
        """Return the number of batches."""
        return self.num_batches

    def reset(self):
        """
        Reset the iterator for a new epoch.

        Call this at the start of each epoch to reshuffle the data (if shuffle=True)
        and reset the batch counter.
        """
        self.current_batch = 0
        self.window_indices = list(range(self.num_windows))
        if self.shuffle:
            random.shuffle(self.window_indices)

    def __iter__(self):
        """Reset iterator."""
        self.reset()
        return self

    def __next__(self):
        """
        Get the next batch.

        Returns:
            Tuple of (batch_tokens, batch_masks) or (batch_tokens, batch_masks, batch_labels)
            depending on return_labels setting.
            - batch_tokens: (batch_size, context_length) token IDs
            - batch_masks: (batch_size, context_length, context_length) attention masks
            - batch_labels: (batch_size, context_length) labels for next-token prediction (if return_labels=True)
        """
        if self.current_batch >= self.num_batches:
            raise StopIteration

        # Get window indices for this batch
        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_windows)
        batch_window_indices = self.window_indices[start_idx:end_idx]

        # Create batch
        batch_tokens = []
        batch_masks = [] if self.return_masks else None

        for window_idx in batch_window_indices:
            doc_indices = self.packed_windows[window_idx]
            tokens, mask = self._create_sample_from_window(doc_indices)
            batch_tokens.append(tokens)
            if self.return_masks:
                batch_masks.append(mask)

        # Convert to arrays
        batch_tokens = np.array(batch_tokens)
        if self.return_masks:
            batch_masks = np.array(batch_masks)

        # Pad last batch if needed and not dropping
        if len(batch_tokens) < self.batch_size and not self.drop_last:
            padding_size = self.batch_size - len(batch_tokens)

            # Create padding samples
            pad_tokens = np.full(
                (padding_size, self.context_length),
                self.pad_token_id,
                dtype=self.tokens.dtype
            )

            batch_tokens = np.vstack([batch_tokens, pad_tokens])

            if self.return_masks:
                pad_masks = np.zeros(
                    (padding_size, self.context_length, self.context_length),
                    dtype=bool
                )
                batch_masks = np.vstack([batch_masks, pad_masks])

        # Generate labels if requested (shifted tokens for next-token prediction)
        batch_labels = None
        if self.return_labels:
            batch_labels = np.roll(batch_tokens, -1, axis=1)
            # Set last token to ignore index (can't predict beyond sequence)
            batch_labels[:, -1] = self.label_ignore_index
            # Set padding positions to ignore index
            padding_mask = batch_tokens == self.pad_token_id
            batch_labels[padding_mask] = self.label_ignore_index

        # Convert to PyTorch tensors
        batch_tokens = torch.from_numpy(batch_tokens).long()

        # Convert masks to additive format (0.0 for attend, -inf for don't attend)
        if self.return_masks:
            batch_masks = torch.from_numpy(batch_masks.astype(np.float32))
            batch_masks = torch.where(batch_masks == 1.0, 0.0, float('-inf'))

        if batch_labels is not None:
            batch_labels = torch.from_numpy(batch_labels).long()

        # Move to device if specified
        if self.device is not None:
            batch_tokens = batch_tokens.to(self.device)
            if self.return_masks:
                batch_masks = batch_masks.to(self.device)
            if batch_labels is not None:
                batch_labels = batch_labels.to(self.device)

        self.current_batch += 1

        if self.return_labels:
            if self.return_masks:
                return batch_tokens, batch_masks, batch_labels
            else:
                return batch_tokens, None, batch_labels
        else:
            if self.return_masks:
                return batch_tokens, batch_masks
            else:
                return batch_tokens, None

    def get_stats(self) -> dict:
        """
        Get statistics about the packed dataset.

        Returns:
            Dictionary with statistics
        """
        docs_per_window = [len(window) for window in self.packed_windows]
        tokens_per_window = []

        for window in self.packed_windows:
            total_tokens = 0
            for doc_idx in window:
                doc_start = self.offsets[doc_idx]
                doc_end = self.offsets[doc_idx + 1]
                total_tokens += min(doc_end - doc_start, self.context_length)
            tokens_per_window.append(total_tokens)

        packing_efficiency = [t / self.context_length for t in tokens_per_window]

        return {
            "total_documents": self.num_documents,
            "total_windows": self.num_windows,
            "total_batches": self.num_batches,
            "context_length": self.context_length,
            "batch_size": self.batch_size,
            "avg_docs_per_window": np.mean(docs_per_window),
            "avg_tokens_per_window": np.mean(tokens_per_window),
            "avg_packing_efficiency": np.mean(packing_efficiency),
            "min_docs_per_window": np.min(docs_per_window),
            "max_docs_per_window": np.max(docs_per_window),
        }


def visualize_attention_mask(mask, max_size: int = 50):
    """
    Visualize an attention mask in ASCII, downsampled to fit the display.

    Args:
        mask: (seq_len, seq_len) attention mask (PyTorch tensor or numpy array)
        max_size: Maximum size to display (for readability)
    """
    # Convert PyTorch tensor to numpy if needed
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Convert additive mask (0/-inf) to boolean (True/False) for visualization
    # In additive format: 0.0 = can attend, -inf = cannot attend
    if mask.dtype in [np.float32, np.float64]:
        mask = ~np.isinf(mask)  # True where not -inf (i.e., can attend)

    seq_len = mask.shape[0]

    if seq_len <= max_size:
        # No downsampling needed
        display_mask = mask
        display_size = seq_len
    else:
        # Downsample using max pooling to compress the entire mask
        # Calculate the pooling factor
        pool_factor = int(np.ceil(seq_len / max_size))
        display_size = max_size

        # Create downsampled mask
        display_mask = np.zeros((display_size, display_size), dtype=bool)

        for i in range(display_size):
            for j in range(display_size):
                # Calculate the region in the original mask
                i_start = i * pool_factor
                i_end = min((i + 1) * pool_factor, seq_len)
                j_start = j * pool_factor
                j_end = min((j + 1) * pool_factor, seq_len)

                # Max pooling: if any element in the region is True, set to True
                display_mask[i, j] = np.any(mask[i_start:i_end, j_start:j_end])

    print(f"Attention Mask Visualization ({seq_len}x{seq_len} compressed to {display_size}x{display_size}):")
    print("(■ = can attend, □ = cannot attend)\n")

    for i in range(display_size):
        row_str = ""
        for j in range(display_size):
            row_str += "■ " if display_mask[i, j] else "□ "
        print(row_str)
    print()


if __name__ == "__main__":
    # Example usage
    print("Document-Aware Batch Loader Demo\n")
    print("=" * 60)

    # Configure paths (update these to your actual paths)
    tokens_path = "datasets/tinystories/pretokenized/tokenizer/train.npy"
    offsets_path = "datasets/tinystories/pretokenized/tokenizer/train_offsets.npy"

    # Check if files exist
    if not Path(tokens_path).exists() or not Path(offsets_path).exists():
        print(f"ERROR: Dataset files not found!")
        print(f"Please ensure you have pretokenized data at:")
        print(f"  - {tokens_path}")
        print(f"  - {offsets_path}")
        print(f"\nRun pretokenize.py first to generate these files.")
    else:
        # Load tokenizer to get special token IDs
        try:
            tokenizer = load_tokenizer("tokenizers/tinystories_tokenizer/tokenizer.json")

            # Get special token IDs
            # For a custom tokenizer, you might have defined these as special tokens
            bos_token_id = tokenizer.token_to_id("<|bos|>")
            eos_token_id = tokenizer.token_to_id("<|eos|>")
            pad_token_id = tokenizer.token_to_id("<|pad|>")

            # Fallback to standard values if not found
            if bos_token_id is None:
                print("Warning: BOS token '<|bos|>' not found, using None")
                bos_token_id = None
            if eos_token_id is None:
                print("Warning: EOS token '<|eos|>' not found, using None")
                eos_token_id = None
            if pad_token_id is None:
                print("Warning: PAD token '<|pad|>' not found, using 0")
                pad_token_id = 0

            print(f"Special tokens: BOS={bos_token_id}, EOS={eos_token_id}, PAD={pad_token_id}\n")

        except Exception as e:
            print(f"Could not load tokenizer: {e}")
            print("Using default token IDs\n")
            bos_token_id = None
            eos_token_id = None
            pad_token_id = 0

        # Create loader
        loader = DocumentBatchLoader(
            tokens_path=tokens_path,
            offsets_path=offsets_path,
            context_length=512,
            batch_size=4,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            shuffle=False
        )

        # Print statistics
        stats = loader.get_stats()
        print("\nDataset Statistics:")
        print(f"  Total documents: {stats['total_documents']:,}")
        print(f"  Total windows: {stats['total_windows']:,}")
        print(f"  Total batches: {stats['total_batches']:,}")
        print(f"  Context length: {stats['context_length']}")
        print(f"  Batch size: {stats['batch_size']}")
        print(f"  Avg docs per window: {stats['avg_docs_per_window']:.2f}")
        print(f"  Avg tokens per window: {stats['avg_tokens_per_window']:.1f}")
        print(f"  Packing efficiency: {stats['avg_packing_efficiency']*100:.1f}%")
        print(f"  Docs per window range: [{stats['min_docs_per_window']}, {stats['max_docs_per_window']}]")

        # Get first batch
        print("\n" + "=" * 60)
        print("First Batch Sample:")
        print("=" * 60 + "\n")

        batch_tokens, batch_masks = next(iter(loader))

        print(f"Batch tokens shape: {batch_tokens.shape}")
        print(f"Batch masks shape: {batch_masks.shape}")
        print(f"Batch tokens dtype: {batch_tokens.dtype}")
        print(f"Batch masks dtype: {batch_masks.dtype}")

        # Show first sample from batch
        print(f"\nFirst sample tokens (first 50):")
        print(batch_tokens[0, :50])

        # Load tokenizer and decode the text
        print("\n" + "=" * 60)
        print("Decoding tokens to text:")
        print("=" * 60 + "\n")

        try:
            # Tokenizer already loaded above
            # Get the first sample
            sample_tokens = batch_tokens[0]

            # Find document boundaries by looking at the attention mask
            # Documents are indicated by the diagonal blocks in the mask
            # In additive format: 0.0 = can attend, -inf = cannot attend
            doc_starts = [0]
            for i in range(1, len(sample_tokens)):
                # Check if token i is in a different document than token i-1
                # This happens when mask[i-1, i] is -inf (cannot attend)
                if torch.isinf(batch_masks[0, i-1, i]) and sample_tokens[i] != loader.pad_token_id:
                    doc_starts.append(i)

            # Find where padding starts
            padding_start = len(sample_tokens)
            for i, token in enumerate(sample_tokens):
                if token == loader.pad_token_id:
                    padding_start = i
                    break

            # Add final boundary
            if padding_start not in doc_starts:
                doc_starts.append(padding_start)

            # Decode each document separately
            print(f"Found {len(doc_starts) - 1} documents in this sample:\n")

            for doc_idx in range(len(doc_starts) - 1):
                start = doc_starts[doc_idx]
                end = doc_starts[doc_idx + 1]

                doc_tokens = sample_tokens[start:end].tolist()

                # Check for BOS/EOS in the token sequence
                has_bos = (loader.bos_token_id is not None and
                          len(doc_tokens) > 0 and
                          doc_tokens[0] == loader.bos_token_id)
                has_eos = (loader.eos_token_id is not None and
                          len(doc_tokens) > 0 and
                          doc_tokens[-1] == loader.eos_token_id)

                print(f"Document {doc_idx + 1} (tokens {start}-{end}, length={end-start}):")
                print(f"  Token IDs: {doc_tokens[:10]}{'...' if len(doc_tokens) > 10 else ''}")

                if has_bos or has_eos:
                    print(f"  Special tokens: BOS={has_bos}, EOS={has_eos}")

                # Decode using the tokenizer
                decoded_text = tokenizer.decode(doc_tokens)

                print(f"  Text: {decoded_text}")
                print()

            if padding_start < len(sample_tokens):
                print(f"Padding: {len(sample_tokens) - padding_start} tokens")

        except Exception as e:
            print(f"Could not decode tokens: {e}")
            print("(This is okay - tokenizer might not be available)")

        print("\n" + "=" * 60)
        print("First sample attention mask:")
        print("=" * 60)
        visualize_attention_mask(batch_masks[0], max_size=40)

        # Iterate through a few batches
        print("=" * 60)
        print("Iterating through batches:")
        print("=" * 60 + "\n")

        for i, (tokens, masks) in enumerate(loader):
            if i >= 3:  # Just show first 3 batches
                break
            print(f"Batch {i+1}: tokens shape = {tokens.shape}, masks shape = {masks.shape}")

        print(f"\n... and {len(loader) - 3} more batches")

        # PyTorch example with labels
        print("\n" + "=" * 60)
        print("PyTorch with Labels Example:")
        print("=" * 60 + "\n")

        # Create loader with labels for training
        train_loader = DocumentBatchLoader(
            tokens_path=tokens_path,
            offsets_path=offsets_path,
            context_length=512,
            batch_size=4,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            shuffle=False,
            return_labels=True,   # Return labels for training
            device=None  # Set to 'cuda' for GPU
        )

        # Get one batch
        tokens, masks, labels = next(iter(train_loader))

        print(f"Tokens shape: {tokens.shape}, dtype: {tokens.dtype}")
        print(f"Masks shape: {masks.shape}, dtype: {masks.dtype}")
        print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
        print(f"\nMask format: additive (0.0 = attend, -inf = don't attend)")
        print(f"Labels: shifted tokens with {train_loader.label_ignore_index} for ignore positions")

        print("\nSample usage in training loop:")
        print("""
# Training loop example
for epoch in range(num_epochs):
    loader.reset()  # Reshuffle data for new epoch

    for tokens, masks, labels in loader:
        # tokens, masks, labels are already PyTorch tensors
        # If device='cuda' was set, they're already on GPU

        outputs = model(tokens, attention_mask=masks)
        loss = criterion(
            outputs.view(-1, vocab_size),
            labels.view(-1)
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        """)
