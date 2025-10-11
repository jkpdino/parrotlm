#!/usr/bin/env python3
"""
Validation loss testing script.

Calculates validation loss on a checkpoint to compare tokenizer compatibility.

Usage:
    # Test validation loss with current tokenizer
    python test_loss.py --checkpoint training_runs/my_model/last_checkpoint.pt

    # Test with custom tokenizer
    python test_loss.py --checkpoint training_runs/my_model/last_checkpoint.pt \\
                        --tokenizer path/to/tokenizer.json

    # Test with more validation samples
    python test_loss.py --checkpoint training_runs/my_model/last_checkpoint.pt \\
                        --num-batches 100

    # Compare against checkpoint's recorded validation loss
    python test_loss.py --checkpoint training_runs/my_model/checkpoint_500.pt --compare
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tokenizer import load_tokenizer
from src.train.config import load_training_config
from src.train.model import GPT
from src.train.pipeline import DocumentBatchLoader


def load_model_from_checkpoint(
    checkpoint_path: Path,
    config,
    device: Optional[torch.device] = None
) -> tuple[GPT, dict]:
    """
    Load a GPT model from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        config: Training run configuration
        device: Device to load model on

    Returns:
        Tuple of (model, checkpoint_data)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = GPT(
        vocab_size=config.model.vocab_size,
        dim=config.model.dim,
        depth=config.model.depth,
        n_heads=config.model.n_heads,
        n_kv_heads=config.model.n_kv_heads,
        context_length=config.model.context_length,
        dropout=0.0,  # No dropout during eval
        ffn_hidden_mult=config.model.ffn_hidden_mult,
        pad_token_id=config.model.pad_token_id,
        rope_scaling_factor=config.model.rope_scaling_factor,
        use_gradient_checkpointing=False,
        # MoE parameters
        use_moe=config.model.use_moe,
        num_experts=config.model.num_experts,
        num_experts_active=config.model.num_experts_active,
        moe_layers=config.model.moe_layers,
        router_aux_loss_coef=config.model.router_aux_loss_coef,
        router_z_loss_coef=config.model.router_z_loss_coef
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


def compute_validation_loss(
    model: GPT,
    val_loader: DocumentBatchLoader,
    num_batches: Optional[int] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> dict:
    """
    Compute validation loss on a dataset.

    Args:
        model: The GPT model
        val_loader: Validation data loader
        num_batches: Number of batches to evaluate (None = all available)
        device: Device to run on
        verbose: Whether to print progress

    Returns:
        Dictionary with validation metrics
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batch_losses = []
    num_batches_processed = 0

    if verbose:
        print("\nComputing validation loss...")

    with torch.no_grad():
        try:
            for batch_idx, (tokens, masks, labels) in enumerate(val_loader):
                if num_batches is not None and batch_idx >= num_batches:
                    break

                # Forward pass
                result = model(tokens, masks)

                # Handle MoE models (return tuple) vs standard models (return tensor)
                if isinstance(result, tuple):
                    logits, aux_losses = result
                else:
                    logits = result
                    aux_losses = None

                # Compute loss
                loss = F.cross_entropy(
                    logits.view(-1, model.vocab_size),
                    labels.view(-1),
                    ignore_index=0,
                    reduction='sum'
                )

                # Add auxiliary losses if using MoE
                if aux_losses is not None:
                    for aux_loss_name, aux_loss_value in aux_losses.items():
                        loss = loss + aux_loss_value

                # Count non-padding tokens
                num_tokens = (labels != 0).sum().item()

                total_loss += loss.item()
                total_tokens += num_tokens
                batch_losses.append(loss.item() / num_tokens if num_tokens > 0 else 0)
                num_batches_processed += 1

                if verbose and (batch_idx + 1) % 10 == 0:
                    avg_loss_so_far = total_loss / total_tokens if total_tokens > 0 else 0
                    print(f"  Batch {batch_idx + 1}/{num_batches or '?'}: loss = {avg_loss_so_far:.4f}")

        except StopIteration:
            pass

    # Calculate metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'total_tokens': total_tokens,
        'num_batches': num_batches_processed,
        'batch_losses': batch_losses
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test validation loss on a trained model checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python test_loss.py --checkpoint training_runs/my_model/last_checkpoint.pt

  # Test with custom tokenizer (to see impact of tokenizer mismatch)
  python test_loss.py --checkpoint training_runs/my_model/last_checkpoint.pt \\
                      --tokenizer tokenizers/new_tokenizer/tokenizer.json

  # More thorough evaluation
  python test_loss.py --checkpoint training_runs/my_model/last_checkpoint.pt \\
                      --num-batches 200

  # Compare against checkpoint's recorded loss
  python test_loss.py --checkpoint training_runs/my_model/checkpoint_500.pt --compare
        """
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config YAML file (if not in checkpoint directory)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Path to tokenizer (overrides config). Useful for testing tokenizer mismatch."
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=50,
        help="Number of validation batches to evaluate (default: 50)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare against the validation loss recorded in the checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to run on (default: auto)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Validate checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    # Load config
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = checkpoint_path.parent / "config.yml"

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        print("Please specify --config explicitly", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"Loading config from: {config_path}")

    config = load_training_config(config_path)

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if not args.quiet:
        print(f"Using device: {device}")

    # Load model and checkpoint
    if not args.quiet:
        print(f"\nLoading model from: {checkpoint_path}")

    model, checkpoint = load_model_from_checkpoint(checkpoint_path, config, device)

    if not args.quiet:
        print(f"✓ Model loaded")
        print(f"  Checkpoint batch: {checkpoint.get('batch_num', 'N/A')}")
        print(f"  Tokens seen: {checkpoint.get('tokens_seen', 'N/A'):,}")
        if checkpoint.get('val_loss'):
            print(f"  Recorded validation loss: {checkpoint['val_loss']:.4f}")

    # Load tokenizer
    if args.tokenizer:
        tokenizer_path = args.tokenizer
        if not args.quiet:
            print(f"\nUsing custom tokenizer: {tokenizer_path}")
    else:
        tokenizer_path = config.data.tokenizer_path
        if not args.quiet:
            print(f"\nLoading tokenizer from config: {tokenizer_path}")

    tokenizer = load_tokenizer(tokenizer_path)

    # Get special token IDs
    pad_token_id = config.model.pad_token_id
    bos_token_id = tokenizer.token_to_id("<|bos|>")
    eos_token_id = tokenizer.token_to_id("<|eos|>")

    # Create validation loader
    dataset_path = Path(config.data.dataset_path)
    batch_size = args.batch_size if args.batch_size else config.data.batch_size

    if not args.quiet:
        print(f"\nLoading validation data from: {dataset_path}")
        print(f"Batch size: {batch_size}")

    val_loader = DocumentBatchLoader(
        tokens_path=str(dataset_path / "validation.npy"),
        offsets_path=str(dataset_path / "validation_offsets.npy"),
        context_length=config.model.context_length,
        batch_size=batch_size,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        shuffle=False,
        return_labels=True,
        device=device,
        return_masks=False
    )

    # Compute validation loss
    print("\n" + "=" * 80)
    results = compute_validation_loss(
        model, val_loader, args.num_batches, device, verbose=not args.quiet
    )

    # Print results
    print("=" * 80)
    print("\nVALIDATION RESULTS:")
    print("-" * 80)
    print(f"  Validation loss:  {results['loss']:.4f}")
    print(f"  Perplexity:       {results['perplexity']:.2f}")
    print(f"  Tokens evaluated: {results['total_tokens']:,}")
    print(f"  Batches:          {results['num_batches']}")

    # Compare if requested
    if args.compare and checkpoint.get('val_loss') is not None:
        recorded_loss = checkpoint['val_loss']
        current_loss = results['loss']
        difference = current_loss - recorded_loss
        percent_increase = (difference / recorded_loss) * 100 if recorded_loss > 0 else 0

        print("\nCOMPARISON:")
        print("-" * 80)
        print(f"  Recorded loss (in checkpoint): {recorded_loss:.4f}")
        print(f"  Current loss:                   {current_loss:.4f}")
        print(f"  Difference:                     {difference:+.4f}")
        print(f"  Percent change:                 {percent_increase:+.2f}%")

        if abs(percent_increase) < 1:
            print("\n  ✓ Loss is very close - tokenizer appears compatible!")
        elif abs(percent_increase) < 5:
            print("\n  ⚠ Small difference - tokenizer might be slightly different")
        elif abs(percent_increase) < 20:
            print("\n  ⚠ Moderate difference - tokenizer mismatch likely")
        else:
            print("\n  ✗ Large difference - significant tokenizer mismatch!")

    print("=" * 80)


if __name__ == "__main__":
    main()
