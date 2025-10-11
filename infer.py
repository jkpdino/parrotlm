#!/usr/bin/env python3
"""
Inference script for GPT models.

Loads a trained checkpoint and generates multiple samples, saving results to CSV.

Usage:
    # Basic usage - uses config from checkpoint directory
    python infer.py --checkpoint training_runs/my_model/last_checkpoint.pt --prompt "Once upon a time" --num-samples 10

    # With custom config
    python infer.py --checkpoint path/to/checkpoint.pt --config path/to/config.yml --prompt "Hello" --num-samples 5

    # With advanced sampling parameters
    python infer.py --checkpoint path/to/checkpoint.pt --prompt "Once upon a time" \\
        --num-samples 20 --temperature 0.8 --top-k 50 --max-tokens 100

    # Multiple prompts from a file
    python infer.py --checkpoint path/to/checkpoint.pt --prompts-file prompts.txt --num-samples 3
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from src.tokenizer import load_tokenizer
from src.train.checkpoints import CheckpointManager
from src.train.config import TrainingRunConfig, load_training_config, ModelConfig
from src.train.model import GPT


def load_model_from_checkpoint(
    checkpoint_path: Path,
    config: TrainingRunConfig,
    device: Optional[torch.device] = None
) -> GPT:
    """
    Load a GPT model from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        config: Training run configuration
        device: Device to load model on

    Returns:
        Loaded GPT model in eval mode
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model on device: {device}")

    # Create model
    model = GPT(
        vocab_size=config.model.vocab_size,
        dim=config.model.dim,
        depth=config.model.depth,
        n_heads=config.model.n_heads,
        n_kv_heads=config.model.n_kv_heads,
        context_length=config.model.context_length,
        dropout=config.model.dropout,
        ffn_hidden_mult=config.model.ffn_hidden_mult,
        pad_token_id=config.model.pad_token_id,
        rope_scaling_factor=config.model.rope_scaling_factor,
        use_gradient_checkpointing=False,  # Never use during inference
        # MoE parameters
        use_moe=config.model.use_moe,
        num_experts=config.model.num_experts,
        num_experts_active=config.model.num_experts_active,
        moe_layers=config.model.moe_layers,
        router_aux_loss_coef=config.model.router_aux_loss_coef,
        router_z_loss_coef=config.model.router_z_loss_coef
    ).to(device)

    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Print checkpoint info
    print(f"Checkpoint info:")
    print(f"  Batch: {checkpoint.get('batch_num', 'N/A')}")
    print(f"  Tokens seen: {checkpoint.get('tokens_seen', 'N/A'):,}")
    print(f"  Train loss: {checkpoint.get('train_loss', 'N/A'):.4f}" if checkpoint.get('train_loss') else "  Train loss: N/A")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}" if checkpoint.get('val_loss') else "  Val loss: N/A")

    return model


def generate_sample(
    model: GPT,
    tokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    device: Optional[torch.device] = None
) -> tuple[str, int, float]:
    """
    Generate a single sample from the model.

    Args:
        model: The GPT model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Text prompt to start generation
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: If set, only sample from top-k tokens
        top_p: If set, use nucleus sampling
        repetition_penalty: Penalty for repeating tokens
        device: Device to run generation on

    Returns:
        Tuple of (generated_text, token_count, generation_time_seconds)
    """
    if device is None:
        device = next(model.parameters()).device

    # Encode prompt
    encoding = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)

    # Generate
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        try:
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            generation_time = time.time() - start_time

            # Decode
            generated_text = tokenizer.decode(output_ids[0].cpu().tolist())
            token_count = len(output_ids[0])

            return generated_text, token_count, generation_time

        except (RuntimeError, ValueError) as e:
            # Handle NaN/Inf errors during generation
            generation_time = time.time() - start_time
            error_msg = f"[Generation failed: {str(e)}]"
            return error_msg, 0, generation_time


def run_inference_batch(
    model: GPT,
    tokenizer,
    prompts: List[str],
    num_samples_per_prompt: int = 1,
    max_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> List[dict]:
    """
    Run inference on multiple prompts.

    Args:
        model: The GPT model
        tokenizer: Tokenizer for encoding/decoding
        prompts: List of prompts to generate from
        num_samples_per_prompt: Number of samples to generate per prompt
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: If set, only sample from top-k tokens
        top_p: If set, use nucleus sampling
        repetition_penalty: Penalty for repeating tokens
        device: Device to run generation on
        verbose: Whether to print progress

    Returns:
        List of dictionaries with inference results
    """
    results = []
    total_generations = len(prompts) * num_samples_per_prompt

    print(f"\nGenerating {total_generations} samples ({num_samples_per_prompt} per prompt)...")
    print(f"Parameters: temp={temperature}, top_k={top_k}, top_p={top_p}, max_tokens={max_tokens}")
    print("=" * 80)

    inference_num = 0
    for prompt_idx, prompt in enumerate(prompts, 1):
        if verbose:
            print(f"\nPrompt {prompt_idx}/{len(prompts)}: \"{prompt}\"")
            print("-" * 80)

        for sample_idx in range(num_samples_per_prompt):
            inference_num += 1

            # Generate
            generated_text, token_count, generation_time = generate_sample(
                model, tokenizer, prompt, max_tokens,
                temperature, top_k, top_p, repetition_penalty, device
            )

            # Calculate tokens per second
            tokens_per_sec = token_count / generation_time if generation_time > 0 else 0

            result = {
                'inference_num': inference_num,
                'prompt_idx': prompt_idx,
                'sample_idx': sample_idx + 1,
                'prompt': prompt,
                'generated_text': generated_text,
                'token_count': token_count,
                'generation_time': generation_time,
                'tokens_per_sec': tokens_per_sec,
                'timestamp': datetime.now().isoformat(),
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
                'max_tokens': max_tokens,
                'repetition_penalty': repetition_penalty
            }

            results.append(result)

            if verbose:
                print(f"\nSample {sample_idx + 1}/{num_samples_per_prompt}:")
                print(f"{generated_text}")
                print(f"[{token_count} tokens in {generation_time:.2f}s = {tokens_per_sec:.1f} tok/s]")

    return results


def save_results_to_csv(results: List[dict], output_path: Path) -> None:
    """
    Save inference results to a CSV file.

    Args:
        results: List of result dictionaries
        output_path: Path to save CSV file
    """
    if not results:
        print("Warning: No results to save")
        return

    # Create parent directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    fieldnames = [
        'inference_num', 'prompt_idx', 'sample_idx', 'prompt', 'generated_text',
        'token_count', 'generation_time', 'tokens_per_sec', 'timestamp',
        'temperature', 'top_k', 'top_p', 'max_tokens', 'repetition_penalty'
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_path}")


def load_prompts_from_file(prompts_file: Path) -> List[str]:
    """
    Load prompts from a text file (one prompt per line).

    Args:
        prompts_file: Path to prompts file

    Returns:
        List of prompts
    """
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    return prompts


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run inference on a trained GPT model and save results to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python infer.py --checkpoint training_runs/my_model/last_checkpoint.pt \\
                  --prompt "Once upon a time" --num-samples 10

  # Multiple prompts
  python infer.py --checkpoint path/to/checkpoint.pt \\
                  --prompts "Once upon a time" "In a land far away" "The brave knight" \\
                  --num-samples 5

  # Load prompts from file
  python infer.py --checkpoint path/to/checkpoint.pt \\
                  --prompts-file prompts.txt --num-samples 3

  # Advanced sampling
  python infer.py --checkpoint path/to/checkpoint.pt \\
                  --prompt "Hello world" --num-samples 20 \\
                  --temperature 0.8 --top-k 50 --top-p 0.9 --max-tokens 100
        """
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)"
    )

    # Prompt arguments (mutually exclusive)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to generate from"
    )
    prompt_group.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        help="Multiple prompts to generate from"
    )
    prompt_group.add_argument(
        "--prompts-file",
        type=str,
        help="Path to file containing prompts (one per line)"
    )

    # Generation parameters
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate per prompt (default: 1)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate (default: 50)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (higher = more random, default: 1.0)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling (default: None)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling (default: None)"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (>1.0 discourages repetition, default: 1.0)"
    )

    # Config and output
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config YAML file (if not provided, looks in checkpoint directory)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file path (default: inferences_<timestamp>.csv)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to run inference on (default: auto)"
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

    # Load or find config
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)
    else:
        # Look for config.yml in checkpoint directory
        config_path = checkpoint_path.parent / "config.yml"
        if not config_path.exists():
            print(f"Error: Config file not found at {config_path}", file=sys.stderr)
            print("Please specify --config explicitly", file=sys.stderr)
            sys.exit(1)

    print(f"Loading config from: {config_path}")
    config = load_training_config(config_path)

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load model
    print(f"\nLoading model from checkpoint...")
    model = load_model_from_checkpoint(checkpoint_path, config, device)

    # Load tokenizer
    print(f"\nLoading tokenizer from: {config.data.tokenizer_path}")
    tokenizer = load_tokenizer(config.data.tokenizer_path)

    # Prepare prompts
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts:
        prompts = args.prompts
    else:  # args.prompts_file
        prompts_file = Path(args.prompts_file)
        prompts = load_prompts_from_file(prompts_file)
        print(f"\nLoaded {len(prompts)} prompts from: {prompts_file}")

    # Run inference
    results = run_inference_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        num_samples_per_prompt=args.num_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=device,
        verbose=not args.quiet
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        # Default output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"inferences_{timestamp}.csv")

    save_results_to_csv(results, output_path)

    # Print summary
    print("\n" + "=" * 80)
    print("Inference Summary:")
    print(f"  Total generations: {len(results)}")
    print(f"  Total prompts: {len(prompts)}")
    print(f"  Samples per prompt: {args.num_samples}")

    if results:
        avg_tokens = sum(r['token_count'] for r in results) / len(results)
        avg_time = sum(r['generation_time'] for r in results) / len(results)
        avg_speed = sum(r['tokens_per_sec'] for r in results) / len(results)

        print(f"  Average tokens per generation: {avg_tokens:.1f}")
        print(f"  Average generation time: {avg_time:.2f}s")
        print(f"  Average speed: {avg_speed:.1f} tokens/sec")

    print("=" * 80)


if __name__ == "__main__":
    main()
