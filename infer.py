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

    # Load model weights with compatibility for checkpoints saved from torch.compile() models
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Print checkpoint info
    print(f"Checkpoint info:")
    print(f"  Batch: {checkpoint.get('batch_num', 'N/A')}")
    print(f"  Tokens seen: {checkpoint.get('tokens_seen', 'N/A'):,}")
    print(f"  Train loss: {checkpoint.get('train_loss', 'N/A'):.4f}" if checkpoint.get('train_loss') else "  Train loss: N/A")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}" if checkpoint.get('val_loss') else "  Val loss: N/A")

    return model


def _decode_token(tokenizer, token_id: int) -> str:
    """
    Decode a single token and replace BPE placeholder characters.

    Args:
        tokenizer: Tokenizer for decoding
        token_id: Token ID to decode

    Returns:
        Decoded string with BPE placeholders replaced
    """
    text = tokenizer.decode([token_id])
    # Replace BPE placeholder characters with their actual representations
    text = text.replace('Ġ', ' ')   # U+0120: space/word boundary
    text = text.replace('Ċ', '\n')  # U+010A: newline
    text = text.replace('Ĉ', '\n')  # U+0108: alternative newline marker
    text = text.replace('ĉ', '\n')  # U+0109: lowercase alternative
    text = text.replace('Č', '\r')  # U+010C: carriage return (rare)
    text = text.replace('č', '\r')  # U+010D: lowercase alternative
    return text


def generate_sample(
    model: GPT,
    tokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    device: Optional[torch.device] = None,
    eos_token_id: Optional[int] = None,
    stream: bool = False
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
        eos_token_id: If set, stop generation when this token is encountered
        stream: If True, print tokens as they are generated

    Returns:
        Tuple of (generated_text, token_count, generation_time_seconds)
    """
    if device is None:
        device = next(model.parameters()).device

    # Encode prompt
    encoding = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)

    # Generate with streaming
    start_time = time.time()
    model.eval()

    if stream:
        # Streaming generation - decode and print tokens one by one
        try:
            tokens = input_ids
            generated_tokens = []
            layer_caches = [None] * model.depth

            for step_idx in range(max_tokens):
                # Determine input tokens for this step
                if step_idx == 0:
                    # First iteration: process all input tokens
                    tokens_input = tokens if tokens.size(1) <= model.context_length else tokens[:, -model.context_length:]
                else:
                    # Subsequent iterations: only process the last generated token
                    tokens_input = tokens[:, -1:]

                # Forward pass with KV cache
                with torch.no_grad():
                    # Embed tokens
                    x = model.token_embedding(tokens_input)
                    x = model.emb_dropout(x)

                    # Pass through blocks with cache
                    new_caches = []
                    for block, cache in zip(model.blocks, layer_caches):
                        x, new_cache, aux_losses = block(x, attention_mask=None, kv_cache=cache, use_cache=True)
                        new_caches.append(new_cache)

                    layer_caches = new_caches

                    # Final norm and projection
                    x = model.norm_f(x)
                    logits = torch.nn.functional.linear(x, model.token_embedding.weight)

                # Get logits for last position
                logits = logits[:, -1, :].clone()

                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for token_id in set(tokens[0].tolist()):
                        if logits[0, token_id] < 0:
                            logits[0, token_id] *= repetition_penalty
                        else:
                            logits[0, token_id] /= repetition_penalty

                # Apply temperature
                logits = logits / temperature

                # Top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                # Top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices[0, sorted_indices_to_remove[0]]
                    logits[0, indices_to_remove] = float('-inf')

                # Sample
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                next_token_id = next_token[0, 0].item()

                # Check for EOS
                if eos_token_id is not None and next_token_id == eos_token_id:
                    break

                # Decode and print token immediately
                token_text = _decode_token(tokenizer, next_token_id)
                print(token_text, end='', flush=True)

                generated_tokens.append(next_token_id)
                tokens = torch.cat([tokens, next_token], dim=1)

            generation_time = time.time() - start_time

            # Reconstruct full text
            generated_text = ''.join([_decode_token(tokenizer, tid) for tid in generated_tokens])
            token_count = len(generated_tokens)

            return generated_text, token_count, generation_time

        except (RuntimeError, ValueError) as e:
            generation_time = time.time() - start_time
            error_msg = f"[Generation failed: {str(e)}]"
            print(error_msg)
            return error_msg, 0, generation_time

    else:
        # Non-streaming generation (original behavior)
        with torch.no_grad():
            try:
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=eos_token_id
                )
                generation_time = time.time() - start_time

                # Decode tokens without adding spaces between them
                output_tokens = output_ids[0].cpu().tolist()

                # Truncate at EOS token if present
                if eos_token_id is not None and eos_token_id in output_tokens:
                    eos_position = output_tokens.index(eos_token_id)
                    output_tokens = output_tokens[:eos_position]  # Exclude the EOS token itself

                token_count = len(output_tokens)

                # Decode each token individually and concatenate without spaces
                generated_text = ''.join([_decode_token(tokenizer, tid) for tid in output_tokens])

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
    verbose: bool = True,
    eos_token_id: Optional[int] = None,
    stream: bool = False
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
        eos_token_id: If set, stop generation when this token is encountered
        stream: If True, print tokens as they are generated

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

            if verbose and stream:
                print(f"\nSample {sample_idx + 1}/{num_samples_per_prompt}:")

            # Generate
            generated_text, token_count, generation_time = generate_sample(
                model, tokenizer, prompt, max_tokens,
                temperature, top_k, top_p, repetition_penalty, device, eos_token_id, stream
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
                if stream:
                    # Token-by-token output already printed, just add stats
                    print(f"\n[{token_count} tokens in {generation_time:.2f}s = {tokens_per_sec:.1f} tok/s]")
                else:
                    # Print all at once
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

  # Streaming output (see tokens as they're generated)
  python infer.py --checkpoint path/to/checkpoint.pt \\
                  --prompt "Once upon a time" --num-samples 1 --stream
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
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream tokens as they are generated (real-time output)"
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

    # Try to get EOS token ID
    eos_token_id = None
    try:
        # Try to encode the <|eos|> token
        eos_encoding = tokenizer.encode("<|eos|>")
        if len(eos_encoding.ids) == 1:
            eos_token_id = eos_encoding.ids[0]
            print(f"Found EOS token '<|eos|>' with ID: {eos_token_id}")
        else:
            print("Warning: '<|eos|>' encodes to multiple tokens, not using EOS stopping")
    except Exception as e:
        print(f"Warning: Could not encode '<|eos|>' token: {e}")

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
        verbose=not args.quiet,
        eos_token_id=eos_token_id,
        stream=args.stream
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
