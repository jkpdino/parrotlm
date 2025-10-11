"""Sample generation utilities for monitoring training progress."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from tokenizers import Tokenizer

from ..tokenizer import load_tokenizer


def generate_sample(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    device: Optional[torch.device] = None
) -> str:
    """
    Generate text from the model given a prompt.

    Args:
        model: The GPT model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Text prompt to start generation
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: If set, only sample from top-k tokens
        top_p: If set, use nucleus sampling
        device: Device to run generation on

    Returns:
        Generated text (including prompt)
    """
    if device is None:
        device = next(model.parameters()).device

    # Encode prompt
    encoding = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)

    # Generate
    model.eval()
    with torch.no_grad():
        try:
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            # Decode
            generated_text = tokenizer.decode(output_ids[0].cpu().tolist())
            return generated_text
        except (RuntimeError, ValueError) as e:
            # Handle NaN/Inf errors during generation
            if "nan" in str(e).lower() or "inf" in str(e).lower():
                return f"[Generation failed: Model outputs contain NaN/Inf. Training may have diverged.]"
            else:
                raise


def generate_multiple_samples(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompts: list[str],
    max_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    device: Optional[torch.device] = None
) -> list[str]:
    """
    Generate text from multiple prompts.

    Args:
        model: The GPT model
        tokenizer: Tokenizer for encoding/decoding
        prompts: List of text prompts
        max_tokens: Maximum number of tokens to generate per prompt
        temperature: Sampling temperature
        top_k: If set, only sample from top-k tokens
        top_p: If set, use nucleus sampling
        device: Device to run generation on

    Returns:
        List of generated texts (including prompts)
    """
    results = []
    for prompt in prompts:
        generated = generate_sample(
            model, tokenizer, prompt, max_tokens,
            temperature, top_k, top_p, device
        )
        results.append(generated)

    return results


class InferenceMonitor:
    """
    Monitor training progress by periodically generating samples.

    Tracks generation history and provides formatted output for TUI display.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer_path: str | Path,
        prompt: str = "Once upon a time",
        max_tokens: int = 50,
        temperature: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize inference monitor.

        Args:
            model: The GPT model
            tokenizer_path: Path to tokenizer file
            prompt: Default prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            device: Device to run generation on
        """
        self.model = model
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.device = device or next(model.parameters()).device

        # History of generations
        self.generation_history = []
        self.latest_generation = None

    def generate(
        self,
        prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate a sample and update history.

        Args:
            prompt: Override default prompt
            max_tokens: Override default max_tokens
            temperature: Override default temperature

        Returns:
            Generated text
        """
        prompt = prompt or self.prompt
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        # Generate
        generated = generate_sample(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens,
            temperature,
            device=self.device
        )

        # Update history
        self.latest_generation = generated
        self.generation_history.append({
            'prompt': prompt,
            'output': generated,
            'length': len(self.tokenizer.encode(generated).ids)
        })

        return generated

    def get_latest(self) -> Optional[str]:
        """
        Get the latest generation.

        Returns:
            Latest generated text, or None if no generations yet
        """
        return self.latest_generation

    def get_formatted_output(self, max_length: int = 200) -> str:
        """
        Get formatted output for TUI display.

        Args:
            max_length: Maximum length to display (truncate if longer)

        Returns:
            Formatted string for display
        """
        if self.latest_generation is None:
            return "No generation yet"

        output = self.latest_generation

        # Truncate if too long
        if len(output) > max_length:
            output = output[:max_length] + "..."

        # Get token count
        token_count = len(self.tokenizer.encode(output).ids)

        return f"{output}\n\n[{token_count} tokens]"

    def get_history_summary(self) -> str:
        """
        Get a summary of generation history.

        Returns:
            Summary string
        """
        if not self.generation_history:
            return "No generations yet"

        total_gens = len(self.generation_history)
        avg_length = sum(g['length'] for g in self.generation_history) / total_gens

        return f"Total generations: {total_gen}, Avg length: {avg_length:.1f} tokens"

    def clear_history(self) -> None:
        """Clear generation history."""
        self.generation_history = []
        self.latest_generation = None


def test_generation(
    model: nn.Module,
    tokenizer_path: str | Path,
    prompts: Optional[list[str]] = None,
    max_tokens: int = 50,
    device: Optional[torch.device] = None
) -> None:
    """
    Test model generation with multiple prompts.

    Args:
        model: The GPT model
        tokenizer_path: Path to tokenizer file
        prompts: List of prompts to test. If None, uses defaults.
        max_tokens: Maximum tokens to generate
        device: Device to run generation on
    """
    if prompts is None:
        prompts = [
            "Once upon a time",
            "The little girl",
            "In a forest far away",
            "One day, a brave knight"
        ]

    tokenizer = load_tokenizer(tokenizer_path)

    print("=" * 60)
    print("Generation Test")
    print("=" * 60)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Prompt: \"{prompt}\"")
        print("-" * 60)

        generated = generate_sample(
            model, tokenizer, prompt, max_tokens,
            temperature=1.0, device=device
        )

        print(generated)
        print()

    print("=" * 60)
