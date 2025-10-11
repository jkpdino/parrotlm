#!/usr/bin/env python3
"""
Model size analyzer for GPT configurations.

Analyzes parameter counts for a given configuration, including:
- Total parameters
- Active parameters (for MoE models)
- Detailed breakdown by component

Usage:
    python size.py --config configs/training/my_config.yml
    python size.py --config configs/training/tinystories_moe.yml
"""

import argparse
import sys
from pathlib import Path

import torch

from src.train.config import load_training_config
from src.train.model import GPT


def count_parameters_detailed(model: GPT) -> dict:
    """
    Count parameters with detailed breakdown.

    Args:
        model: GPT model instance

    Returns:
        Dictionary with detailed parameter counts
    """
    breakdown = {
        'embeddings': {},
        'attention': {},
        'ffn_or_moe': {},
        'normalization': {},
        'total': 0,
        'active': 0  # For MoE models
    }

    # Token embeddings (tied with output projection)
    emb_params = model.token_embedding.weight.numel()
    breakdown['embeddings']['token_embedding'] = emb_params

    # Per-layer analysis
    total_attn = 0
    total_ffn = 0
    total_norm = 0

    for layer_idx, block in enumerate(model.blocks):
        # Attention parameters
        attn_params = sum(p.numel() for p in block.attn.parameters())
        total_attn += attn_params

        # FFN or MoE parameters
        if block.use_moe:
            # MoE: router + all experts
            router_params = block.ffn.router.weight.numel()
            expert_params = sum(p.numel() for p in block.ffn.experts.parameters())
            ffn_layer_params = router_params + expert_params
            total_ffn += ffn_layer_params
        else:
            # Standard FFN
            ffn_params = sum(p.numel() for p in block.ffn.parameters())
            total_ffn += ffn_params

        # Normalization parameters (2 RMSNorm per block)
        norm_params = block.norm1.weight.numel() + block.norm2.weight.numel()
        total_norm += norm_params

    breakdown['attention']['total'] = total_attn
    breakdown['attention']['per_layer'] = total_attn // model.depth

    breakdown['ffn_or_moe']['total'] = total_ffn
    breakdown['ffn_or_moe']['per_layer'] = total_ffn // model.depth

    breakdown['normalization']['per_block'] = total_norm // model.depth
    breakdown['normalization']['final'] = model.norm_f.weight.numel()
    breakdown['normalization']['total'] = total_norm + model.norm_f.weight.numel()

    # Total parameters
    breakdown['total'] = emb_params + total_attn + total_ffn + breakdown['normalization']['total']

    # Active parameters (for MoE)
    if model.use_moe:
        # Calculate active FFN params
        # For each MoE layer: router + (num_experts_active * single_expert_size)
        num_moe_layers = sum(1 for block in model.blocks if block.use_moe)

        if num_moe_layers > 0:
            # Get size of one expert from first MoE layer
            first_moe_block = next(block for block in model.blocks if block.use_moe)
            single_expert_params = sum(p.numel() for p in first_moe_block.ffn.experts[0].parameters())
            router_params = first_moe_block.ffn.router.weight.numel()

            # Active params per MoE layer = router + k experts
            active_ffn_per_layer = router_params + (model.num_experts_active * single_expert_params)
            total_active_ffn = active_ffn_per_layer * num_moe_layers

            # Non-MoE FFN layers
            num_standard_ffn = model.depth - num_moe_layers
            if num_standard_ffn > 0:
                standard_ffn_block = next((block for block in model.blocks if not block.use_moe), None)
                if standard_ffn_block:
                    standard_ffn_params = sum(p.numel() for p in standard_ffn_block.ffn.parameters())
                    total_active_ffn += standard_ffn_params * num_standard_ffn

            # Active total = embeddings + attention + active_ffn + norms
            breakdown['active'] = emb_params + total_attn + total_active_ffn + breakdown['normalization']['total']

            # Add breakdown for MoE
            breakdown['moe_details'] = {
                'num_experts': model.num_experts,
                'num_experts_active': model.num_experts_active,
                'num_moe_layers': num_moe_layers,
                'single_expert_params': single_expert_params,
                'router_params': router_params,
                'active_ffn_per_layer': active_ffn_per_layer,
                'total_ffn_per_layer': router_params + (model.num_experts * single_expert_params),
            }
    else:
        breakdown['active'] = breakdown['total']

    return breakdown


def format_number(n: int) -> str:
    """Format number with commas and abbreviated form."""
    # Abbreviated forms
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B ({n:,})"
    elif n >= 1_000_000:
        return f"{n/1_000_000:.2f}M ({n:,})"
    elif n >= 1_000:
        return f"{n/1_000:.2f}K ({n:,})"
    else:
        return f"{n:,}"


def print_size_analysis(config_path: Path):
    """Print detailed size analysis for a config."""
    # Load config
    config = load_training_config(config_path)

    print("\n" + "=" * 80)
    print(f"Model Size Analysis: {config.name}")
    print("=" * 80)

    # Create model (but don't load to device to save memory)
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
        use_gradient_checkpointing=config.model.use_gradient_checkpointing,
        use_moe=config.model.use_moe,
        num_experts=config.model.num_experts,
        num_experts_active=config.model.num_experts_active,
        moe_layers=config.model.moe_layers,
        router_aux_loss_coef=config.model.router_aux_loss_coef,
        router_z_loss_coef=config.model.router_z_loss_coef
    )

    # Get detailed breakdown
    breakdown = count_parameters_detailed(model)

    # Print configuration
    print("\n[Architecture]")
    print(f"  Vocabulary size:      {config.model.vocab_size:>12,}")
    print(f"  Model dimension:      {config.model.dim:>12,}")
    print(f"  Transformer layers:   {config.model.depth:>12,}")
    print(f"  Query heads:          {config.model.n_heads:>12,}")
    print(f"  KV heads (GQA):       {config.model.n_kv_heads:>12,}")
    print(f"  Head dimension:       {config.model.dim // config.model.n_heads:>12,}")
    print(f"  Context length:       {config.model.context_length:>12,}")
    print(f"  FFN multiplier:       {config.model.ffn_hidden_mult:>12,}x")

    if config.model.use_moe:
        print(f"\n[MoE Configuration]")
        print(f"  Total experts:        {config.model.num_experts:>12,}")
        print(f"  Active experts:       {config.model.num_experts_active:>12,}")
        if config.model.moe_layers is None:
            print(f"  MoE layers:           {'all':>12}")
        else:
            print(f"  MoE layers:           {len(config.model.moe_layers):>12,} / {config.model.depth}")
        print(f"  Load balance coef:    {config.model.router_aux_loss_coef:>12.4f}")
        print(f"  Router z-loss coef:   {config.model.router_z_loss_coef:>12.4f}")

    # Print parameter breakdown
    print("\n[Parameter Breakdown]")
    print(f"  Token embeddings:     {format_number(breakdown['embeddings']['token_embedding']):>25}")
    print(f"  Attention (total):    {format_number(breakdown['attention']['total']):>25}")
    print(f"    - Per layer:        {format_number(breakdown['attention']['per_layer']):>25}")
    print(f"  FFN/MoE (total):      {format_number(breakdown['ffn_or_moe']['total']):>25}")
    print(f"    - Per layer:        {format_number(breakdown['ffn_or_moe']['per_layer']):>25}")
    print(f"  Normalization:        {format_number(breakdown['normalization']['total']):>25}")
    print(f"    - Per block (2x):   {format_number(breakdown['normalization']['per_block']):>25}")
    print(f"    - Final norm:       {format_number(breakdown['normalization']['final']):>25}")

    # Print totals
    print("\n[Total Parameters]")
    print(f"  Total parameters:     {format_number(breakdown['total']):>25}")

    if config.model.use_moe:
        print(f"  Active parameters:    {format_number(breakdown['active']):>25}")
        efficiency = (breakdown['active'] / breakdown['total']) * 100
        print(f"  Active efficiency:    {efficiency:>24.1f}%")

        # MoE details
        moe = breakdown['moe_details']
        print(f"\n[MoE Details]")
        print(f"  Single expert size:   {format_number(moe['single_expert_params']):>25}")
        print(f"  Router size:          {format_number(moe['router_params']):>25}")
        print(f"  Total FFN/layer:      {format_number(moe['total_ffn_per_layer']):>25}")
        print(f"  Active FFN/layer:     {format_number(moe['active_ffn_per_layer']):>25}")

        # Compute capacity increase
        capacity_mult = moe['num_experts'] / moe['num_experts_active']
        print(f"  Capacity multiplier:  {capacity_mult:>24.1f}x")

    # Memory estimate (FP32)
    memory_fp32_mb = (breakdown['total'] * 4) / (1024 ** 2)
    memory_fp16_mb = (breakdown['total'] * 2) / (1024 ** 2)

    print(f"\n[Memory Estimates]")
    print(f"  FP32 (4 bytes/param): {memory_fp32_mb:>24.1f} MB")
    print(f"  FP16 (2 bytes/param): {memory_fp16_mb:>24.1f} MB")

    if config.model.use_moe:
        active_memory_fp32_mb = (breakdown['active'] * 4) / (1024 ** 2)
        active_memory_fp16_mb = (breakdown['active'] * 2) / (1024 ** 2)
        print(f"  Active FP32:          {active_memory_fp32_mb:>24.1f} MB")
        print(f"  Active FP16:          {active_memory_fp16_mb:>24.1f} MB")

    print("\n" + "=" * 80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze parameter counts for GPT model configurations"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration YAML file"
    )

    args = parser.parse_args()

    # Check config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Print analysis
    print_size_analysis(config_path)


if __name__ == "__main__":
    main()
