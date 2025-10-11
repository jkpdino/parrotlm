"""GPT model implementation with configurable architecture.

Modern features:
- Flash Attention (PyTorch SDPA) for 2-4x speedup
- RoPE (Rotary Position Embeddings) for better length extrapolation
- RMSNorm (Root Mean Square Normalization) for faster training
- SwiGLU activation for better model quality
- GQA (Grouped Query Attention) for efficient inference
- Depth-scaled initialization for training stability in deep networks
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) from "RoFormer: Enhanced Transformer with Rotary Position Embedding".

    Instead of adding position info, RoPE rotates Q and K vectors based on their absolute position.
    This encodes relative positional information in the attention scores.
    """

    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        # Compute inverse frequencies for each dimension pair
        # inv_freq[i] = 1 / (base^(2i/dim)) for i in [0, dim/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos and sin for all positions up to max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Precompute cos/sin values for positions [0, seq_len)."""
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)

        # Cache cos and sin (no duplication - we'll handle that in apply_rotary_emb)
        self.register_buffer("cos_cached", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", freqs.sin()[None, None, :, :], persistent=False)
        self.max_seq_len_cached = seq_len

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos/sin values for sequence length.

        Args:
            seq_len: Sequence length

        Returns:
            (cos, sin) both of shape (1, 1, seq_len, dim/2)
        """
        # Rebuild cache if sequence is longer than cached
        if seq_len > self.max_seq_len_cached:
            self._build_cache(seq_len)

        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :]
        )


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to query or key tensors.

    Args:
        x: (batch, n_heads, seq_len, head_dim) query or key tensor
        cos: (1, 1, seq_len, head_dim/2) cosine values
        sin: (1, 1, seq_len, head_dim/2) sine values

    Returns:
        Rotated tensor of same shape as x
    """
    # Split x into even and odd dimensions: [x0, x1, x2, x3, ...] -> ([x0, x2, ...], [x1, x3, ...])
    x1 = x[..., ::2]   # Even indices (batch, n_heads, seq_len, head_dim/2)
    x2 = x[..., 1::2]  # Odd indices (batch, n_heads, seq_len, head_dim/2)

    # Apply rotation: [x_even, x_odd] -> [x_even*cos - x_odd*sin, x_even*sin + x_odd*cos]
    # Then interleave back to original structure
    rotated = torch.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1).flatten(-2)

    return rotated.type_as(x)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Simpler and faster than LayerNorm - only normalizes by RMS without mean centering.
    Used in: Llama, Mistral, Gemini, and most modern LLMs.

    Args:
        dim: Normalization dimension
        eps: Small value to prevent division by zero
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of any shape (..., dim)

        Returns:
            Normalized tensor of same shape
        """
        # Compute RMS (root mean square) using rsqrt for better numerical stability
        # x² → mean → rsqrt → multiply (more stable than sqrt → divide)
        rms_inv = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize and scale
        return self.weight * x * rms_inv


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with RoPE, Flash Attention, and GQA.

    GQA (Grouped Query Attention): Multiple query heads share the same key/value heads.
    This dramatically reduces the KV cache size during inference while maintaining quality.

    Example: n_heads=6, n_kv_heads=2 means 3 Q heads share each K/V head.
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int = None, dropout: float = 0.0, max_seq_len: int = 2048):
        super().__init__()
        assert dim % n_heads == 0, f"dim ({dim}) must be divisible by n_heads ({n_heads})"

        # Default to MHA (multi-head attention) if n_kv_heads not specified
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        assert n_heads % self.n_kv_heads == 0, f"n_heads ({n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // self.n_kv_heads  # How many Q heads share each KV head
        self.dropout = dropout

        # Flash Attention compatibility check
        if self.head_dim > 256:
            import warnings
            warnings.warn(
                f"head_dim={self.head_dim} > 256 may not use Flash Attention. "
                f"Consider reducing dim or increasing n_heads for optimal performance.",
                UserWarning
            )

        # Separate Q, K, V projections for GQA
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # RoPE for positional encoding
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len=max_seq_len)

        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
            attention_mask: (batch, 1, seq_len, seq_len) pre-combined additive mask (0 for attend, -inf for mask)
                           Already includes both causal + document masking combined
                           If None, automatic causal masking is used

        Returns:
            (batch, seq_len, dim)
        """
        batch_size, seq_len, dim = x.shape

        # Compute Q, K, V separately (for GQA)
        q = self.q_proj(x)  # (batch, seq_len, n_heads * head_dim)
        k = self.k_proj(x)  # (batch, seq_len, n_kv_heads * head_dim)
        v = self.v_proj(x)  # (batch, seq_len, n_kv_heads * head_dim)

        # Reshape to separate heads
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # (batch, n_heads, seq_len, head_dim)

        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # (batch, n_kv_heads, seq_len, head_dim)

        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # (batch, n_kv_heads, seq_len, head_dim)

        # Apply RoPE to Q and K (not V!)
        cos, sin = self.rope(seq_len)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Repeat K and V to match number of Q heads (GQA)
        # If n_heads=6 and n_kv_heads=2, we repeat each KV head 3 times
        if self.n_kv_heads != self.n_heads:
            k = k.repeat_interleave(self.n_rep, dim=1)  # (batch, n_heads, seq_len, head_dim)
            v = v.repeat_interleave(self.n_rep, dim=1)  # (batch, n_heads, seq_len, head_dim)

        # Flash Attention (PyTorch SDPA)
        # Mask is already pre-combined by GPT.forward(), just use it directly
        if attention_mask is not None:
            # Mask already includes causal + document combined
            # Shape: (batch, 1, seq_len, seq_len)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False  # We already have the combined mask
            )
        else:
            # No mask provided, use automatic causal masking (faster, enables Flash Attention)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        # (batch, n_heads, seq_len, head_dim)

        # Reshape back
        out = out.transpose(1, 2).contiguous()  # (batch, seq_len, n_heads, head_dim)
        out = out.reshape(batch_size, seq_len, dim)

        # Output projection
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network with SwiGLU activation.

    SwiGLU: Swish-Gated Linear Unit from "GLU Variants Improve Transformer"
    Formula: SwiGLU(x) = (Swish(W_gate·x) ⊙ W_up·x) · W_down

    Used in: Llama, PaLM, likely GPT-4
    """

    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        # For SwiGLU, we use 2/3 of the standard 4x to maintain similar param count
        # Standard FFN: 2 layers × (dim → 4*dim) = 2 × (dim × 4*dim) = 8*dim² params
        # SwiGLU FFN: 3 layers × (dim → 8/3*dim) = 3 × (dim × 8/3*dim) = 8*dim² params (same!)
        # Calculation: hidden_dim = 2/3 × 4 × dim = 8/3 × dim
        hidden_dim = int(2 * hidden_mult * dim / 3)

        # Three projections for SwiGLU
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)

        Returns:
            (batch, seq_len, dim)
        """
        # SwiGLU activation
        gate = F.silu(self.gate_proj(x))  # SiLU = Swish activation
        up = self.up_proj(x)
        x = self.down_proj(gate * up)  # Gated multiplication
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture using RMSNorm."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int = None,
        ffn_hidden_mult: int = 4,
        dropout: float = 0.0,
        max_seq_len: int = 2048
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, n_heads, n_kv_heads, dropout, max_seq_len)

        self.norm2 = RMSNorm(dim)
        self.ffn = FeedForward(dim, ffn_hidden_mult, dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
            attention_mask: (batch, seq_len, seq_len) additive mask

        Returns:
            (batch, seq_len, dim)
        """
        # Pre-norm attention with residual
        x = x + self.attn(self.norm1(x), attention_mask)

        # Pre-norm FFN with residual
        x = x + self.ffn(self.norm2(x))

        return x


class GPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer) model with modern improvements.

    Features:
    - RoPE (Rotary Position Embeddings) instead of learned positions
    - Flash Attention via PyTorch SDPA for 2-4x speedup
    - RMSNorm for faster normalization (~15% speedup)
    - SwiGLU activation for better quality
    - GQA (Grouped Query Attention) for efficient inference
    - Depth-scaled initialization (GPT-2/3 style) for training stability
    - Tied embeddings (input/output share weights)
    - Pre-norm architecture

    Configuration for ~1.8M parameters (excluding embeddings):
        GPT(vocab_size=8000, dim=96, depth=16, n_heads=6, n_kv_heads=2)

    Args:
        vocab_size: Size of the vocabulary
        dim: Model dimension (embedding size)
        depth: Number of transformer layers
        n_heads: Number of query attention heads
        n_kv_heads: Number of key/value attention heads (for GQA). If None, uses n_heads (standard MHA)
        context_length: Maximum sequence length (for cache, can exceed during inference)
        dropout: Dropout probability
        ffn_hidden_mult: FFN hidden dimension multiplier (default: 4x)
        pad_token_id: Padding token ID for masking (default: 0)
        rope_scaling_factor: RoPE cache scaling factor (default: 4.0) - allows extrapolation beyond context_length
        use_gradient_checkpointing: If True, use gradient checkpointing to reduce memory usage during training

    Example:
        >>> model = GPT(vocab_size=8000, dim=96, depth=16, n_heads=6)
        >>> print(f"Transformer params: {model.count_parameters()['transformer']:,}")
        >>> print(f"Total params: {model.count_parameters()['total']:,}")
        >>>
        >>> # Forward pass
        >>> tokens = torch.randint(0, 8000, (4, 512))  # (batch, seq_len)
        >>> masks = torch.zeros(4, 512, 512)  # (batch, seq_len, seq_len)
        >>> logits = model(tokens, masks)  # (4, 512, 8000)
    """

    def __init__(
        self,
        vocab_size: int = 8000,
        dim: int = 96,
        depth: int = 16,
        n_heads: int = 6,
        n_kv_heads: int = None,
        context_length: int = 512,
        dropout: float = 0.0,
        ffn_hidden_mult: int = 4,
        pad_token_id: int = 0,
        rope_scaling_factor: float = 4.0,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.context_length = context_length
        self.pad_token_id = pad_token_id
        self.rope_scaling_factor = rope_scaling_factor
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Token embeddings (position info comes from RoPE instead)
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.emb_dropout = nn.Dropout(dropout)

        # Causal mask cache (shared across all layers)
        self.register_buffer("_causal_mask_cache", None, persistent=False)
        self._causal_mask_seq_len = 0

        # Transformer blocks (RoPE is inside attention layers)
        # Scale RoPE cache by rope_scaling_factor to allow extrapolation beyond context_length
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim, n_heads, self.n_kv_heads, ffn_hidden_mult, dropout,
                max_seq_len=int(context_length * rope_scaling_factor)
            )
            for _ in range(depth)
        ])

        # Final RMS norm
        self.norm_f = RMSNorm(dim)

        # Output projection (tied with token embeddings)
        # We don't create a separate layer; instead we use token_embedding.weight

        # Initialize weights
        self._init_weights()

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Get cached causal mask, creating it if needed.

        Args:
            seq_len: Sequence length
            device: Device to create mask on

        Returns:
            Causal mask of shape (1, 1, seq_len, seq_len) with 0/-inf values
        """
        # Rebuild cache if sequence is longer or device changed
        if seq_len > self._causal_mask_seq_len or self._causal_mask_cache is None or self._causal_mask_cache.device != device:
            # Create causal mask
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
            causal_mask = causal_mask.view(1, 1, seq_len, seq_len)

            # Convert to additive format (0/-inf)
            causal_additive = torch.where(causal_mask, 0.0, float('-inf'))

            self._causal_mask_cache = causal_additive
            self._causal_mask_seq_len = seq_len

        return self._causal_mask_cache[:, :, :seq_len, :seq_len]

    def _init_weights(self):
        """
        Initialize model weights with depth-scaled initialization.

        Uses GPT-style depth scaling: residual projection layers are scaled down
        by 1/sqrt(2*depth) to prevent activation/gradient explosion in deep networks.
        """
        # Token embeddings: normal distribution
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

        # Depth scaling factor for residual projections
        # Each layer has 2 residual connections (attention + FFN)
        residual_scale = 1.0 / math.sqrt(2 * self.depth)

        # Linear layers: normal distribution with depth scaling for residual paths
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Check if this is a residual projection layer
                # These layers output directly to the residual stream
                if 'out_proj' in name or 'down_proj' in name:
                    # Scale down for residual path (GPT-2/3 style)
                    std = 0.02 * residual_scale
                else:
                    # Normal initialization for other projections
                    std = 0.02

                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                # RMSNorm only has weight (scale), no bias
                nn.init.ones_(module.weight)

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            tokens: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len, seq_len) additive mask (0/-inf format)
                           0 = can attend, -inf = cannot attend
                           If None, only causal masking is applied

        Returns:
            logits: (batch, seq_len, vocab_size) unnormalized predictions
        """
        batch_size, seq_len = tokens.shape

        # Validate token IDs
        if tokens.min() < 0 or tokens.max() >= self.vocab_size:
            raise ValueError(
                f"Token IDs must be in range [0, {self.vocab_size}), "
                f"got range [{tokens.min().item()}, {tokens.max().item()}]"
            )

        # Validate attention mask shape if provided
        if attention_mask is not None:
            expected_shape = (batch_size, seq_len, seq_len)
            if attention_mask.shape != expected_shape:
                raise ValueError(
                    f"Expected attention_mask shape {expected_shape}, "
                    f"got {attention_mask.shape}"
                )

        # Combine causal mask with document mask ONCE (not in each layer)
        # This is the key optimization - masks are created once and reused
        if attention_mask is not None:
            # Get cached causal mask
            causal_mask = self._get_causal_mask(seq_len, tokens.device)  # (1, 1, seq_len, seq_len)

            # Combine with document mask
            # Document mask: (batch, seq_len, seq_len)
            # Expand for broadcast: (batch, 1, seq_len, seq_len)
            combined_mask = attention_mask.unsqueeze(1) + causal_mask
        else:
            # No document mask, use None and let attention layers use is_causal=True
            combined_mask = None

        # Embed tokens (position info comes from RoPE in attention layers)
        x = self.token_embedding(tokens)  # (batch, seq_len, dim)
        x = self.emb_dropout(x)

        # Apply transformer blocks with optional gradient checkpointing
        # Pass the SAME combined mask to all blocks (no recreating masks per layer)
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory during training
                x = torch.utils.checkpoint.checkpoint(block, x, combined_mask, use_reentrant=False)
            else:
                x = block(x, combined_mask)

        # Final RMS norm
        x = self.norm_f(x)

        # Project to vocabulary (using tied embeddings)
        logits = F.linear(x, self.token_embedding.weight)  # (batch, seq_len, vocab_size)

        return logits

    def count_parameters(self, exclude_embeddings: bool = False) -> dict[str, int]:
        """
        Count model parameters.

        Args:
            exclude_embeddings: If True, exclude embedding parameters

        Returns:
            Dictionary with parameter counts:
            - 'transformer': transformer parameters (excluding embeddings)
            - 'embeddings': token embedding parameters
            - 'total': total parameters
            Note: Output weights are tied with token embeddings
        """
        # Token embedding parameters
        emb_params = self.token_embedding.weight.numel()

        # Transformer parameters (everything except token embeddings)
        transformer_params = sum(
            p.numel() for p in self.parameters()
        ) - emb_params

        total_params = emb_params + transformer_params

        return {
            'embeddings': emb_params,
            'transformer': transformer_params,
            'total': total_params
        }

    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        repetition_penalty: float = 1.0,
        eos_token_id: int = None
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively with advanced sampling options.

        Args:
            tokens: (batch, seq_len) initial token IDs
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top-k tokens
            top_p: If set, use nucleus sampling (keep tokens with cumulative prob >= p)
            repetition_penalty: Penalty for repeating tokens (>1.0 discourages repetition)
            eos_token_id: If set, stop generation when this token is generated

        Returns:
            (batch, seq_len + max_new_tokens) generated token IDs
        """
        self.eval()
        batch_size = tokens.size(0)

        # Track if each sequence has generated EOS
        finished = torch.zeros(batch_size, dtype=torch.bool, device=tokens.device)

        for _ in range(max_new_tokens):
            # Stop if all sequences have finished
            if eos_token_id is not None and finished.all():
                break

            # Crop to context length
            tokens_crop = tokens if tokens.size(1) <= self.context_length else tokens[:, -self.context_length:]

            # Forward pass
            with torch.no_grad():
                logits = self(tokens_crop)  # (batch, seq_len, vocab_size)

            # Get logits for last position
            logits = logits[:, -1, :].clone()  # (batch, vocab_size)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(tokens[i].tolist()):
                        # If score < 0, multiply by penalty (make more negative)
                        # If score > 0, divide by penalty (make smaller)
                        if logits[i, token_id] < 0:
                            logits[i, token_id] *= repetition_penalty
                        else:
                            logits[i, token_id] /= repetition_penalty

            # Apply temperature
            logits = logits / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[:, 0] = False

                # Scatter back to original indexing
                for i in range(batch_size):
                    indices_to_remove = sorted_indices[i, sorted_indices_to_remove[i]]
                    logits[i, indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            # Mark finished sequences
            if eos_token_id is not None:
                finished |= (next_token.squeeze(-1) == eos_token_id)

            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens

    def __repr__(self) -> str:
        """Return a detailed string representation of the model."""
        return (
            f"GPT(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  dim={self.dim},\n"
            f"  depth={self.depth},\n"
            f"  n_heads={self.n_heads},\n"
            f"  n_kv_heads={self.n_kv_heads},\n"
            f"  context_length={self.context_length},\n"
            f"  use_gradient_checkpointing={self.use_gradient_checkpointing},\n"
            f"  total_params={self.count_parameters()['total']:,}\n"
            f")"
        )


def create_default_gpt(use_mqa=False):
    """
    Create GPT model with default configuration (~1.8M transformer parameters).

    Args:
        use_mqa: If True, use MQA (1 KV head). If False, use GQA (2 KV heads).

    GQA (default): 6 query heads and 2 key/value heads (3:1 ratio) - Llama 3, Mistral
    MQA: 6 query heads and 1 key/value head (6:1 ratio) - PaLM, Falcon
    """
    n_kv_heads = 1 if use_mqa else 2

    return GPT(
        vocab_size=8000,
        dim=96,
        depth=16,
        n_heads=6,
        n_kv_heads=n_kv_heads,
        context_length=512,
        dropout=0.0,  # Modern LLMs don't use dropout
        ffn_hidden_mult=4
    )


def create_mqa_gpt():
    """
    Create GPT model with MQA (Multi-Query Attention).

    All 6 query heads share a single K/V head for maximum KV cache compression.
    Fastest inference at the cost of slight quality reduction vs GQA.
    """
    return create_default_gpt(use_mqa=True)


if __name__ == "__main__":
    # Demo
    print("GPT Model Configuration")
    print("=" * 60)

    model = create_default_gpt()

    params = model.count_parameters()
    print(f"\nParameter counts:")
    print(f"  Token embeddings:     {params['embeddings']:>10,}")
    print(f"  Transformer (16L):    {params['transformer']:>10,}")
    print(f"  Total:                {params['total']:>10,}")

    print(f"\nModel configuration:")
    print(f"  Vocabulary size:      {model.vocab_size:>10,}")
    print(f"  Model dimension:      {model.dim:>10,}")
    print(f"  Transformer layers:   {model.depth:>10,}")
    print(f"  Query heads (Q):      {model.n_heads:>10,}")
    print(f"  KV heads (GQA):       {model.n_kv_heads:>10,}")
    print(f"  Head dimension:       {model.dim // model.n_heads:>10,}")
    print(f"  Context length:       {model.context_length:>10,}")
    print(f"  Dropout:              {model.emb_dropout.p:>10.1f}")

    # Show depth-scaled initialization
    residual_scale = 1.0 / math.sqrt(2 * model.depth)
    print(f"\nInitialization:")
    print(f"  Base std:             {0.02:>10.4f}")
    print(f"  Residual scale:       {residual_scale:>10.4f}")
    print(f"  Residual std:         {0.02 * residual_scale:>10.4f}")

    # Test forward pass
    print("\n" + "=" * 60)
    print("Testing forward pass...")

    batch_size = 4
    seq_len = 512

    # Create dummy input
    tokens = torch.randint(0, model.vocab_size, (batch_size, seq_len))

    # Create dummy attention mask (all zeros = can attend everywhere)
    attention_mask = torch.zeros(batch_size, seq_len, seq_len)

    # Forward pass
    with torch.no_grad():
        logits = model(tokens, attention_mask)

    print(f"\nInput shape:  {tokens.shape}")
    print(f"Mask shape:   {attention_mask.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output dtype: {logits.dtype}")

    # Test with DocumentBatchLoader format
    print("\n" + "=" * 60)
    print("Testing with DocumentBatchLoader format...")

    # Simulate document boundaries with additive mask
    # Create a mask with 2 documents: [0:256] and [256:512]
    doc_mask = torch.zeros(batch_size, seq_len, seq_len)
    doc_mask[:, :256, 256:] = float('-inf')  # First doc can't attend to second
    doc_mask[:, 256:, :256] = float('-inf')  # Second doc can't attend to first

    with torch.no_grad():
        logits = model(tokens, doc_mask)

    print(f"\nDocument-aware mask applied successfully!")
    print(f"Output shape: {logits.shape}")

    # Test generation
    print("\n" + "=" * 60)
    print("Testing autoregressive generation...")

    # Start with a single token
    start_tokens = torch.randint(0, model.vocab_size, (2, 10))

    with torch.no_grad():
        generated = model.generate(start_tokens, max_new_tokens=20, temperature=1.0)

    print(f"\nStart tokens shape: {start_tokens.shape}")
    print(f"Generated shape:    {generated.shape}")
    print(f"Generated {generated.shape[1] - start_tokens.shape[1]} new tokens")

    # Compare GQA vs MQA
    print("\n" + "=" * 60)
    print("GQA vs MQA Comparison:")
    print("=" * 60)

    mqa_model = create_mqa_gpt()
    mqa_params = mqa_model.count_parameters()

    print(f"\n{'Metric':<25} {'GQA (2 KV)':<15} {'MQA (1 KV)':<15} {'Difference':<15}")
    print("-" * 70)
    print(f"{'Transformer params':<25} {params['transformer']:>14,} {mqa_params['transformer']:>14,} {mqa_params['transformer'] - params['transformer']:>14,}")
    print(f"{'Total params':<25} {params['total']:>14,} {mqa_params['total']:>14,} {mqa_params['total'] - params['total']:>14,}")

    # KV cache comparison
    seq_len_cache = 2048
    kv_cache_gqa = 2 * seq_len_cache * model.n_kv_heads * (model.dim // model.n_heads)
    kv_cache_mqa = 2 * seq_len_cache * mqa_model.n_kv_heads * (mqa_model.dim // mqa_model.n_heads)
    print(f"{'KV cache (seq=2048)':<25} {kv_cache_gqa:>14,} {kv_cache_mqa:>14,} {kv_cache_mqa - kv_cache_gqa:>14,}")
    print(f"{'Cache compression':<25} {'1.0x':>14} {f'{kv_cache_gqa/kv_cache_mqa:.1f}x':>14} {'':<15}")

    print("\n" + "=" * 60)
    print("Model ready for training!")
