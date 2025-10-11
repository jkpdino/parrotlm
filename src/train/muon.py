"""
Muon optimizer implementation.

Muon: Momentum Orthogonalized by Newton-schulz
Paper: https://arxiv.org/abs/2402.03432

Muon is designed for training transformers efficiently with:
- Orthogonalized momentum using Newton-Schulz iteration
- Better gradient scaling properties
- Improved training stability

For optimal results, use a hybrid approach:
- Muon for transformer weights (attention, FFN)
- AdamW for embeddings (token embeddings)
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import List, Optional, Callable


def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute matrix inverse square root.

    Computes G^(-1/2) using Newton-Schulz iteration:
    X_{k+1} = X_k * (3*I - G*X_k^2) / 2

    Args:
        G: Input matrix (must be positive definite)
        steps: Number of iteration steps
        eps: Small value for numerical stability

    Returns:
        Approximate G^(-1/2)
    """
    # Add regularization for numerical stability
    if G.ndim == 1:
        # For 1D, use simple inverse sqrt with clamping
        return 1.0 / torch.sqrt(G.clamp(min=eps))

    # For 2D matrices, add small regularization to diagonal
    G_reg = G + eps * torch.eye(G.shape[0], device=G.device, dtype=G.dtype)

    # Normalize G for better numerical stability
    trace = torch.trace(G_reg)
    G_norm = G_reg / (trace / G_reg.shape[0])

    # Initialize with scaled identity (simpler initialization)
    X = torch.eye(G_norm.shape[0], device=G_norm.device, dtype=G_norm.dtype)

    # Newton-Schulz iteration with damping
    for _ in range(steps):
        A = X @ G_norm @ X
        # Add small damping for stability
        I = torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
        X = X @ (1.5 * I - 0.5 * A)

        # Clamp to prevent overflow
        X = torch.clamp(X, min=-100, max=100)

    # Denormalize
    scale_factor = torch.sqrt(trace / G_reg.shape[0])
    return X * scale_factor


class Muon(Optimizer):
    """
    Muon optimizer: Momentum Orthogonalized by Newton-schulz.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.02)
        momentum: Momentum coefficient (default: 0.95)
        ns_steps: Number of Newton-Schulz iteration steps (default: 5)
        weight_decay: Weight decay coefficient (default: 0.0)

    Example:
        >>> optimizer = Muon(model.parameters(), lr=0.02, momentum=0.95)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.0
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"Invalid Newton-Schulz steps: {ns_steps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss

        Returns:
            Loss value if closure is provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            ns_steps = group['ns_steps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(grad)

                buf = state['momentum_buffer']
                state['step'] += 1

                # Update momentum buffer: m_t = β*m_{t-1} + g_t
                buf.mul_(momentum).add_(grad)

                # Orthogonalize momentum for 2D tensors (weights)
                if buf.ndim >= 2:
                    # Compute Gram matrix: G = m_t @ m_t^T
                    # For efficiency, work with the smaller dimension
                    if buf.shape[0] <= buf.shape[1]:
                        # G = m @ m^T (shape: d_out x d_out)
                        G = buf @ buf.t()
                        # Compute G^(-1/2) with increased eps for MPS stability
                        G_inv_sqrt = newton_schulz(G, steps=ns_steps, eps=1e-6)
                        # Orthogonalize: m_orth = G^(-1/2) @ m
                        buf_orth = G_inv_sqrt @ buf
                    else:
                        # G = m^T @ m (shape: d_in x d_in)
                        G = buf.t() @ buf
                        # Compute G^(-1/2) with increased eps for MPS stability
                        G_inv_sqrt = newton_schulz(G, steps=ns_steps, eps=1e-6)
                        # Orthogonalize: m_orth = m @ G^(-1/2)
                        buf_orth = buf @ G_inv_sqrt

                    # Check for NaN/Inf in orthogonalized momentum
                    if not torch.isfinite(buf_orth).all():
                        # Fall back to standard momentum if orthogonalization fails
                        buf_orth = buf

                    # Update parameters with orthogonalized momentum
                    p.data.add_(buf_orth, alpha=-lr)
                else:
                    # For 1D tensors (biases, norms), use standard momentum
                    p.data.add_(buf, alpha=-lr)

        return loss


def create_hybrid_optimizer(
    model: nn.Module,
    lr: float = 0.02,
    muon_momentum: float = 0.95,
    adamw_lr: Optional[float] = None,
    adamw_betas: tuple = (0.9, 0.999),
    weight_decay: float = 0.1
) -> Optimizer:
    """
    Create a hybrid optimizer: Muon for transformer weights, AdamW for embeddings.

    This is the recommended setup for training transformers with Muon.

    Args:
        model: The model to optimize
        lr: Learning rate for Muon (default: 0.02)
        muon_momentum: Momentum for Muon (default: 0.95)
        adamw_lr: Learning rate for AdamW. If None, uses lr/10 (default: None)
        adamw_betas: Beta parameters for AdamW (default: (0.9, 0.999))
        weight_decay: Weight decay for both optimizers (default: 0.1)

    Returns:
        Optimizer (actually a list of optimizers that can be stepped together)

    Example:
        >>> optimizer = create_hybrid_optimizer(model, lr=0.02)
        >>> # In training loop:
        >>> for opt in optimizer:
        >>>     opt.zero_grad()
        >>> loss.backward()
        >>> for opt in optimizer:
        >>>     opt.step()
    """
    if adamw_lr is None:
        adamw_lr = lr / 10.0

    # Separate parameters: embeddings vs transformer weights
    embedding_params = []
    transformer_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Embeddings typically contain "embedding" or "emb" in their name
        if 'embedding' in name.lower() or 'emb' in name.lower():
            embedding_params.append(param)
        else:
            transformer_params.append(param)

    optimizers = []

    # Muon for transformer weights
    if transformer_params:
        muon = Muon(
            transformer_params,
            lr=lr,
            momentum=muon_momentum,
            weight_decay=weight_decay
        )
        optimizers.append(muon)

    # AdamW for embeddings
    if embedding_params:
        adamw = torch.optim.AdamW(
            embedding_params,
            lr=adamw_lr,
            betas=adamw_betas,
            weight_decay=weight_decay
        )
        optimizers.append(adamw)

    # Return a wrapper that allows treating multiple optimizers as one
    return HybridOptimizer(optimizers)


class HybridOptimizer:
    """
    Wrapper for multiple optimizers to provide a unified interface.

    Allows treating multiple optimizers as a single optimizer for convenience.
    """

    def __init__(self, optimizers: List[Optimizer]):
        """
        Initialize hybrid optimizer.

        Args:
            optimizers: List of optimizers to wrap
        """
        self.optimizers = optimizers

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients for all optimizers."""
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Optional[Callable[[], float]] = None) -> None:
        """Step all optimizers."""
        for optimizer in self.optimizers:
            optimizer.step(closure)

    def state_dict(self) -> dict:
        """Get state dict for all optimizers."""
        return {
            'optimizers': [opt.state_dict() for opt in self.optimizers]
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state dict for all optimizers."""
        for optimizer, opt_state in zip(self.optimizers, state_dict['optimizers']):
            optimizer.load_state_dict(opt_state)

    def __iter__(self):
        """Allow iteration over optimizers."""
        return iter(self.optimizers)

    def __len__(self):
        """Number of optimizers."""
        return len(self.optimizers)

    def __repr__(self):
        """String representation."""
        return f"HybridOptimizer({len(self.optimizers)} optimizers)"
