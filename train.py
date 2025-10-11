#!/usr/bin/env python3
"""
Training script for GPT models with TUI interface.

Features:
- Interactive configuration selection
- Real-time training metrics visualization
- Pause/resume training
- Checkpoint management with auto-resume
- Sample generation monitoring
- Validation loss tracking

Usage:
    python train.py --config configs/training/my_config.yml
    python train.py --resume training_runs/my_model/last_checkpoint.pt
    python train.py  # Interactive config selection
"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Label, Static
from textual_plotext import PlotextPlot

from src.tokenizer import load_tokenizer
from src.train.checkpoints import CheckpointManager
from src.train.config import TrainingRunConfig, load_training_config
from src.train.inference import InferenceMonitor
from src.train.metrics import MetricsTracker
from src.train.model import GPT
from src.train.muon import Muon, create_hybrid_optimizer, HybridOptimizer
from src.train.pipeline import DocumentBatchLoader


class ConfigReviewScreen(Static):
    """Screen showing configuration details before training starts."""

    def __init__(self, config: TrainingRunConfig, resume_info: Optional[dict] = None):
        super().__init__()
        self.config = config
        self.resume_info = resume_info

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        # Determine device
        if torch.cuda.is_available():
            device_info = f"cuda ({torch.cuda.get_device_name(0)})"
        # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        #     device_info = "mps (Apple Silicon)"
        else:
            device_info = "cpu"

        # Model info
        model_params = self._estimate_params()
        model_info = f"""
[bold cyan]Model Configuration:[/bold cyan]
  Name: {self.config.name}
  Architecture: GPT
  Dimensions: {self.config.model.dim}
  Depth: {self.config.model.depth} layers
  Heads: {self.config.model.n_heads} query, {self.config.model.n_kv_heads} kv (GQA)
  Context: {self.config.model.context_length} tokens
  Vocab: {self.config.model.vocab_size:,}
  Parameters: ~{model_params:,}
  Device: {device_info}

[bold cyan]Data Configuration:[/bold cyan]
  Dataset: {self.config.data.dataset_path}
  Tokenizer: {self.config.data.tokenizer_path}
  Batch size: {self.config.data.batch_size}
  Shuffle: {self.config.data.shuffle}

[bold cyan]Training Configuration:[/bold cyan]
  Learning rate: {self.config.training.learning_rate}
  Optimizer: {self.config.training.optimizer}
  Max batches: {self.config.training.max_batches or 'N/A'}
  Max tokens: {f'{self.config.training.max_tokens:,}' if self.config.training.max_tokens else 'N/A'}
  Warmup: {self.config.training.warmup_batches} batches
  LR schedule: {self.config.training.lr_schedule}
  Weight decay: {self.config.training.weight_decay}
  Gradient clip: {self.config.training.grad_clip}

[bold cyan]Checkpointing:[/bold cyan]
  Save every: {self.config.checkpointing.save_every_n_batches} batches
  Keep last: {self.config.checkpointing.keep_last_n} checkpoints
  Output: {self.config.checkpointing.output_dir}/{self.config.name}/

[bold cyan]Monitoring:[/bold cyan]
  Validation every: {self.config.monitoring.validate_every_n_batches} batches
  Validation samples: {self.config.monitoring.validation_samples}
  Inference every: {self.config.monitoring.inference_every_n_batches} batches
  Inference prompt: "{self.config.monitoring.inference_prompt}"
"""

        if self.resume_info:
            resume_text = f"""
[bold yellow]Resuming from checkpoint:[/bold yellow]
  Batch: {self.resume_info['batch']}
  Tokens: {self.resume_info['tokens']:,}
  Train loss: {self.resume_info['train_loss']:.4f if self.resume_info['train_loss'] else 'N/A'}
  Val loss: {self.resume_info['val_loss']:.4f if self.resume_info['val_loss'] else 'N/A'}
"""
            model_info += resume_text

        model_info += "\n[bold green]Press ENTER to start training, Q to quit[/bold green]"

        yield Label(model_info)

    def _estimate_params(self) -> int:
        """Estimate parameter count."""
        cfg = self.config.model
        # Token embeddings
        emb = cfg.vocab_size * cfg.dim

        # Transformer blocks
        per_block = (
            # Attention
            cfg.dim * cfg.n_heads * (cfg.dim // cfg.n_heads) * 3 +  # QKV projections
            cfg.dim * cfg.dim +  # Output projection
            # FFN
            cfg.dim * (int(2 * 4 * cfg.dim / 3)) * 3 +  # SwiGLU has 3 projections
            # Norms
            cfg.dim * 2
        )
        transformer = per_block * cfg.depth

        return emb + transformer


class TrainingApp(App):
    """Main training TUI application."""

    CSS = """
    #inference-output {
        height: 8;
        border: solid cyan;
        padding: 1;
    }

    #loss-graphs {
        height: 15;
    }

    #metrics-bar {
        height: 5;
        dock: bottom;
        border: solid green;
    }

    .plot {
        height: 100%;
    }
    """

    BINDINGS = [
        ("space", "toggle_pause", "Pause/Resume"),
        ("s", "save_checkpoint", "Save"),
        ("r", "generate_sample", "Generate"),
        ("q", "quit_training", "Quit"),
    ]

    def __init__(
        self,
        config: TrainingRunConfig,
        resume_checkpoint: Optional[Path] = None
    ):
        super().__init__()
        self.config = config
        self.resume_checkpoint = resume_checkpoint

        # Training state
        self.is_paused = False
        self.is_training = False
        self.should_quit = False

        # Components (will be initialized in on_mount)
        # Priority: CUDA > CPU (MPS disabled for testing)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        #     self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model: Optional[GPT] = None
        self.optimizer = None
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.metrics_tracker: Optional[MetricsTracker] = None
        self.inference_monitor: Optional[InferenceMonitor] = None
        self.train_loader: Optional[DocumentBatchLoader] = None
        self.val_loader: Optional[DocumentBatchLoader] = None

        # Training progress
        self.current_batch = 0
        self.tokens_seen = 0
        self.current_lr = config.training.learning_rate
        self.start_time = time.time()

    def compose(self) -> ComposeResult:
        """Create child widgets for training console."""
        yield Header()

        # Inference output at top
        yield Static("Waiting for first generation...", id="inference-output")

        # Loss graphs
        with Horizontal(id="loss-graphs"):
            yield PlotextPlot(classes="plot")
            yield PlotextPlot(classes="plot")

        # Metrics bar at bottom
        yield Static("Initializing...", id="metrics-bar")

        yield Footer()

    async def on_mount(self) -> None:
        """Initialize training components."""
        # Show loading message
        self.query_one("#metrics-bar", Static).update("Loading model and data...")

        # Initialize components
        await self._initialize_training()

        # Update UI
        self.query_one("#metrics-bar", Static).update("Ready to train! Press SPACE to pause/resume")

        # Start training loop
        self.set_interval(0.1, self._training_step)
        self.set_interval(1.0, self._update_ui)

    async def _initialize_training(self) -> None:
        """Initialize all training components."""
        # Create model
        self.model = GPT(
            vocab_size=self.config.model.vocab_size,
            dim=self.config.model.dim,
            depth=self.config.model.depth,
            n_heads=self.config.model.n_heads,
            n_kv_heads=self.config.model.n_kv_heads,
            context_length=self.config.model.context_length,
            dropout=self.config.model.dropout,
            ffn_hidden_mult=self.config.model.ffn_hidden_mult,
            pad_token_id=self.config.model.pad_token_id,
            rope_scaling_factor=self.config.model.rope_scaling_factor,
            use_gradient_checkpointing=self.config.model.use_gradient_checkpointing
        ).to(self.device)

        # Create optimizer
        if self.config.training.optimizer == "muon":
            self.optimizer = create_hybrid_optimizer(
                self.model,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )

        # Create checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            run_name=self.config.name,
            output_dir=self.config.checkpointing.output_dir,
            keep_last_n=self.config.checkpointing.keep_last_n,
            save_every_n_batches=self.config.checkpointing.save_every_n_batches
        )

        # Create metrics tracker
        run_dir = Path(self.config.checkpointing.output_dir) / self.config.name
        resume = self.resume_checkpoint is not None
        self.metrics_tracker = MetricsTracker(
            run_dir=run_dir,
            window_size=self.config.monitoring.graph_window_size,
            resume=resume
        )

        # Resume from checkpoint if specified
        if self.resume_checkpoint:
            self.current_batch, self.tokens_seen, _ = self.checkpoint_manager.resume_from_checkpoint(
                self.resume_checkpoint,
                self.model,
                self.optimizer,
                self.device
            )

        # Create data loaders
        tokenizer = load_tokenizer(self.config.data.tokenizer_path)
        dataset_path = Path(self.config.data.dataset_path)

        # Get special token IDs
        pad_token_id = self.config.model.pad_token_id
        bos_token_id = tokenizer.token_to_id("<|bos|>")
        eos_token_id = tokenizer.token_to_id("<|eos|>")

        # Training loader
        self.train_loader = DocumentBatchLoader(
            tokens_path=str(dataset_path / "train.npy"),
            offsets_path=str(dataset_path / "train_offsets.npy"),
            context_length=self.config.model.context_length,
            batch_size=self.config.data.batch_size,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            shuffle=self.config.data.shuffle,
            return_labels=True,
            device=self.device,
            return_masks=False  # TEMP: Disable to test Flash Attention
        )

        # Validation loader
        self.val_loader = DocumentBatchLoader(
            tokens_path=str(dataset_path / "validation.npy"),
            offsets_path=str(dataset_path / "validation_offsets.npy"),
            context_length=self.config.model.context_length,
            batch_size=self.config.data.batch_size,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            shuffle=False,
            return_labels=True,
            device=self.device,
            return_masks=False  # TEMP: Disable to test Flash Attention
        )

        # Create inference monitor
        self.inference_monitor = InferenceMonitor(
            model=self.model,
            tokenizer_path=self.config.data.tokenizer_path,
            prompt=self.config.monitoring.inference_prompt,
            max_tokens=self.config.monitoring.inference_max_tokens,
            temperature=self.config.monitoring.inference_temperature,
            device=self.device
        )

        # Generate initial sample
        self.inference_monitor.generate()
        self._update_inference_display()

        # Start training
        self.is_training = True

    def _training_step(self) -> None:
        """Execute one training step."""
        if not self.is_training or self.is_paused or self.should_quit:
            return

        # Check if we've reached max batches/tokens
        if self.config.training.max_batches and self.current_batch >= self.config.training.max_batches:
            self.is_training = False
            self._save_checkpoint()
            return

        if self.config.training.max_tokens and self.tokens_seen >= self.config.training.max_tokens:
            self.is_training = False
            self._save_checkpoint()
            return

        # Get batch
        try:
            tokens, masks, labels = next(self.train_loader)
        except StopIteration:
            # Epoch complete, reset loader
            self.train_loader.reset()
            tokens, masks, labels = next(self.train_loader)

        # Forward pass
        self.model.train()
        logits = self.model(tokens, masks)

        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, self.config.model.vocab_size),
            labels.view(-1),
            ignore_index=0
        )

        # Check for NaN/Inf
        if not torch.isfinite(loss):
            self.is_training = False
            self.query_one("#metrics-bar", Static).update(
                "[bold red]TRAINING STOPPED: Loss became NaN/Inf. Try lowering learning rate.[/bold red]"
            )
            return

        # Backward pass
        self.optimizer.zero_grad()

        loss.backward()

        # Check for gradient NaN/Inf after backward
        has_nan_grad = False
        for param in self.model.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                has_nan_grad = True
                break

        if has_nan_grad:
            self.is_training = False
            self.query_one("#metrics-bar", Static).update(
                "[bold red]TRAINING STOPPED: Gradients became NaN/Inf. Try lowering learning rate.[/bold red]"
            )
            return

        # Gradient clipping
        if self.config.training.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.grad_clip
            )

        # Optimizer step
        self.optimizer.step()

        # Update progress
        self.current_batch += 1
        self.tokens_seen += tokens.numel()

        # Update learning rate
        self.current_lr = self._get_lr()
        self._set_lr(self.current_lr)

        # Log metrics
        self.metrics_tracker.log(
            batch=self.current_batch,
            tokens_seen=self.tokens_seen,
            train_loss=loss.item(),
            learning_rate=self.current_lr
        )

        # Periodic tasks
        if self.current_batch % self.config.checkpointing.save_every_n_batches == 0:
            self._save_checkpoint(create_numbered=True)

        if self.current_batch % self.config.monitoring.validate_every_n_batches == 0:
            self._run_validation()

        if self.current_batch % self.config.monitoring.inference_every_n_batches == 0:
            self.inference_monitor.generate()
            self._update_inference_display()

    def _save_checkpoint(self, create_numbered: bool = False) -> None:
        """Save a checkpoint."""
        self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            batch_num=self.current_batch,
            tokens_seen=self.tokens_seen,
            train_loss=self.metrics_tracker.get_current_train_loss(),
            val_loss=self.metrics_tracker.get_current_val_loss(),
            learning_rate=self.current_lr,
            config=self.config,
            create_numbered=create_numbered
        )

    def _run_validation(self) -> None:
        """Run validation and log results."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        num_valid_batches = 0

        with torch.no_grad():
            try:
                for tokens, masks, labels in self.val_loader:
                    if num_batches >= self.config.monitoring.validation_samples // self.config.data.batch_size:
                        break

                    logits = self.model(tokens, masks)
                    loss = F.cross_entropy(
                        logits.view(-1, self.config.model.vocab_size),
                        labels.view(-1),
                        ignore_index=0
                    )

                    # Only accumulate if loss is finite
                    if torch.isfinite(loss):
                        total_loss += loss.item()
                        num_valid_batches += 1

                    num_batches += 1

            except StopIteration:
                # Reset validation loader if we run out
                self.val_loader.reset()

        # Only compute average if we have valid batches
        if num_valid_batches > 0:
            avg_val_loss = total_loss / num_valid_batches
        else:
            # If all validation losses were NaN, don't log validation loss
            avg_val_loss = None

        # Log validation loss (only if valid)
        if avg_val_loss is not None:
            self.metrics_tracker.log(
                batch=self.current_batch,
                tokens_seen=self.tokens_seen,
                val_loss=avg_val_loss,
                learning_rate=self.current_lr
            )

    def _get_lr(self) -> float:
        """Calculate current learning rate based on schedule."""
        base_lr = self.config.training.learning_rate

        # Warmup
        if self.current_batch < self.config.training.warmup_batches:
            return base_lr * (self.current_batch + 1) / self.config.training.warmup_batches

        # Main schedule
        if self.config.training.lr_schedule == "constant":
            return base_lr
        elif self.config.training.lr_schedule == "cosine":
            if self.config.training.max_batches:
                progress = (self.current_batch - self.config.training.warmup_batches) / \
                          (self.config.training.max_batches - self.config.training.warmup_batches)
                progress = min(1.0, max(0.0, progress))
                return self.config.training.min_lr_ratio * base_lr + \
                       (1 - self.config.training.min_lr_ratio) * base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        elif self.config.training.lr_schedule == "linear":
            if self.config.training.max_batches:
                progress = (self.current_batch - self.config.training.warmup_batches) / \
                          (self.config.training.max_batches - self.config.training.warmup_batches)
                progress = min(1.0, max(0.0, progress))
                return base_lr * (1 - progress * (1 - self.config.training.min_lr_ratio))

        return base_lr

    def _set_lr(self, lr: float) -> None:
        """Set learning rate for optimizer."""
        if isinstance(self.optimizer, HybridOptimizer):
            # HybridOptimizer wraps multiple optimizers
            for opt in self.optimizer.optimizers:
                for param_group in opt.param_groups:
                    param_group['lr'] = lr
        elif isinstance(self.optimizer, list):
            # Legacy list of optimizers
            for opt in self.optimizer:
                for param_group in opt.param_groups:
                    param_group['lr'] = lr
        else:
            # Single optimizer
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def _update_ui(self) -> None:
        """Update all UI elements."""
        if not self.is_training:
            return

        # Update graphs
        self._update_graphs()

        # Update metrics bar
        self._update_metrics_bar()

    def _update_graphs(self) -> None:
        """Update loss graphs."""
        train_losses = self.metrics_tracker.get_windowed_train_losses()
        val_losses = self.metrics_tracker.get_windowed_val_losses()

        # Get plot widgets
        plots = self.query("PlotextPlot")

        if len(plots) >= 2:
            # Train loss graph
            train_plot = plots[0].plt
            train_plot.clear_data()
            train_plot.title("Training Loss")
            train_plot.xlabel("Batch")
            train_plot.ylabel("Loss")

            if train_losses:
                batches, losses = zip(*train_losses)
                train_plot.plot(batches, losses, marker="dot")

            # Val loss graph
            val_plot = plots[1].plt
            val_plot.clear_data()
            val_plot.title("Validation Loss")
            val_plot.xlabel("Batch")
            val_plot.ylabel("Loss")

            if val_losses:
                batches, losses = zip(*val_losses)
                val_plot.plot(batches, losses, marker="dot", color="cyan")

    def _get_memory_usage(self) -> str:
        """Get current memory usage as a formatted string."""
        try:
            import psutil
            import os

            # Get process memory info
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()

            # RSS = Resident Set Size (physical RAM)
            rss_gb = mem_info.rss / 1024**3

            # Try to get macOS-specific memory info
            try:
                # On macOS, use memory_full_info() for more details
                full_info = process.memory_full_info()
                # USS = Unique Set Size (memory unique to this process)
                uss_gb = full_info.uss / 1024**3 if hasattr(full_info, 'uss') else rss_gb
            except:
                uss_gb = rss_gb

            # Get device memory (what PyTorch tracks)
            if torch.cuda.is_available():
                device_memory = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
                return f"{device_memory:.2f}/{rss_gb:.2f} GB"
            # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            #     device_memory = torch.mps.current_allocated_memory() / 1024**3  # GB
            #     return f"{device_memory:.2f}/{rss_gb:.2f} GB (USS:{uss_gb:.2f})"
            else:
                return f"{rss_gb:.2f} GB"
        except Exception as e:
            return f"N/A ({str(e)[:20]})"

    def _update_metrics_bar(self) -> None:
        """Update metrics display bar."""
        elapsed = time.time() - self.start_time
        tokens_per_sec = self.tokens_seen / elapsed if elapsed > 0 else 0

        current_train_loss = self.metrics_tracker.get_current_train_loss()
        current_val_loss = self.metrics_tracker.get_current_val_loss()

        status = "PAUSED" if self.is_paused else "TRAINING"
        status_color = "yellow" if self.is_paused else "green"

        # Get device name
        device_name = str(self.device).upper()

        # Get memory usage
        memory_usage = self._get_memory_usage()

        # Format losses
        train_loss_str = f"{current_train_loss:.4f}" if current_train_loss is not None else "N/A"
        val_loss_str = f"{current_val_loss:.4f}" if current_val_loss is not None else "N/A"

        metrics_text = (
            f"[bold {status_color}]{status}[/bold {status_color}] "
            f"[{device_name}] | "
            f"Mem: {memory_usage} | "
            f"Batch: {self.current_batch} | "
            f"Tokens: {self.tokens_seen:,} | "
            f"Train Loss: {train_loss_str} | "
            f"Val Loss: {val_loss_str} | "
            f"LR: {self.current_lr:.6f} | "
            f"Speed: {tokens_per_sec:.0f} tok/s"
        )

        self.query_one("#metrics-bar", Static).update(metrics_text)

    def _update_inference_display(self) -> None:
        """Update inference output display."""
        output = self.inference_monitor.get_formatted_output(max_length=300)
        self.query_one("#inference-output", Static).update(
            f"[bold cyan]Latest Generation:[/bold cyan]\n{output}"
        )

    def action_toggle_pause(self) -> None:
        """Toggle training pause."""
        self.is_paused = not self.is_paused

    def action_save_checkpoint(self) -> None:
        """Manually save a checkpoint."""
        self._save_checkpoint(create_numbered=True)
        self.notify("Checkpoint saved!")

    def action_generate_sample(self) -> None:
        """Manually generate a sample."""
        self.inference_monitor.generate()
        self._update_inference_display()
        self.notify("Generated new sample!")

    def action_quit_training(self) -> None:
        """Quit and save."""
        self._save_checkpoint()
        self.should_quit = True
        self.exit()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train GPT models with interactive TUI"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to training configuration YAML file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    # Load config
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)

        config = load_training_config(config_path)
    else:
        # TODO: Interactive config selection
        print("Error: --config is required (interactive selection not yet implemented)", file=sys.stderr)
        sys.exit(1)

    # Check for resume
    resume_checkpoint = None
    resume_info = None

    if args.resume:
        resume_checkpoint = Path(args.resume)
        if not resume_checkpoint.exists():
            print(f"Error: Checkpoint not found: {resume_checkpoint}", file=sys.stderr)
            sys.exit(1)

        # Load checkpoint info
        checkpoint_manager = CheckpointManager(config.name, config.checkpointing.output_dir)
        metadata = checkpoint_manager.get_checkpoint_info(resume_checkpoint)
        resume_info = {
            'batch': metadata.batch_num,
            'tokens': metadata.tokens_seen,
            'train_loss': metadata.train_loss,
            'val_loss': metadata.val_loss
        }

    # Determine device
    if torch.cuda.is_available():
        device_info = f"CUDA ({torch.cuda.get_device_name(0)})"
    # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     device_info = "MPS (Apple Silicon)"
    else:
        device_info = "CPU"

    # Show config review (simple terminal version for now)
    print("\n" + "="*60)
    print("Training Configuration Review")
    print("="*60)
    print(f"\nModel: {config.name}")
    print(f"Device: {device_info}")
    print(f"Optimizer: {config.training.optimizer}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Batch size: {config.data.batch_size}")

    if resume_info:
        print(f"\nResuming from batch {resume_info['batch']}")

    print("\nPress ENTER to start training...")
    input()

    # Run training app
    app = TrainingApp(config, resume_checkpoint)
    app.run()


if __name__ == "__main__":
    main()
