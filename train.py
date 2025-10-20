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
import threading
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
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
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_info = "mps (Apple Silicon)"
        else:
            device_info = "cpu"

        # Model info
        model_params = self._estimate_params()

        # Build parameter info string
        if self.config.model.use_moe:
            params_str = f"~{model_params['total']:,} total, ~{model_params['active']:,} active"
        else:
            params_str = f"~{model_params['total']:,}"

        model_info = f"""
[bold cyan]Model Configuration:[/bold cyan]
  Name: {self.config.name}
  Architecture: GPT
  Dimensions: {self.config.model.dim}
  Depth: {self.config.model.depth} layers
  Heads: {self.config.model.n_heads} query, {self.config.model.n_kv_heads} kv (GQA)
  Context: {self.config.model.context_length} tokens
  Vocab: {self.config.model.vocab_size:,}
  Parameters: {params_str}
  Device: {device_info}
"""

        # Add MoE section if enabled
        if self.config.model.use_moe:
            if self.config.model.moe_layers is None:
                moe_layers_str = "all layers"
            else:
                moe_layers_str = f"layers {self.config.model.moe_layers}"

            model_info += f"""
[bold cyan]MoE Configuration:[/bold cyan]
  Enabled: Yes
  Total experts: {self.config.model.num_experts}
  Active per token: {self.config.model.num_experts_active}
  MoE layers: {moe_layers_str}
  Load balance loss coef: {self.config.model.router_aux_loss_coef}
  Router z-loss coef: {self.config.model.router_z_loss_coef}
"""

        model_info += f"""
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

    def _estimate_params(self) -> dict:
        """
        Estimate parameter count.

        Returns:
            Dictionary with 'total', 'active', 'transformer', 'embeddings' keys
            For non-MoE models, 'total' and 'active' are the same
        """
        cfg = self.config.model
        # Token embeddings
        emb = cfg.vocab_size * cfg.dim

        # Calculate attention parameters (same for all blocks)
        attn_params = (
            cfg.dim * cfg.n_heads * (cfg.dim // cfg.n_heads) * 3 +  # QKV projections
            cfg.dim * cfg.dim  # Output projection
        )

        # Calculate FFN or MoE parameters
        if cfg.use_moe:
            # MoE: Calculate per-expert parameters
            hidden_dim = int(2 * cfg.ffn_hidden_mult * cfg.dim / 3)
            params_per_expert = (
                cfg.dim * hidden_dim +  # gate_proj
                cfg.dim * hidden_dim +  # up_proj
                hidden_dim * cfg.dim    # down_proj
            )
            router_params = cfg.dim * cfg.num_experts

            # Determine which layers use MoE
            if cfg.moe_layers is None:
                num_moe_layers = cfg.depth
                num_dense_layers = 0
            else:
                num_moe_layers = len(cfg.moe_layers)
                num_dense_layers = cfg.depth - num_moe_layers

            # Dense FFN parameters (for non-MoE layers if any)
            dense_ffn_params = cfg.dim * (int(2 * cfg.ffn_hidden_mult * cfg.dim / 3)) * 3

            # Total MoE parameters
            moe_ffn_params = num_moe_layers * (cfg.num_experts * params_per_expert + router_params)
            dense_total_ffn = num_dense_layers * dense_ffn_params

            # Active MoE parameters (only k experts active per token)
            active_moe_ffn_params = num_moe_layers * (cfg.num_experts_active * params_per_expert + router_params)

            # Norms (2 per block: pre-attn and pre-ffn)
            norm_params = cfg.dim * 2 * cfg.depth

            # Total transformer params
            transformer_total = (
                attn_params * cfg.depth +
                moe_ffn_params + dense_total_ffn +
                norm_params
            )

            # Active transformer params (for MoE, only count active experts)
            transformer_active = (
                attn_params * cfg.depth +
                active_moe_ffn_params + dense_total_ffn +
                norm_params
            )

            return {
                'embeddings': emb,
                'transformer': transformer_total,
                'transformer_active': transformer_active,
                'total': emb + transformer_total,
                'active': emb + transformer_active
            }
        else:
            # Dense FFN
            ffn_params = cfg.dim * (int(2 * cfg.ffn_hidden_mult * cfg.dim / 3)) * 3
            norm_params = cfg.dim * 2

            per_block = attn_params + ffn_params + norm_params
            transformer = per_block * cfg.depth
            total = emb + transformer

            return {
                'embeddings': emb,
                'transformer': transformer,
                'transformer_active': transformer,
                'total': total,
                'active': total
            }


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
        height: 6;
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
        resume_checkpoint: Optional[Path] = None,
        enable_profiling: bool = False,
        profile_dir: str = "./profiling_logs",
        profile_schedule: Optional[dict] = None
    ):
        super().__init__()
        self.config = config
        self.resume_checkpoint = resume_checkpoint

        # Training state (accessed by both UI and training thread)
        self.is_paused = False
        self.is_training = False
        self.should_quit = False
        self.training_thread: Optional[threading.Thread] = None
        self.state_lock = threading.Lock()  # Lock for thread-safe state access

        # Profiling configuration
        self.enable_profiling = enable_profiling
        self.profile_dir = Path(profile_dir)
        self.profile_schedule = profile_schedule or {
            'wait': 5,
            'warmup': 2,
            'active': 3,
            'repeat': 1
        }
        self.profiler = None

        # Components (will be initialized in on_mount)
        # Priority: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Mixed precision training (CUDA only for now)
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        self.model: Optional[GPT] = None
        self.model_is_compiled = False  # Track if torch.compile succeeded
        self.optimizer = None
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.metrics_tracker: Optional[MetricsTracker] = None
        self.inference_monitor: Optional[InferenceMonitor] = None
        self.train_loader: Optional[DocumentBatchLoader] = None
        self.val_loader: Optional[DocumentBatchLoader] = None

        # Training progress
        self.current_batch = 0
        self.tokens_seen = 0
        self.tokens_seen_at_start = 0  # Track tokens at session start for accurate tok/s
        self.current_lr = config.training.learning_rate
        self.start_time = time.time()

        # Data prefetching (load next batch while GPU processes current)
        self.next_batch = None
        self.data_queue = None  # Will hold prefetched batch
        self.prefetch_thread = None

        # Performance timing diagnostics
        self.timing_samples = 100  # Average over last N samples
        self.timings = {
            'data_loading': [],
            'model_forward': [],
            'loss_computation': [],
            'backward': [],
            'grad_check': [],
            'grad_clip': [],
            'optimizer_step': [],
            'lr_update': [],
            'metrics_log': [],
            'validation': [],
            'inference': [],
            'checkpoint': [],
            'total_step': []
        }

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

        # Start training in a background thread (runs at full GPU speed)
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()

        # Update UI every second (on main thread)
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
            use_gradient_checkpointing=self.config.model.use_gradient_checkpointing,
            # MoE parameters
            use_moe=self.config.model.use_moe,
            num_experts=self.config.model.num_experts,
            num_experts_active=self.config.model.num_experts_active,
            moe_layers=self.config.model.moe_layers,
            router_aux_loss_coef=self.config.model.router_aux_loss_coef,
            router_z_loss_coef=self.config.model.router_z_loss_coef
        ).to(self.device)

        # Compile model for 2-3x speedup
        if torch.cuda.is_available():
            print("Compiling model with torch.compile() - this takes ~1 minute...")
            try:
                self.model = torch.compile(self.model)
                self.model_is_compiled = True
                print("✓ Model compilation successful!")
            except Exception as e:
                print(f"Warning: torch.compile failed ({e}), continuing without compilation")
                self.model_is_compiled = False
        elif self.device.type == "mps":
            # Try compiling on MPS with workaround for macOS multiprocessing issue
            print("Compiling model for MPS (using eager backend to avoid subprocess issues)...")
            try:
                import multiprocessing
                try:
                    multiprocessing.set_start_method('spawn', force=True)
                except RuntimeError:
                    pass  # Already set
                # Use eager backend which avoids the inductor subprocess issue
                self.model = torch.compile(self.model, backend="aot_eager")
                self.model_is_compiled = True
                print("✓ Model compilation successful!")
            except Exception as e:
                print(f"Warning: torch.compile failed on MPS ({e}), continuing without compilation")
                self.model_is_compiled = False
        else:
            print(f"Skipping torch.compile on CPU (not beneficial)")
            self.model_is_compiled = False

        # Create optimizer
        if self.config.training.optimizer == "muon":
            self.optimizer = create_hybrid_optimizer(
                self.model,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        else:
            # Use fused AdamW for 2-3x speedup on CUDA
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                fused=torch.cuda.is_available()  # Fused kernel for CUDA
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
            # Track starting tokens for accurate tok/s calculation
            self.tokens_seen_at_start = self.tokens_seen

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

        # Initialize profiler if enabled
        if self.enable_profiling:
            self.profile_dir.mkdir(parents=True, exist_ok=True)

            # Determine activities to profile based on device
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)

            # Create profiler schedule
            profiler_schedule = schedule(
                wait=self.profile_schedule['wait'],
                warmup=self.profile_schedule['warmup'],
                active=self.profile_schedule['active'],
                repeat=self.profile_schedule['repeat']
            )

            # Create profiler with TensorBoard output
            self.profiler = profile(
                activities=activities,
                schedule=profiler_schedule,
                on_trace_ready=tensorboard_trace_handler(str(self.profile_dir)),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True
            )
            self.profiler.__enter__()

        # Start training
        self.is_training = True

    def _do_single_training_step(self) -> None:
        """Execute one training step (synchronous)."""
        step_start = time.time()

        if not self.is_training or self.is_paused or self.should_quit:
            return

        # Check if we've reached max batches/tokens
        if self.config.training.max_batches and self.current_batch >= self.config.training.max_batches:
            with self.state_lock:
                self.is_training = False
            self._save_checkpoint()
            self._cleanup_profiler()
            return

        if self.config.training.max_tokens and self.tokens_seen >= self.config.training.max_tokens:
            with self.state_lock:
                self.is_training = False
            self._save_checkpoint()
            self._cleanup_profiler()
            return

        # Get batch - just load synchronously for now
        # TODO: Implement proper async prefetching with background thread
        data_start = time.time()
        with record_function("data_loading"):
            try:
                tokens, masks, labels = next(self.train_loader)
            except StopIteration:
                # Epoch complete, reset loader
                self.train_loader.reset()
                tokens, masks, labels = next(self.train_loader)
        data_time = time.time() - data_start

        # Forward pass with mixed precision
        self.model.train()

        # Model inference
        model_forward_start = time.time()
        with record_function("forward"):
            # Use autocast for CUDA mixed precision training
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=self.use_amp):
                result = self.model(tokens, masks)
        model_forward_time = time.time() - model_forward_start

        # Handle MoE models (return tuple) vs standard models (return tensor)
        loss_start = time.time()
        if isinstance(result, tuple):
            logits, aux_losses = result
        else:
            logits = result
            aux_losses = None

        # Compute main loss
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=self.use_amp):
            loss = F.cross_entropy(
                logits.view(-1, self.config.model.vocab_size),
                labels.view(-1),
                ignore_index=0
            )

            # Add auxiliary losses if using MoE
            if aux_losses is not None:
                for aux_loss_name, aux_loss_value in aux_losses.items():
                    loss = loss + aux_loss_value
        loss_compute_time = time.time() - loss_start

        # Check for NaN/Inf
        if not torch.isfinite(loss):
            with self.state_lock:
                self.is_training = False
            # Safely update UI from training thread
            self.call_from_thread(
                self.query_one("#metrics-bar", Static).update,
                "[bold red]TRAINING STOPPED: Loss became NaN/Inf. Try lowering learning rate.[/bold red]"
            )
            return

        # Backward pass with gradient scaling for mixed precision
        backward_start = time.time()
        with record_function("backward"):
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        backward_time = time.time() - backward_start

        # Check for gradient NaN/Inf after backward (disabled by default - loss check above catches most issues)
        grad_check_time = 0.0
        # GPU-only NaN check (much faster - no CPU sync):
        # Uncomment to enable (checks every 1000 batches, takes ~5-10ms instead of 143ms):
        # if self.current_batch % 1000 == 0:
        #     check_start = time.time()
        #     # Flatten all gradients into a single tensor (stays on GPU)
        #     grads = torch.cat([p.grad.flatten() for p in self.model.parameters() if p.grad is not None])
        #     # Single GPU check for all gradients (no CPU sync until .item())
        #     has_nan_grad = not torch.isfinite(grads).all().item()
        #     grad_check_time = time.time() - check_start
        #
        #     if has_nan_grad:
        #         with self.state_lock:
        #             self.is_training = False
        #         self.call_from_thread(
        #             self.query_one("#metrics-bar", Static).update,
        #             "[bold red]TRAINING STOPPED: Gradients became NaN/Inf. Try lowering learning rate.[/bold red]"
        #         )
        #         return

        # Gradient clipping
        grad_clip_time = 0.0
        if self.config.training.grad_clip:
            clip_start = time.time()
            with record_function("gradient_clipping"):
                if self.use_amp:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.grad_clip
                )
            grad_clip_time = time.time() - clip_start

        # Optimizer step with gradient scaling
        optim_start = time.time()
        with record_function("optimizer_step"):
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
        optim_time = time.time() - optim_start

        # Update progress
        self.current_batch += 1
        self.tokens_seen += tokens.numel()

        # Update learning rate
        lr_start = time.time()
        self.current_lr = self._get_lr()
        self._set_lr(self.current_lr)
        lr_time = time.time() - lr_start

        # Log metrics (only every 500 batches to reduce overhead - CSV write is slow)
        metrics_time = 0.0
        if self.current_batch % 500 == 0:
            metrics_start = time.time()
            self.metrics_tracker.log(
                batch=self.current_batch,
                tokens_seen=self.tokens_seen,
                train_loss=loss.item(),
                learning_rate=self.current_lr
            )
            metrics_time = time.time() - metrics_start

        # Periodic tasks
        checkpoint_time = 0.0
        if self.current_batch % self.config.checkpointing.save_every_n_batches == 0:
            ckpt_start = time.time()
            self._save_checkpoint(create_numbered=True)
            checkpoint_time = time.time() - ckpt_start

        validation_time = 0.0
        if self.current_batch % self.config.monitoring.validate_every_n_batches == 0:
            val_start = time.time()
            self._run_validation()
            validation_time = time.time() - val_start

        inference_time = 0.0
        if self.current_batch % self.config.monitoring.inference_every_n_batches == 0:
            # Generate inference and update display safely from training thread
            inf_start = time.time()
            self.inference_monitor.generate()
            self.call_from_thread(self._update_inference_display)
            inference_time = time.time() - inf_start

        # Profiler step
        # if self.profiler is not None:
        #     self.profiler.step()

        # Record timings
        total_step_time = time.time() - step_start
        self._record_timing('data_loading', data_time)
        self._record_timing('model_forward', model_forward_time)
        self._record_timing('loss_computation', loss_compute_time)
        self._record_timing('backward', backward_time)
        self._record_timing('grad_check', grad_check_time)
        self._record_timing('grad_clip', grad_clip_time)
        self._record_timing('optimizer_step', optim_time)
        self._record_timing('lr_update', lr_time)
        self._record_timing('metrics_log', metrics_time)
        self._record_timing('validation', validation_time)
        self._record_timing('inference', inference_time)
        self._record_timing('checkpoint', checkpoint_time)
        self._record_timing('total_step', total_step_time)

    def _record_timing(self, category: str, duration: float) -> None:
        """Record timing for a category, keeping only last N samples."""
        if duration > 0:  # Only record non-zero times
            self.timings[category].append(duration)
            # Keep only last N samples
            if len(self.timings[category]) > self.timing_samples:
                self.timings[category].pop(0)

    def _get_avg_timing(self, category: str) -> float:
        """Get average timing for a category in milliseconds."""
        if not self.timings[category]:
            return 0.0
        return sum(self.timings[category]) / len(self.timings[category]) * 1000  # Convert to ms

    def _training_loop(self) -> None:
        """Main training loop that runs in a background thread.

        This runs continuously at full GPU speed while the UI updates
        independently on the main thread.
        """
        while True:
            with self.state_lock:
                if self.should_quit:
                    break
                if self.is_paused or not self.is_training:
                    time.sleep(0.01)  # Sleep when paused to avoid busy-waiting
                    continue

            # Run a single training step at full GPU speed
            self._do_single_training_step()

            # No sleep needed - let GPU run at full speed!

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

                    result = self.model(tokens, masks)

                    # Handle MoE models (return tuple) vs standard models (return tensor)
                    if isinstance(result, tuple):
                        logits, aux_losses = result
                    else:
                        logits = result
                        aux_losses = None

                    loss = F.cross_entropy(
                        logits.view(-1, self.config.model.vocab_size),
                        labels.view(-1),
                        ignore_index=0
                    )

                    # Add auxiliary losses if using MoE
                    if aux_losses is not None:
                        for aux_loss_name, aux_loss_value in aux_losses.items():
                            loss = loss + aux_loss_value

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
                # Handle edge case where warmup_batches >= max_batches
                decay_batches = max(1, self.config.training.max_batches - self.config.training.warmup_batches)
                progress = (self.current_batch - self.config.training.warmup_batches) / decay_batches
                progress = min(1.0, max(0.0, progress))
                return self.config.training.min_lr_ratio * base_lr + \
                       (1 - self.config.training.min_lr_ratio) * base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        elif self.config.training.lr_schedule == "linear":
            if self.config.training.max_batches:
                # Handle edge case where warmup_batches >= max_batches
                decay_batches = max(1, self.config.training.max_batches - self.config.training.warmup_batches)
                progress = (self.current_batch - self.config.training.warmup_batches) / decay_batches
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
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_memory = torch.mps.current_allocated_memory() / 1024**3  # GB
                return f"{device_memory:.2f}/{rss_gb:.2f} GB"
            else:
                return f"{rss_gb:.2f} GB"
        except Exception as e:
            return f"N/A ({str(e)[:20]})"

    def _update_metrics_bar(self) -> None:
        """Update metrics display bar."""
        elapsed = time.time() - self.start_time
        tokens_this_session = self.tokens_seen - self.tokens_seen_at_start
        tokens_per_sec = tokens_this_session / elapsed if elapsed > 0 else 0

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

        # Get timing breakdown
        avg_total = self._get_avg_timing('total_step')
        avg_data = self._get_avg_timing('data_loading')
        avg_model_fwd = self._get_avg_timing('model_forward')
        avg_loss = self._get_avg_timing('loss_computation')
        avg_backward = self._get_avg_timing('backward')
        avg_grad_check = self._get_avg_timing('grad_check')
        avg_grad_clip = self._get_avg_timing('grad_clip')
        avg_optim = self._get_avg_timing('optimizer_step')
        avg_lr = self._get_avg_timing('lr_update')
        avg_metrics = self._get_avg_timing('metrics_log')
        avg_val = self._get_avg_timing('validation')
        avg_inf = self._get_avg_timing('inference')

        # Calculate accounted time and overhead
        accounted = avg_data + avg_model_fwd + avg_loss + avg_backward + avg_grad_clip + avg_optim + avg_lr + avg_metrics
        overhead = avg_total - accounted if avg_total > accounted else 0

        # Build timing string (only show significant times)
        timing_parts = []
        if avg_total > 0:
            timing_parts.append(f"Total:{avg_total:.1f}ms")
        if avg_data > 0:
            timing_parts.append(f"Data:{avg_data:.1f}ms")
        if avg_model_fwd > 0:
            timing_parts.append(f"ModelFwd:{avg_model_fwd:.1f}ms")
        if avg_loss > 0:
            timing_parts.append(f"Loss:{avg_loss:.1f}ms")
        if avg_backward > 0:
            timing_parts.append(f"Bwd:{avg_backward:.1f}ms")
        if avg_grad_check > 1:
            timing_parts.append(f"GradChk:{avg_grad_check:.1f}ms")
        if avg_grad_clip > 0:
            timing_parts.append(f"GradClip:{avg_grad_clip:.1f}ms")
        if avg_optim > 0:
            timing_parts.append(f"Opt:{avg_optim:.1f}ms")
        if avg_lr > 0.5:
            timing_parts.append(f"LR:{avg_lr:.1f}ms")
        if avg_metrics > 0.5:
            timing_parts.append(f"Metrics:{avg_metrics:.1f}ms")
        if overhead > 1:
            timing_parts.append(f"[yellow]Overhead:{overhead:.1f}ms[/yellow]")
        if avg_val > 1:  # Only show if > 1ms
            timing_parts.append(f"Val:{avg_val:.0f}ms")
        if avg_inf > 1:  # Only show if > 1ms
            timing_parts.append(f"Inf:{avg_inf:.0f}ms")

        timing_str = " ".join(timing_parts) if timing_parts else "Measuring..."

        # Build optimization flags
        opt_flags = []
        if self.model_is_compiled:
            opt_flags.append("Compiled✓")
        if self.use_amp:
            opt_flags.append("FP16✓")
        if not self.config.model.use_gradient_checkpointing:
            opt_flags.append("NoGradCkpt✓")
        opt_str = " ".join(opt_flags) if opt_flags else "No optimizations"

        metrics_text = (
            f"[bold {status_color}]{status}[/bold {status_color}] "
            f"[{device_name}] | "
            f"Mem: {memory_usage} | "
            f"Batch: {self.current_batch} | "
            f"Tokens: {self.tokens_seen:,} | "
            f"Train Loss: {train_loss_str} | "
            f"Val Loss: {val_loss_str} | "
            f"LR: {self.current_lr:.6f} | "
            f"Speed: {tokens_per_sec:.0f} tok/s\n"
            f"[dim]Timing: {timing_str} | Opts: {opt_str}[/dim]"
        )

        self.query_one("#metrics-bar", Static).update(metrics_text)

    def _update_inference_display(self) -> None:
        """Update inference output display."""
        output = self.inference_monitor.get_formatted_output(max_length=300)
        self.query_one("#inference-output", Static).update(
            f"[bold cyan]Latest Generation:[/bold cyan]\n{output}"
        )

    def _cleanup_profiler(self) -> None:
        """Cleanup profiler and save memory snapshot if CUDA profiling was enabled."""
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)

            # Save CUDA memory snapshot if available
            if torch.cuda.is_available() and self.enable_profiling:
                try:
                    memory_snapshot_path = self.profile_dir / "cuda_memory_snapshot.pickle"
                    torch.cuda.memory._dump_snapshot(str(memory_snapshot_path))
                    print(f"\nCUDA memory snapshot saved to: {memory_snapshot_path}")
                except Exception as e:
                    print(f"\nWarning: Could not save CUDA memory snapshot: {e}")

            print(f"\nProfiler traces saved to: {self.profile_dir}")
            print(f"View with: tensorboard --logdir={self.profile_dir}")
            self.profiler = None

    def action_toggle_pause(self) -> None:
        """Toggle training pause."""
        with self.state_lock:
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
        # Signal training thread to stop
        with self.state_lock:
            self.should_quit = True

        # Wait for training thread to finish (with timeout)
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5.0)

        # Save final checkpoint and cleanup
        self._save_checkpoint()
        self._cleanup_profiler()
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
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable PyTorch profiler for performance analysis"
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="./profiling_logs",
        help="Directory to save profiling results (default: ./profiling_logs)"
    )
    parser.add_argument(
        "--profile-wait",
        type=int,
        default=5,
        help="Number of steps to skip before profiling (default: 5)"
    )
    parser.add_argument(
        "--profile-warmup",
        type=int,
        default=2,
        help="Number of warmup steps for profiler (default: 2)"
    )
    parser.add_argument(
        "--profile-active",
        type=int,
        default=3,
        help="Number of steps to actively profile (default: 3)"
    )
    parser.add_argument(
        "--profile-repeat",
        type=int,
        default=1,
        help="Number of times to repeat profiling cycle (default: 1)"
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
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_info = "MPS (Apple Silicon)"
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

    # Prepare profiling configuration
    profile_schedule_config = None
    if args.profile:
        profile_schedule_config = {
            'wait': args.profile_wait,
            'warmup': args.profile_warmup,
            'active': args.profile_active,
            'repeat': args.profile_repeat
        }
        print(f"\n[PROFILING ENABLED]")
        print(f"  Output: {args.profile_dir}")
        print(f"  Schedule: wait={args.profile_wait}, warmup={args.profile_warmup}, active={args.profile_active}, repeat={args.profile_repeat}")

    # Run training app
    app = TrainingApp(
        config,
        resume_checkpoint,
        enable_profiling=args.profile,
        profile_dir=args.profile_dir,
        profile_schedule=profile_schedule_config
    )
    app.run()


if __name__ == "__main__":
    main()
