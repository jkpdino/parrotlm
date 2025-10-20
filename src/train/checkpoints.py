"""Checkpoint management for training runs."""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import TrainingRunConfig, save_training_config


@dataclass
class CheckpointMetadata:
    """Metadata stored in a checkpoint."""

    batch_num: int
    tokens_seen: int
    train_loss: Optional[float]
    val_loss: Optional[float]
    learning_rate: float
    timestamp: str


class CheckpointManager:
    """
    Manages model checkpoints during training.

    Features:
    - Always saves to 'last_checkpoint.pt'
    - Creates numbered checkpoints every N batches
    - Automatically cleans up old numbered checkpoints (keeps last N)
    - Stores complete training state for resumption
    - Preserves RNG state for reproducibility
    """

    def __init__(
        self,
        run_name: str,
        output_dir: str | Path = "training_runs",
        keep_last_n: int = 10,
        save_every_n_batches: int = 100
    ):
        """
        Initialize checkpoint manager.

        Args:
            run_name: Name of the training run
            output_dir: Base directory for training runs
            keep_last_n: Number of numbered checkpoints to keep
            save_every_n_batches: Frequency of numbered checkpoints
        """
        self.run_name = run_name
        self.output_dir = Path(output_dir)
        self.keep_last_n = keep_last_n
        self.save_every_n_batches = save_every_n_batches

        # Create run directory
        self.run_dir = self.output_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint paths
        self.last_checkpoint_path = self.run_dir / "last_checkpoint.pt"

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch_num: int,
        tokens_seen: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        learning_rate: float = 0.0,
        config: Optional[TrainingRunConfig] = None,
        loss_history: Optional[List[Tuple[int, float]]] = None,
        create_numbered: bool = False
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer to save
            batch_num: Current batch number
            tokens_seen: Total tokens processed
            train_loss: Current training loss
            val_loss: Current validation loss
            learning_rate: Current learning rate
            config: Training configuration
            loss_history: History of (batch, loss) tuples
            create_numbered: Whether to also create a numbered checkpoint

        Returns:
            Path to the saved checkpoint
        """
        import datetime

        # Prepare checkpoint data
        checkpoint = {
            'batch_num': batch_num,
            'tokens_seen': tokens_seen,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': learning_rate,
            'timestamp': datetime.datetime.now().isoformat(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'rng_state': {
                'python': torch.get_rng_state(),
                'numpy': torch.random.get_rng_state() if torch.cuda.is_available() else None,
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            },
            'loss_history': loss_history or [],
        }

        # Save config if provided
        if config is not None:
            config_path = self.run_dir / "config.yml"
            save_training_config(config, config_path)

        # Always save to last_checkpoint.pt
        torch.save(checkpoint, self.last_checkpoint_path)

        # Optionally create numbered checkpoint
        numbered_path = None
        if create_numbered:
            numbered_path = self.run_dir / f"checkpoint_{batch_num}.pt"
            torch.save(checkpoint, numbered_path)

            # Cleanup old numbered checkpoints
            self._cleanup_old_checkpoints()

        return numbered_path if numbered_path else self.last_checkpoint_path

    def load(
        self,
        checkpoint_path: Optional[str | Path] = None,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file. If None, loads last_checkpoint.pt
            model: Model to load state into (optional)
            optimizer: Optimizer to load state into (optional)
            device: Device to load checkpoint on

        Returns:
            Dictionary with checkpoint data
        """
        if checkpoint_path is None:
            checkpoint_path = self.last_checkpoint_path
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state if model provided
        if model is not None:
            model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state if optimizer provided
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore RNG state
        if 'rng_state' in checkpoint:
            rng_state = checkpoint['rng_state']
            if rng_state.get('python') is not None:
                # Ensure RNG state is ByteTensor
                python_state = rng_state['python']
                if not isinstance(python_state, torch.ByteTensor):
                    python_state = python_state.byte()
                torch.set_rng_state(python_state)
            if rng_state.get('numpy') is not None:
                torch.random.set_rng_state(rng_state['numpy'])
            if rng_state.get('cuda') is not None and torch.cuda.is_available():
                # Ensure CUDA RNG states are ByteTensors
                cuda_states = rng_state['cuda']
                if isinstance(cuda_states, list):
                    cuda_states = [s.byte() if not isinstance(s, torch.ByteTensor) else s for s in cuda_states]
                torch.cuda.set_rng_state_all(cuda_states)

        return checkpoint

    def resume_from_checkpoint(
        self,
        checkpoint_path: Optional[str | Path],
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: Optional[torch.device] = None
    ) -> Tuple[int, int, List[Tuple[int, float]]]:
        """
        Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file. If None, uses last_checkpoint.pt
            model: Model to load state into
            optimizer: Optimizer to load state into
            device: Device to load checkpoint on

        Returns:
            Tuple of (batch_num, tokens_seen, loss_history)
        """
        checkpoint = self.load(checkpoint_path, model, optimizer, device)

        batch_num = checkpoint.get('batch_num', 0)
        tokens_seen = checkpoint.get('tokens_seen', 0)
        loss_history = checkpoint.get('loss_history', [])

        return batch_num, tokens_seen, loss_history

    def get_latest_checkpoint(self) -> Optional[Path]:
        """
        Get the path to the latest checkpoint.

        Returns:
            Path to latest checkpoint, or None if no checkpoints exist
        """
        if self.last_checkpoint_path.exists():
            return self.last_checkpoint_path

        # Look for numbered checkpoints
        numbered_checkpoints = self.get_numbered_checkpoints()
        if numbered_checkpoints:
            return numbered_checkpoints[-1]

        return None

    def get_numbered_checkpoints(self) -> List[Path]:
        """
        Get all numbered checkpoint paths, sorted by batch number.

        Returns:
            List of paths to numbered checkpoints
        """
        checkpoints = list(self.run_dir.glob("checkpoint_*.pt"))

        # Sort by batch number
        def extract_batch_num(path: Path) -> int:
            try:
                # Extract number from "checkpoint_123.pt"
                return int(path.stem.split('_')[1])
            except (IndexError, ValueError):
                return 0

        checkpoints.sort(key=extract_batch_num)
        return checkpoints

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old numbered checkpoints, keeping only the last N."""
        if self.keep_last_n <= 0:
            return

        checkpoints = self.get_numbered_checkpoints()

        # Remove old checkpoints if we have too many
        if len(checkpoints) > self.keep_last_n:
            to_remove = checkpoints[:-self.keep_last_n]
            for checkpoint_path in to_remove:
                try:
                    checkpoint_path.unlink()
                except Exception as e:
                    print(f"Warning: Failed to remove old checkpoint {checkpoint_path}: {e}")

    def checkpoint_exists(self) -> bool:
        """
        Check if any checkpoint exists for this run.

        Returns:
            True if at least one checkpoint exists
        """
        return self.get_latest_checkpoint() is not None

    def get_checkpoint_info(self, checkpoint_path: Optional[str | Path] = None) -> CheckpointMetadata:
        """
        Get metadata from a checkpoint without loading the full state.

        Args:
            checkpoint_path: Path to checkpoint file. If None, uses last_checkpoint.pt

        Returns:
            CheckpointMetadata object
        """
        if checkpoint_path is None:
            checkpoint_path = self.last_checkpoint_path
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        return CheckpointMetadata(
            batch_num=checkpoint.get('batch_num', 0),
            tokens_seen=checkpoint.get('tokens_seen', 0),
            train_loss=checkpoint.get('train_loss'),
            val_loss=checkpoint.get('val_loss'),
            learning_rate=checkpoint.get('learning_rate', 0.0),
            timestamp=checkpoint.get('timestamp', 'unknown')
        )

    def delete_all_checkpoints(self) -> None:
        """Delete all checkpoints for this run. Use with caution!"""
        if self.run_dir.exists():
            shutil.rmtree(self.run_dir)
            self.run_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        """String representation of checkpoint manager."""
        return (
            f"CheckpointManager(\n"
            f"  run_name='{self.run_name}',\n"
            f"  run_dir='{self.run_dir}',\n"
            f"  keep_last_n={self.keep_last_n},\n"
            f"  checkpoint_exists={self.checkpoint_exists()}\n"
            f")"
        )
