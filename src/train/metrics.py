"""Metrics tracking and visualization for training."""

import csv
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Tuple


@dataclass
class MetricPoint:
    """A single metric measurement."""

    batch: int
    tokens_seen: int
    train_loss: Optional[float]
    val_loss: Optional[float]
    learning_rate: float
    timestamp: float


class MetricsTracker:
    """
    Track and log training metrics with CSV persistence.

    Features:
    - Tracks train/validation losses with batch numbers
    - CSV logging with append mode for resume support
    - Windowed history for efficient graph rendering
    - Running statistics (mean, min, max, trends)
    """

    def __init__(
        self,
        run_dir: str | Path,
        window_size: int = 1000,
        resume: bool = False
    ):
        """
        Initialize metrics tracker.

        Args:
            run_dir: Directory for this training run
            window_size: Number of recent points to keep in memory for graphs
            resume: If True, load existing metrics from CSV
        """
        self.run_dir = Path(run_dir)
        self.window_size = window_size

        # Create run directory
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # CSV file path
        self.csv_path = self.run_dir / "metrics.csv"

        # In-memory storage (windowed for efficiency)
        self.train_losses: Deque[Tuple[int, float]] = deque(maxlen=window_size)
        self.val_losses: Deque[Tuple[int, float]] = deque(maxlen=window_size)
        self.learning_rates: Deque[Tuple[int, float]] = deque(maxlen=window_size)

        # Full history (for statistics)
        self.all_train_losses: List[float] = []
        self.all_val_losses: List[float] = []

        # Statistics
        self.best_train_loss: Optional[float] = None
        self.best_val_loss: Optional[float] = None
        self.total_batches: int = 0
        self.total_tokens: int = 0

        # Initialize or load CSV
        if resume and self.csv_path.exists():
            self._load_from_csv()
        else:
            self._create_csv()

    def _create_csv(self) -> None:
        """Create a new CSV file with headers."""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'batch',
                'tokens_seen',
                'train_loss',
                'val_loss',
                'learning_rate',
                'timestamp'
            ])

    def _load_from_csv(self) -> None:
        """Load existing metrics from CSV file."""
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                batch = int(row['batch'])
                tokens_seen = int(row['tokens_seen'])
                train_loss = float(row['train_loss']) if row['train_loss'] else None
                val_loss = float(row['val_loss']) if row['val_loss'] else None
                learning_rate = float(row['learning_rate'])

                # Add to windowed storage
                if train_loss is not None:
                    self.train_losses.append((batch, train_loss))
                    self.all_train_losses.append(train_loss)

                if val_loss is not None:
                    self.val_losses.append((batch, val_loss))
                    self.all_val_losses.append(val_loss)

                self.learning_rates.append((batch, learning_rate))

                # Update statistics
                self.total_batches = max(self.total_batches, batch)
                self.total_tokens = max(self.total_tokens, tokens_seen)

        # Calculate best losses
        if self.all_train_losses:
            self.best_train_loss = min(self.all_train_losses)
        if self.all_val_losses:
            self.best_val_loss = min(self.all_val_losses)

    def log(
        self,
        batch: int,
        tokens_seen: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        learning_rate: float = 0.0
    ) -> None:
        """
        Log a metric point.

        Args:
            batch: Current batch number
            tokens_seen: Total tokens processed
            train_loss: Training loss (if available)
            val_loss: Validation loss (if available)
            learning_rate: Current learning rate
        """
        timestamp = time.time()

        # Update in-memory storage
        if train_loss is not None:
            self.train_losses.append((batch, train_loss))
            self.all_train_losses.append(train_loss)

            # Update best
            if self.best_train_loss is None or train_loss < self.best_train_loss:
                self.best_train_loss = train_loss

        if val_loss is not None:
            self.val_losses.append((batch, val_loss))
            self.all_val_losses.append(val_loss)

            # Update best
            if self.best_val_loss is None or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

        self.learning_rates.append((batch, learning_rate))

        # Update totals
        self.total_batches = max(self.total_batches, batch)
        self.total_tokens = max(self.total_tokens, tokens_seen)

        # Append to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                batch,
                tokens_seen,
                train_loss if train_loss is not None else '',
                val_loss if val_loss is not None else '',
                learning_rate,
                timestamp
            ])

    def get_windowed_train_losses(self) -> List[Tuple[int, float]]:
        """
        Get recent training losses for graphing.

        Returns:
            List of (batch, loss) tuples
        """
        return list(self.train_losses)

    def get_windowed_val_losses(self) -> List[Tuple[int, float]]:
        """
        Get recent validation losses for graphing.

        Returns:
            List of (batch, loss) tuples
        """
        return list(self.val_losses)

    def get_windowed_learning_rates(self) -> List[Tuple[int, float]]:
        """
        Get recent learning rates for graphing.

        Returns:
            List of (batch, lr) tuples
        """
        return list(self.learning_rates)

    def get_current_train_loss(self) -> Optional[float]:
        """
        Get the most recent training loss.

        Returns:
            Most recent training loss, or None if no losses recorded
        """
        if not self.train_losses:
            return None
        return self.train_losses[-1][1]

    def get_current_val_loss(self) -> Optional[float]:
        """
        Get the most recent validation loss.

        Returns:
            Most recent validation loss, or None if no losses recorded
        """
        if not self.val_losses:
            return None
        return self.val_losses[-1][1]

    def get_average_train_loss(self, last_n: Optional[int] = None) -> Optional[float]:
        """
        Get average training loss.

        Args:
            last_n: If specified, average over last N points. Otherwise, all points.

        Returns:
            Average training loss, or None if no losses
        """
        if not self.train_losses:
            return None

        losses = [loss for _, loss in self.train_losses]
        if last_n:
            losses = losses[-last_n:]

        return sum(losses) / len(losses) if losses else None

    def get_average_val_loss(self, last_n: Optional[int] = None) -> Optional[float]:
        """
        Get average validation loss.

        Args:
            last_n: If specified, average over last N points. Otherwise, all points.

        Returns:
            Average validation loss, or None if no losses
        """
        if not self.val_losses:
            return None

        losses = [loss for _, loss in self.val_losses]
        if last_n:
            losses = losses[-last_n:]

        return sum(losses) / len(losses) if losses else None

    def get_statistics(self) -> dict:
        """
        Get comprehensive statistics about metrics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_batches': self.total_batches,
            'total_tokens': self.total_tokens,
            'best_train_loss': self.best_train_loss,
            'best_val_loss': self.best_val_loss,
            'current_train_loss': self.get_current_train_loss(),
            'current_val_loss': self.get_current_val_loss(),
            'avg_train_loss': self.get_average_train_loss(),
            'avg_val_loss': self.get_average_val_loss(),
            'train_loss_count': len(self.all_train_losses),
            'val_loss_count': len(self.all_val_losses),
        }

        return stats

    def export_full_history(self, output_path: str | Path) -> None:
        """
        Export full metrics history to a new CSV file.

        Args:
            output_path: Path to export CSV
        """
        import shutil
        shutil.copy2(self.csv_path, output_path)

    def __repr__(self) -> str:
        """String representation of metrics tracker."""
        stats = self.get_statistics()
        return (
            f"MetricsTracker(\n"
            f"  total_batches={stats['total_batches']},\n"
            f"  total_tokens={stats['total_tokens']:,},\n"
            f"  train_losses={stats['train_loss_count']},\n"
            f"  val_losses={stats['val_loss_count']},\n"
            f"  best_train_loss={stats['best_train_loss']:.4f if stats['best_train_loss'] else 'N/A'},\n"
            f"  best_val_loss={stats['best_val_loss']:.4f if stats['best_val_loss'] else 'N/A'}\n"
            f")"
        )
