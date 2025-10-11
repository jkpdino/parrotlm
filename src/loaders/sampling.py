"""Sampling strategies for data loaders.

Implements Strategy Pattern to separate sampling concerns from data loading.
"""

import random
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List


class SamplingStrategy(ABC):
    """Abstract base class for sampling strategies."""

    def __init__(self, random_seed: int = 42):
        """
        Initialize sampling strategy.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self._random = random.Random(random_seed)

    @abstractmethod
    def apply(
        self,
        iterator: Iterator[Dict],
        offset: int = 0,
        limit: int | None = None
    ) -> Iterator[Dict]:
        """
        Apply sampling strategy to data iterator.

        Args:
            iterator: Input data iterator
            offset: Number of records to skip
            limit: Maximum number of records to return

        Yields:
            Sampled records
        """
        pass


class SequentialSampling(SamplingStrategy):
    """
    Sequential sampling strategy - streams data in order.

    Most memory-efficient as it doesn't materialize the entire dataset.
    """

    def apply(
        self,
        iterator: Iterator[Dict],
        offset: int = 0,
        limit: int | None = None
    ) -> Iterator[Dict]:
        """
        Apply sequential sampling with offset and limit.

        Args:
            iterator: Input data iterator
            offset: Number of records to skip from beginning
            limit: Maximum number of records to yield

        Yields:
            Records in sequential order after offset/limit
        """
        # Skip offset records
        for _ in range(offset):
            try:
                next(iterator)
            except StopIteration:
                return

        # Yield up to limit records
        if limit is not None:
            for i, record in enumerate(iterator):
                if i >= limit:
                    break
                yield record
        else:
            yield from iterator


class RandomSampling(SamplingStrategy):
    """
    Random sampling strategy - shuffles data before yielding.

    Note: Requires materializing the entire dataset into memory for shuffling.
    For truly large datasets that don't fit in memory, consider using
    reservoir sampling or pre-shuffled data.
    """

    def apply(
        self,
        iterator: Iterator[Dict],
        offset: int = 0,
        limit: int | None = None
    ) -> Iterator[Dict]:
        """
        Apply random sampling with offset and limit.

        Args:
            iterator: Input data iterator
            offset: Number of records to skip from shuffled data
            limit: Maximum number of records to yield

        Yields:
            Records in random order after offset/limit
        """
        # Materialize all data (required for shuffling)
        records = list(iterator)

        # Shuffle using instance random generator (thread-safe)
        self._random.shuffle(records)

        # Apply offset and limit
        end = len(records)
        if limit is not None:
            end = min(offset + limit, len(records))

        for i in range(offset, end):
            yield records[i]


def create_sampling_strategy(
    strategy_name: str,
    random_seed: int = 42
) -> SamplingStrategy:
    """
    Factory function to create sampling strategy by name.

    Args:
        strategy_name: Name of strategy ("sequential" or "random")
        random_seed: Random seed for reproducibility

    Returns:
        Appropriate SamplingStrategy instance

    Raises:
        ValueError: If strategy name is unknown
    """
    strategies = {
        "sequential": SequentialSampling,
        "random": RandomSampling,
    }

    strategy_class = strategies.get(strategy_name.lower())

    if strategy_class is None:
        supported = ", ".join(strategies.keys())
        raise ValueError(
            f"Unknown sampling strategy: '{strategy_name}'. "
            f"Supported strategies: {supported}"
        )

    return strategy_class(random_seed)
