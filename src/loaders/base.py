"""Abstract base class for data loaders following SOLID principles."""

from abc import ABC, abstractmethod
from typing import Dict, Iterator, Optional

from .sampling import create_sampling_strategy


class DataLoader(ABC):
    """
    Abstract base class for all data loaders.

    Uses Template Method pattern: subclasses implement _load_raw(),
    base class handles sampling, offset, and limit.

    Implements SOLID principles:
    - Single Responsibility: Loading data
    - Open/Closed: Extendable without modification
    - Liskov Substitution: All loaders are interchangeable
    - Interface Segregation: Minimal interface
    - Dependency Inversion: Depends on abstractions
    """

    def __init__(
        self,
        source: str,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        sampling: str = "sequential",
        random_seed: int = 42,
    ):
        """
        Initialize the data loader.

        Args:
            source: Data source identifier (path, dataset name, etc.)
            limit: Maximum number of records to load
            offset: Number of records to skip from the beginning
            sampling: Sampling strategy ("random" or "sequential")
            random_seed: Random seed for reproducibility
        """
        self.source = source
        self.limit = limit
        self.offset = offset or 0
        self.sampling = sampling
        self.random_seed = random_seed

        # Create sampling strategy (Strategy Pattern)
        self._sampling_strategy = create_sampling_strategy(sampling, random_seed)

    def load(self) -> Iterator[Dict]:
        """
        Load data from the source with sampling applied.

        Template Method: Orchestrates the loading process.
        Subclasses only need to implement _load_raw().

        Returns:
            Iterator yielding dictionaries representing data records

        Raises:
            LoaderError: If data source is invalid or cannot be loaded
        """
        # Step 1: Load raw data (implemented by subclasses)
        raw_iterator = self._load_raw()

        # Step 2: Apply sampling strategy (handled by base class)
        sampled_iterator = self._sampling_strategy.apply(
            raw_iterator,
            offset=self.offset,
            limit=self.limit
        )

        # Step 3: Yield processed records
        yield from sampled_iterator

    @abstractmethod
    def _load_raw(self) -> Iterator[Dict]:
        """
        Load raw data from the source without sampling.

        Subclasses must implement this method to load data from
        their specific source (HuggingFace, local files, etc.).

        Returns:
            Iterator yielding raw data records

        Raises:
            LoaderError: If data cannot be loaded
        """
        pass

    def get_description(self) -> str:
        """
        Get a human-readable description of the loader configuration.

        Returns:
            Description string
        """
        desc = f"Source: {self.source}"
        if self.offset:
            desc += f", Offset: {self.offset}"
        if self.limit:
            desc += f", Limit: {self.limit}"
        desc += f", Sampling: {self.sampling}"
        return desc
