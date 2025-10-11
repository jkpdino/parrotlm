"""HuggingFace datasets loader with streaming support."""

from typing import Dict, Iterator

from datasets import load_dataset

from ..exceptions import LoaderError
from .base import DataLoader


class HuggingFaceLoader(DataLoader):
    """
    Loader for HuggingFace datasets with streaming support.

    Follows Single Responsibility Principle: Only handles loading HuggingFace datasets.
    Sampling, offset, and limit are handled by the base class.
    """

    def _load_raw(self) -> Iterator[Dict]:
        """
        Load raw data from HuggingFace datasets with streaming.

        Returns:
            Iterator yielding dictionaries with the loaded data

        Raises:
            LoaderError: If dataset cannot be loaded
        """
        # Parse dataset path (format: "dataset_name" or "dataset_name:split")
        parts = self.source.split(':')
        dataset_name = parts[0]
        split = parts[1] if len(parts) > 1 else 'train'

        try:
            # Load dataset with streaming enabled for memory efficiency
            dataset = load_dataset(dataset_name, split=split, streaming=True)

            # Yield records as dictionaries
            for item in dataset:
                yield dict(item)

        except Exception as e:
            raise LoaderError(
                f"Failed to load HuggingFace dataset '{self.source}': {str(e)}"
            )

    def get_description(self) -> str:
        """Get human-readable description."""
        return f"HuggingFace - {super().get_description()}"
