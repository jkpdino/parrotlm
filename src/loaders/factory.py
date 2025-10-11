"""Factory for creating data loaders.

Implements the Factory Pattern and follows Open/Closed Principle:
- Open for extension: New loaders can be registered
- Closed for modification: Existing code doesn't need to change
"""

from pathlib import Path
from typing import Dict, Type

from ..config import SliceConfig
from ..exceptions import UnsupportedFormatError, ValidationError
from .base import DataLoader
from .huggingface_loader import HuggingFaceLoader
from .local_loaders import CSVLoader, JSONLLoader, JSONLoader, ParquetLoader


class LoaderFactory:
    """
    Factory for creating appropriate data loaders based on configuration.

    Implements Open/Closed Principle: Can register new loaders without
    modifying existing code.
    """

    # Registry mapping file extensions to loader classes
    _file_loaders: Dict[str, Type[DataLoader]] = {
        '.jsonl': JSONLLoader,
        '.json': JSONLoader,
        '.csv': CSVLoader,
        '.parquet': ParquetLoader,
        '.pq': ParquetLoader,
    }

    @classmethod
    def create_loader(cls, slice_config: SliceConfig) -> DataLoader:
        """
        Create appropriate loader based on slice configuration.

        Args:
            slice_config: Configuration for the data slice

        Returns:
            Appropriate DataLoader instance

        Raises:
            ValueError: If configuration is invalid or unsupported
        """
        if slice_config.huggingface:
            return cls._create_huggingface_loader(slice_config)
        elif slice_config.local:
            return cls._create_local_loader(slice_config)
        else:
            raise ValidationError(
                f"Slice '{slice_config.key}' must have either "
                "'huggingface' or 'local' property"
            )

    @classmethod
    def _create_huggingface_loader(cls, slice_config: SliceConfig) -> HuggingFaceLoader:
        """Create HuggingFace loader."""
        return HuggingFaceLoader(
            source=slice_config.huggingface,
            limit=slice_config.limit,
            offset=slice_config.offset,
            sampling=slice_config.sampling,
        )

    @classmethod
    def _create_local_loader(cls, slice_config: SliceConfig) -> DataLoader:
        """
        Create appropriate local file loader based on file extension.

        Args:
            slice_config: Slice configuration with local path

        Returns:
            Appropriate local file loader

        Raises:
            ValueError: If file format is unsupported
        """
        path = Path(slice_config.local)
        extension = path.suffix.lower()

        loader_class = cls._file_loaders.get(extension)

        if loader_class is None:
            supported_formats = ', '.join(cls._file_loaders.keys())
            raise UnsupportedFormatError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {supported_formats}"
            )

        return loader_class(
            source=slice_config.local,
            limit=slice_config.limit,
            offset=slice_config.offset,
            sampling=slice_config.sampling,
        )

    @classmethod
    def register_loader(cls, extension: str, loader_class: Type[DataLoader]) -> None:
        """
        Register a new loader for a file extension.

        This allows extending the factory with new loaders without modifying
        the existing code (Open/Closed Principle).

        Args:
            extension: File extension (e.g., '.xml', '.avro')
            loader_class: DataLoader subclass to handle this format

        Example:
            LoaderFactory.register_loader('.xml', XMLLoader)
        """
        if not extension.startswith('.'):
            extension = f'.{extension}'

        cls._file_loaders[extension.lower()] = loader_class

    @classmethod
    def get_supported_formats(cls) -> list[str]:
        """
        Get list of supported file formats.

        Returns:
            List of supported file extensions
        """
        return sorted(cls._file_loaders.keys())
