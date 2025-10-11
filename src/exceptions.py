"""Custom exceptions for the datasets utility."""


class DatasetError(Exception):
    """Base exception for all dataset-related errors."""
    pass


class ConfigError(DatasetError):
    """Raised when there's an error in configuration parsing or validation."""
    pass


class LoaderError(DatasetError):
    """Raised when a data loader encounters an error."""
    pass


class ValidationError(DatasetError):
    """Raised when data validation fails."""
    pass


class UnsupportedFormatError(LoaderError):
    """Raised when attempting to load an unsupported file format."""
    pass


class FileNotFoundError(LoaderError):
    """Raised when a data source file is not found."""
    pass


class SamplingError(DatasetError):
    """Raised when there's an error applying sampling strategy."""
    pass
