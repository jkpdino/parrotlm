"""Local file loaders with streaming support.

Each loader follows Single Responsibility Principle by handling one file format.
Sampling, offset, and limit are handled by the base class.
"""

import json
from pathlib import Path
from typing import Dict, Iterator

import pandas as pd

from ..exceptions import LoaderError
from .base import DataLoader


class JSONLLoader(DataLoader):
    """
    Loader for JSONL (JSON Lines) files with streaming support.

    Most memory-efficient local file format for streaming.
    """

    def _load_raw(self) -> Iterator[Dict]:
        """
        Load raw data from JSONL file with streaming.

        Yields:
            Dictionary for each line in the JSONL file

        Raises:
            LoaderError: If file cannot be loaded
        """
        path = Path(self.source)

        if not path.exists():
            raise LoaderError(f"File not found: {path}")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        except json.JSONDecodeError as e:
            raise LoaderError(f"Invalid JSON in file '{path}': {str(e)}")
        except Exception as e:
            raise LoaderError(f"Failed to load JSONL file '{path}': {str(e)}")

    def get_description(self) -> str:
        """Get human-readable description."""
        return f"JSONL - {super().get_description()}"


class JSONLoader(DataLoader):
    """
    Loader for JSON files.

    Note: Requires loading entire file into memory.
    """

    def _load_raw(self) -> Iterator[Dict]:
        """
        Load raw data from JSON file.

        Yields:
            Dictionary records from the JSON file

        Raises:
            LoaderError: If file cannot be loaded
        """
        path = Path(self.source)

        if not path.exists():
            raise LoaderError(f"File not found: {path}")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle both list and single object
            if isinstance(data, list):
                yield from data
            else:
                yield data

        except json.JSONDecodeError as e:
            raise LoaderError(f"Invalid JSON in file '{path}': {str(e)}")
        except Exception as e:
            raise LoaderError(f"Failed to load JSON file '{path}': {str(e)}")

    def get_description(self) -> str:
        """Get human-readable description."""
        return f"JSON - {super().get_description()}"


class CSVLoader(DataLoader):
    """
    Loader for CSV files with streaming support using chunked reading.
    """

    def _load_raw(self) -> Iterator[Dict]:
        """
        Load raw data from CSV file with chunked streaming.

        Yields:
            Dictionary for each row in the CSV file

        Raises:
            LoaderError: If file cannot be loaded
        """
        path = Path(self.source)

        if not path.exists():
            raise LoaderError(f"File not found: {path}")

        try:
            # Stream CSV with chunks for memory efficiency
            chunk_size = 1000
            for chunk in pd.read_csv(path, chunksize=chunk_size):
                for record in chunk.to_dict('records'):
                    yield record

        except Exception as e:
            raise LoaderError(f"Failed to load CSV file '{path}': {str(e)}")

    def get_description(self) -> str:
        """Get human-readable description."""
        return f"CSV - {super().get_description()}"


class ParquetLoader(DataLoader):
    """
    Loader for Parquet files.

    Note: Loads entire file into memory as pandas doesn't support streaming parquet.
    For true streaming of large Parquet files, consider implementing with pyarrow directly.
    """

    def _load_raw(self) -> Iterator[Dict]:
        """
        Load raw data from Parquet file.

        Yields:
            Dictionary for each row in the Parquet file

        Raises:
            LoaderError: If file cannot be loaded
        """
        path = Path(self.source)

        if not path.exists():
            raise LoaderError(f"File not found: {path}")

        try:
            # pandas read_parquet doesn't support chunking
            df = pd.read_parquet(path)
            yield from df.to_dict('records')

        except Exception as e:
            raise LoaderError(f"Failed to load Parquet file '{path}': {str(e)}")

    def get_description(self) -> str:
        """Get human-readable description."""
        return f"Parquet - {super().get_description()}"
