"""Base interface for inference handlers."""

from abc import ABC, abstractmethod


class InferenceHandler(ABC):
    """Common interface implemented by all input-type inference handlers."""

    @abstractmethod
    def can_handle(self, config: dict) -> bool:
        """Return True when this handler supports the provided config."""
        ...

    @abstractmethod
    def load(self, config: dict) -> None:
        """Load models, preprocessors, and other runtime resources."""
        ...

    @abstractmethod
    def infer(self, inputs: list, config: dict) -> list[dict]:
        """Run inference and return result dictionaries."""
        ...

    @abstractmethod
    def get_output_schema(self) -> dict:
        """Return the SQLite-compatible output schema for this handler."""
        ...
