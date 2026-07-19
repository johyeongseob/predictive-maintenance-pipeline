"""Registry for inference handler classes."""

from .handler import InferenceHandler


_handlers: dict[str, type[InferenceHandler]] = {}


def register_handler(name: str):
    """Register an inference handler class under a stable name."""
    def decorator(cls: type[InferenceHandler]):
        _handlers[name] = cls
        return cls

    return decorator


def get_handler_class(name: str) -> type[InferenceHandler]:
    """Return the handler class registered for name."""
    return _handlers[name]


def registered_handlers() -> dict[str, type[InferenceHandler]]:
    """Return a copy of the registered handler classes."""
    return dict(_handlers)
