"""Dispatcher for selecting inference handlers."""

from .handler import InferenceHandler
from .registry import get_handler_class, registered_handlers
from . import handlers as _handlers_module  # noqa: F401 - importing registers built-in handlers


def dispatch(config: dict) -> InferenceHandler:
    """Select an inference handler from config or can_handle() detection."""
    inference_cfg = config.get("inference", {})
    handler_name = inference_cfg.get("handler") or config.get("handler")

    if handler_name:
        handler_cls = get_handler_class(handler_name)
        return handler_cls()

    for handler_cls in registered_handlers().values():
        handler = handler_cls()
        if handler.can_handle(config):
            return handler

    raise ValueError("No handler found")
