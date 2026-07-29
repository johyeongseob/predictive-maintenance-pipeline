"""Dispatcher for selecting inference handlers."""

from .handler import InferenceHandler
from .registry import get_handler_class, registered_handlers
from . import handlers as _handlers_module  # noqa: F401 - importing registers built-in handlers


def dispatch(config: dict) -> InferenceHandler:
    """Select an inference handler from config."""
    inference_cfg = config.get("inference", {})
    handler_name = inference_cfg.get("handler") or config.get("handler")

    if not handler_name:
        raise ValueError("Missing inference handler. Set inference.handler in config YAML.")

    handler_cls = get_handler_class(handler_name)
    return handler_cls()