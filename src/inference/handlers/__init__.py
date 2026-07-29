"""Built-in inference handlers."""

from .dlstreamer_detect import DLStreamerDetectHandler
from .openvino_classify import OpenVINOClassifyHandler
from .sensor_flat import SensorFlatHandler

__all__ = ["DLStreamerDetectHandler", "OpenVINOClassifyHandler", "SensorFlatHandler"]
