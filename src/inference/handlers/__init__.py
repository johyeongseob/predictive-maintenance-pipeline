"""Built-in inference handlers."""

from .dlstreamer_detect import DLStreamerDetectHandler
from .openvino_audio_text_classify import OpenVINOAudioTextClassifyHandler
from .openvino_classify import OpenVINOClassifyHandler
from .openvino_detect import OpenVINODetectHandler
from .sensor_flat import SensorFlatHandler

__all__ = [
    "DLStreamerDetectHandler",
    "OpenVINOAudioTextClassifyHandler",
    "OpenVINOClassifyHandler",
    "OpenVINODetectHandler",
    "SensorFlatHandler"
]
