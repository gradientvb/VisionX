from pathlib import Path

from ultralytics import YOLO
from ultralytics.engine.results import Results


BACKENDS = {
    "pytorch_cpu": {
        "model": "yolo11n.pt",
        "device": "cpu",
    },
    "openvino_cpu": {
        "model": "yolo11n_openvino_model",
        "device": "CPU",
    },
    "openvino_gpu": {
        "model": "yolo11n_openvino_model",
        "device": "intel:gpu",
    },
}


class VisionXDetector:
    """Object detection engine with selectable inference backends."""

    def __init__(
        self,
        backend: str = "openvino_gpu",
        confidence: float = 0.25,
        image_size: int = 640,
    ) -> None:
        if backend not in BACKENDS:
            choices = ", ".join(BACKENDS)
            raise ValueError(
                f"Unknown backend '{backend}'. Choose from: {choices}"
            )

        config = BACKENDS[backend]
        self.backend = backend
        self.device = config["device"]
        self.image_size = image_size
        self.confidence = confidence

        model_path = Path(config["model"])
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path.resolve()}"
            )

        self.model = YOLO(str(model_path))

    def detect_image(self, image_path: str) -> Results:
        """Run object detection on an image."""
        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Image not found: {path.resolve()}"
            )

        results = self.model.predict(
            source=str(path),
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

        return results[0]