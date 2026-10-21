import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visionx.detector import BACKENDS, VisionXDetector


class TestVisionXDetector(unittest.TestCase):

    def test_three_backends_are_configured(self):
        self.assertEqual(
            set(BACKENDS),
            {"pytorch_cpu", "openvino_cpu", "openvino_gpu"},
        )

    def test_invalid_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            VisionXDetector(backend="invalid_backend")

    def test_missing_image_raises_file_not_found(self):
        detector = VisionXDetector.__new__(VisionXDetector)

        with self.assertRaises(FileNotFoundError):
            detector.detect_image("this_image_does_not_exist.jpg")

    def test_inference_uses_configured_parameters(self):
        detector = VisionXDetector.__new__(VisionXDetector)
        detector.confidence = 0.4
        detector.image_size = 320
        detector.device = "cpu"
        detector.model = Mock()

        expected_result = object()
        detector.model.predict.return_value = [expected_result]

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.jpg"
            image_path.write_bytes(b"test image placeholder")

            result = detector.detect_image(str(image_path))

        self.assertIs(result, expected_result)
        detector.model.predict.assert_called_once_with(
            source=str(image_path),
            conf=0.4,
            imgsz=320,
            device="cpu",
            verbose=False,
        )


if __name__ == "__main__":
    unittest.main()
