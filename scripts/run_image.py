import argparse
from pathlib import Path

import cv2

from src.visionx.detector import BACKENDS, VisionXDetector


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VisionX object detection."
    )
    parser.add_argument("image", help="Path to an input image")
    parser.add_argument(
        "--backend",
        choices=list(BACKENDS),
        default="openvino_gpu",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()

    detector = VisionXDetector(
        backend=args.backend,
        confidence=args.confidence,
        image_size=args.size,
    )

    result = detector.detect_image(args.image)

    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (
        f"{Path(args.image).stem}_{args.backend}_detected.jpg"
    )

    if not cv2.imwrite(str(output_path), result.plot()):
        raise RuntimeError(f"Could not save image: {output_path}")

    print(f"Backend: {detector.backend}")
    print(f"Device: {detector.device}")
    print(f"Image size: {args.size}")
    print(f"Detected objects: {len(result.boxes)}")

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        print(f"- {result.names[class_id]}: {confidence:.2%}")

    print(f"Saved result: {output_path.resolve()}")


if __name__ == "__main__":
    main()