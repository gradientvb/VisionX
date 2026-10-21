
import csv
import statistics
import time
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO


IMAGE_DIR = Path("assets/sample_images")
PYTORCH_MODEL = "yolo11n.pt"
OPENVINO_MODEL = "yolo11n_openvino_model"

IMAGE_SIZES = [320, 480, 640]
CONFIDENCE = 0.25
WARMUP_RUNS = 3
REPEATS_PER_IMAGE = 10

RESULTS_DIR = Path("experiments/results")
SUMMARY_CSV = RESULTS_DIR / "benchmark_summary.csv"
DETAIL_CSV = RESULTS_DIR / "benchmark_per_image.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CONFIGURATIONS = [
    ("PyTorch CPU", PYTORCH_MODEL, "cpu"),
    ("OpenVINO CPU", OPENVINO_MODEL, "CPU"),
    ("OpenVINO Intel GPU", OPENVINO_MODEL, "intel:gpu"),
]


def get_images():
    images = sorted(
        path
        for path in IMAGE_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise FileNotFoundError(
            f"No supported images found in {IMAGE_DIR.resolve()}"
        )

    return images


def run_benchmark(name, model_path, device, images):
    print(f"\n{'=' * 65}")
    print(f"Configuration: {name} | Device: {device}")
    print("=" * 65)

    model = YOLO(model_path)
    detail_rows = []

    for image_size in IMAGE_SIZES:
        print(f"\nImage size: {image_size}")

        # Warm up before recording measurements.
        for _ in range(WARMUP_RUNS):
            model.predict(
                source=str(images[0]),
                imgsz=image_size,
                conf=CONFIDENCE,
                device=device,
                verbose=False,
            )

        for image_index, image_path in enumerate(images, start=1):
            latencies = []
            detection_counts = []

            for _ in range(REPEATS_PER_IMAGE):
                start = time.perf_counter()

                result = model.predict(
                    source=str(image_path),
                    imgsz=image_size,
                    conf=CONFIDENCE,
                    device=device,
                    verbose=False,
                )[0]

                latencies.append(
                    (time.perf_counter() - start) * 1000
                )
                detection_counts.append(
                    len(result.boxes) if result.boxes is not None else 0
                )

            avg_ms = statistics.mean(latencies)

            detail_rows.append({
                "backend": name,
                "device": device,
                "image": str(image_path),
                "image_size": image_size,
                "repeats": REPEATS_PER_IMAGE,
                "average_latency_ms": round(avg_ms, 2),
                "median_latency_ms": round(
                    statistics.median(latencies), 2
                ),
                "latency_std_ms": round(
                    statistics.stdev(latencies), 2
                ) if len(latencies) > 1 else 0,
                "estimated_fps": round(1000 / avg_ms, 2),
                "average_detection_count": round(
                    statistics.mean(detection_counts), 2
                ),
            })

            print(
                f"[{image_index}/{len(images)}] {image_path.name}: "
                f"{avg_ms:.2f} ms, "
                f"{1000 / avg_ms:.2f} estimated FPS"
            )

    return detail_rows


def write_csv(path, rows):
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    images = get_images()
    print(f"Images found: {len(images)}")

    for image in images:
        print(f"  - {image.name}")

    all_details = []

    for name, model_path, device in CONFIGURATIONS:
        try:
            all_details.extend(
                run_benchmark(name, model_path, device, images)
            )
        except Exception as error:
            print(f"\nFAILED: {name}")
            print(f"{type(error).__name__}: {error}")
            print("Continuing with the next configuration.")

    if not all_details:
        raise RuntimeError("No benchmark configurations completed.")

    # Aggregate results across images and repeated predictions.
    groups = defaultdict(list)
    for row in all_details:
        groups[(row["backend"], row["device"], row["image_size"])].append(row)

    summary_rows = []

    for (backend, device, image_size), rows in groups.items():
        image_latencies = [r["average_latency_ms"] for r in rows]
        average_ms = statistics.mean(image_latencies)

        summary_rows.append({
            "backend": backend,
            "device": device,
            "image_size": image_size,
            "images_tested": len(rows),
            "repeats_per_image": REPEATS_PER_IMAGE,
            "average_latency_ms": round(average_ms, 2),
            "median_image_latency_ms": round(
                statistics.median(image_latencies), 2
            ),
            "latency_std_across_images_ms": round(
                statistics.stdev(image_latencies), 2
            ) if len(image_latencies) > 1 else 0,
            "estimated_fps": round(1000 / average_ms, 2),
            "mean_detection_count": round(
                statistics.mean(
                    r["average_detection_count"] for r in rows
                ), 2
            ),
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(DETAIL_CSV, all_details)
    write_csv(SUMMARY_CSV, summary_rows)

    print("\n\nAGGREGATE SUMMARY")
    print("-" * 90)
    print(
        f"{'Backend':<22} {'Size':>5} {'Images':>7} "
        f"{'Avg ms':>10} {'FPS':>9} {'Mean detections':>17}"
    )
    print("-" * 90)

    for row in summary_rows:
        print(
            f"{row['backend']:<22} "
            f"{row['image_size']:>5} "
            f"{row['images_tested']:>7} "
            f"{row['average_latency_ms']:>10.2f} "
            f"{row['estimated_fps']:>9.2f} "
            f"{row['mean_detection_count']:>17.2f}"
        )

    print(f"\nSummary CSV: {SUMMARY_CSV.resolve()}")
    print(f"Per-image CSV: {DETAIL_CSV.resolve()}")


if __name__ == "__main__":
    main()
