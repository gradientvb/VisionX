r"""
Compare VisionX detection with tracking OFF vs ON on the SAME video.

Run from the VisionX project root, for example:
  .\.venv\Scripts\python.exe .\scripts\compare_tracking.py `
      --video .\assets\sample_videos\Traffic_mp4.mp4 `
      --backend pytorch_cpu --size 640 --confidence 0.25 `
      --runs 5 --warmup-frames 20

This is an engineering performance comparison, not an accuracy benchmark.
Without ground-truth labels, detection counts and track-label consistency
must not be interpreted as precision, recall, mAP, or tracking accuracy.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from src.visionx.detector import VisionXDetector


DEVICE_BY_BACKEND = {
    "pytorch_cpu": "cpu",
    "openvino_cpu": "CPU",
    "openvino_gpu": "intel:gpu",
}


def percentile(values: list[float], percent: float) -> float:
    """Return a nearest-rank-style percentile for a non-empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((percent / 100) * (len(ordered) - 1))
    return ordered[index]


def run_condition(
    video_path: Path,
    backend: str,
    confidence: float,
    image_size: int,
    tracking: bool,
    run_number: int,
    warmup_frames: int,
) -> tuple[dict, list[dict]]:
    """Run one OFF or ON condition and return its summary and frame rows."""
    detector = VisionXDetector(
        backend=backend,
        confidence=confidence,
        image_size=image_size,
    )
    device = DEVICE_BY_BACKEND[backend]
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video: {video_path}")

    reported_fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_times_ms: list[float] = []
    per_frame_rows: list[dict] = []
    track_class_counts: dict[int, Counter] = defaultdict(Counter)
    total_detections = 0
    measured_frames = 0
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            start = time.perf_counter()

            if tracking:
                results = detector.model.track(
                    source=frame,
                    conf=confidence,
                    imgsz=image_size,
                    device=device,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )
            else:
                results = detector.model.predict(
                    source=frame,
                    conf=confidence,
                    imgsz=image_size,
                    device=device,
                    verbose=False,
                )

            if not results:
                elapsed_ms = (time.perf_counter() - start) * 1000
                result = None
            else:
                result = results[0]
                # Include drawing cost in the measured processing loop.
                _ = result.plot()
                elapsed_ms = (time.perf_counter() - start) * 1000

            # Warm-up frames are processed but excluded from reported metrics.
            if frame_index >= warmup_frames:
                frame_detection_count = (
                    len(result.boxes)
                    if result is not None and result.boxes is not None
                    else 0
                )
                track_ids: list[int] = []

                if (
                    tracking
                    and result is not None
                    and result.boxes is not None
                    and result.boxes.id is not None
                ):
                    class_ids = result.boxes.cls.int().cpu().tolist()
                    track_ids = result.boxes.id.int().cpu().tolist()
                    for track_id, class_id in zip(track_ids, class_ids):
                        class_name = result.names[int(class_id)]
                        track_class_counts[track_id][class_name] += 1

                frame_times_ms.append(elapsed_ms)
                measured_frames += 1
                total_detections += frame_detection_count

                per_frame_rows.append(
                    {
                        "video": str(video_path),
                        "backend": backend,
                        "tracking": "on" if tracking else "off",
                        "run": run_number,
                        "frame_index": frame_index,
                        "inference_and_plot_ms": round(elapsed_ms, 3),
                        "detections_in_frame": frame_detection_count,
                        "track_ids": ";".join(map(str, track_ids)),
                    }
                )

            frame_index += 1
    finally:
        capture.release()

    mean_ms = statistics.mean(frame_times_ms) if frame_times_ms else 0.0
    p50_ms = percentile(frame_times_ms, 50)
    p95_ms = percentile(frame_times_ms, 95)

    track_consistency_values = []
    for counts in track_class_counts.values():
        count_total = sum(counts.values())
        if count_total:
            track_consistency_values.append(max(counts.values()) / count_total)

    summary = {
        "video": str(video_path),
        "backend": backend,
        "tracking": "on" if tracking else "off",
        "run": run_number,
        "source_width": width,
        "source_height": height,
        "source_fps": round(reported_fps, 3) if reported_fps else 0,
        "source_frames": frame_total,
        "warmup_frames_excluded": min(warmup_frames, frame_index),
        "measured_frames": measured_frames,
        "mean_processing_ms": round(mean_ms, 3),
        "p50_processing_ms": round(p50_ms, 3),
        "p95_processing_ms": round(p95_ms, 3),
        "processing_fps": round(1000 / mean_ms, 3) if mean_ms else 0,
        "total_frame_detections": total_detections,
        "unique_track_ids": len(track_class_counts) if tracking else "",
        "mean_track_label_majority_fraction": (
            round(statistics.mean(track_consistency_values), 4)
            if track_consistency_values
            else ""
        ),
    }
    return summary, per_frame_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write all accumulated rows to CSV, including a header."""
    if not rows:
        print(f"Warning: no rows to write for {path.name}")
        return

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare VisionX tracking OFF vs ON on the same video."
    )
    parser.add_argument("--video", required=True, help="Path to a local video file.")
    parser.add_argument(
        "--backend",
        choices=list(DEVICE_BY_BACKEND),
        default="pytorch_cpu",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--size", type=int, choices=[320, 480, 640], default=640)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-frames", type=int, default=20)
    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        raise SystemExit(f"Video file not found: {video_path}")
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1.")
    if args.warmup_frames < 0:
        raise SystemExit("--warmup-frames cannot be negative.")
    if not 0.05 <= args.confidence <= 0.95:
        raise SystemExit("--confidence must be between 0.05 and 0.95.")

    project_root = Path.cwd()
    results_dir = project_root / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "tracking_comparison_summary.csv"
    frames_path = results_dir / "tracking_comparison_frames.csv"

    summaries: list[dict] = []
    frame_rows: list[dict] = []

    for run_number in range(1, args.runs + 1):
        # Alternate order each run to reduce order-related bias.
        if run_number % 2 == 1:
            conditions = [(False, "OFF"), (True, "ON")]
        else:
            conditions = [(True, "ON"), (False, "OFF")]

        for tracking, label in conditions:
            print(
                f"\nRun {run_number}/{args.runs} | Tracking {label} | "
                f"{args.backend} | size={args.size} | conf={args.confidence:.2f}"
            )
            summary, rows = run_condition(
                video_path=video_path,
                backend=args.backend,
                confidence=args.confidence,
                image_size=args.size,
                tracking=tracking,
                run_number=run_number,
                warmup_frames=args.warmup_frames,
            )
            summaries.append(summary)
            frame_rows.extend(rows)
            print(
                f"  frames={summary['measured_frames']} | "
                f"mean={summary['mean_processing_ms']} ms | "
                f"FPS={summary['processing_fps']} | "
                f"detections={summary['total_frame_detections']} | "
                f"unique IDs={summary['unique_track_ids']}"
            )

    # Export only after all requested runs have completed.
    # This preserves every run in the summary and per-frame CSVs.
    write_csv(summary_path, summaries)
    write_csv(frames_path, frame_rows)

    print("\nExperiment complete.")
    print(f"Summary rows written: {len(summaries)}")
    print(f"Per-frame rows written: {len(frame_rows)}")
    print(f"Summary CSV: {summary_path}")
    print(f"Per-frame CSV: {frames_path}")
    print(
        "\nInterpretation note: latency and FPS are performance measures. "
        "Detection counts are not accuracy. The track-label majority fraction "
        "is a consistency proxy, not ground-truth correctness."
    )


if __name__ == "__main__":
    main()
