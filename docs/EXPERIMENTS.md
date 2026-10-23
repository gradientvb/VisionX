# VisionX experiment methodology

This document explains how to reproduce VisionX's initial inference benchmark and the detection-versus-tracking comparison. The goal is transparent, repeatable engineering measurement—not unsupported claims about model accuracy.

## 1. Environment to record

Before comparing runs, record:

- Operating system and version
- Python version and installed package versions
- CPU model, RAM, GPU model, and graphics driver
- YOLO checkpoint/export version and selected inference backend
- Input image set or video source, resolution, and confidence threshold
- Warm-up configuration, repeat count, and any other running workloads

The repository's dependency file does not pin exact versions. For an environment snapshot, run:

```powershell
..venvScriptspython.exe --version
..venvScriptspython.exe -m pip freeze > experimentsesultsenvironment.txt
```

Run these commands from the repository root. The output file is generated locally and is not intended to be committed automatically.

## 2. Image inference benchmark

### What the script does

Run:

```powershell
..venvScriptspython.exe .scriptsenchmark.py
```

The script searches **assets/sample_images/** recursively for supported image formats. It tests three input sizes (320, 480, 640) and these configurations:

- PyTorch CPU using `yolo11n.pt`
- OpenVINO CPU using `yolo11n_openvino_model`
- OpenVINO Intel GPU using `yolo11n_openvino_model`

For each backend and input size, it performs three warm-up predictions and then ten predictions per image at confidence threshold 0.25. It records per-image average and median latency, latency standard deviation, estimated FPS, and average detection count. Aggregate and per-image CSV files are written to **experiments/results/**.

### Output files

- `benchmark_summary.csv`: one aggregate row per backend and input size
- `benchmark_per_image.csv`: per-image measurement summaries

If a backend fails, the script reports the error and continues with the next configuration. Confirm that each intended backend completed before comparing results.

### Interpreting results

The reported estimated FPS is calculated as `1000 / average_latency_ms`. This is the reciprocal of measured prediction latency; it is **not a measurement of sustained video throughput**. The timing surrounds the Ultralytics prediction call and does not represent the full Streamlit app, camera capture, video decoding, or display pipeline.

Detection counts are descriptive outputs, not accuracy metrics. Without labeled ground truth, these measurements do not establish precision, recall, mAP, or whether detections are correct.

The README's initial comparison used 21 images. The image set was small and largely slide-like, so its results should be treated as preliminary and not generalized to arbitrary scenes or devices.

## 3. Tracking OFF versus ON

### Command

Use a local video that you have permission to process:

```powershell
..venvScriptspython.exe .scriptscompare_tracking.py --video .path	oyour_video.mp4 --backend pytorch_cpu --size 640 --confidence 0.25 --runs 5 --warmup-frames 20
```

The script accepts backends `pytorch_cpu`, `openvino_cpu`, and `openvino_gpu`; image sizes 320, 480, and 640; a confidence threshold from 0.05 to 0.95; a run count of at least one; and a non-negative warm-up frame count.

### Procedure

- Uses the same source video and selected backend/settings for both conditions.
- Compares tracking disabled with ByteTrack enabled.
- Excludes the configured initial warm-up frames from reported timing metrics.
- Alternates the order of tracking OFF and ON between runs to reduce order-related bias.
- Collects mean, p50, and p95 processing latency, estimated processing FPS, detection counts, unique track IDs, and a track-label majority-fraction proxy.
- Writes results after all requested runs complete.

### Output files

- `tracking_comparison_summary.csv`: one row for each condition in each run
- `tracking_comparison_frames.csv`: per-frame measurements from the runs

These files are generated under **experiments/results/** and are excluded from normal source control by the project's ignore rules. Preserve them with your local experiment notes when you need to review a result.

### Timing and metric caveats

The timing covers model inference and result plotting in the comparison script, but excludes video decoding. Its processing FPS is therefore not end-to-end application FPS and should not be presented as a webcam or full application frame rate.

The track-label majority fraction summarizes whether a track was assigned the same predicted class on most of its observations. It is a **consistency proxy**, not proof that the track corresponds to the same real-world object or that the class is correct. Unique track IDs and detection counts also do not measure tracking accuracy.

A rigorous accuracy study needs ground-truth labels and suitable metrics. Depending on the task, those may include precision, recall, and mAP for detection, and established identity/association metrics for multi-object tracking.

## 4. Reporting checklist

When publishing results, include:

- Hardware, OS, Python and package versions
- Model artifact and backend/device
- Input source description, number of images or video properties
- Image size, confidence threshold, warm-up count, and number of repeats
- Whether the reported timing includes decoding, rendering, and UI work
- Aggregate statistics and, where useful, variability across images/runs
- Limitations, including lack of ground truth where applicable

Do not report a single best run as a stable performance guarantee. Avoid comparing results produced with different sources, settings, hardware, or measurement boundaries.

## 5. Current limitations and next steps

The current experiments are local performance probes rather than a formal model evaluation. Improvements that would make future results stronger include:

1. Use a larger, varied, documented set of images and videos.
2. Pin dependencies or publish a tested environment lock file.
3. Retain raw run outputs and summarize variation across repeated runs.
4. Evaluate detection and tracking with ground truth before making accuracy claims.
5. Measure end-to-end latency separately if the goal is real-time user experience.
