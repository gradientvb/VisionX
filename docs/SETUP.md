# VisionX setup guide

This guide covers a fresh Windows 10/11 setup using PowerShell. Run commands from the repository root. VisionX currently targets Python 3.13.

## 1. Prerequisites

- Python 3.13 (available through the Windows Python launcher)
- Git
- Internet access for package and model downloads

```powershell
py -3.13 --version
git --version
```

## 2. Clone and install dependencies

```powershell
git clone https://github.com/gradientvb/VisionX.git
cd VisionX
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\requirement.txt
```

The dependency file is named **requirement.txt** (singular). Calling the environment's Python directly avoids the need to activate the virtual environment.

## 3. Prepare model files

Model weights and exported model artifacts are excluded from Git. Download the YOLO11n checkpoint:

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

For OpenVINO backends, export the checkpoint:

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='openvino')"
```

The export should create **yolo11n_openvino_model/** in the project root.

## 4. Launch the app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit. Select a backend in the sidebar. The OpenVINO Intel GPU backend requires compatible Intel graphics hardware, drivers, and an available OpenVINO GPU device. If it is unavailable, try **PyTorch CPU** or **OpenVINO CPU**.

## 5. Download sample images

Sample images are not committed to the repository. From the project root, run:

```powershell
.\download_visionx_test_data.ps1
```

The script downloads a small set of COCO 2017 validation images into **assets/sample_images/**. These are useful for smoke tests and latency experiments, not standalone accuracy evaluation.

Run detection on one image:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_image.py .\assets\sample_images\000000179765.jpg --backend pytorch_cpu --size 640 --confidence 0.25
```

Annotated output is saved under **experiments/results/**.

## 6. Run tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

## 7. Compare tracking OFF versus ON

VisionX does not distribute sample videos. Supply a local video file you have permission to use:

```powershell
.\.venv\Scripts\python.exe .\scripts\compare_tracking.py --video .\path\to\your_video.mp4 --backend pytorch_cpu --size 640 --confidence 0.25 --runs 5 --warmup-frames 20
```

The script records summary and per-frame results under **experiments/results/**. It measures model inference plus result plotting, but excludes video decoding; its throughput is therefore not end-to-end application FPS.

Interpret the metrics carefully:
- Latency and throughput are performance measures, not accuracy measures.
- Detection counts are not precision, recall, or mAP.
- The track-label majority fraction is a consistency proxy, not proof of correct object identity.
- Ground-truth annotations and suitable metrics are needed for accuracy or tracking-quality claims.

## Troubleshooting

**Model not found:** Run the checkpoint download and OpenVINO export commands above from the repository root.

**Intel GPU unavailable:** Verify device and driver support. Start with **openvino_cpu** to confirm that the OpenVINO model loads.

**Sample images missing:** Run the PowerShell downloader. Video files are not included; provide your own valid video path.

## Reproducibility checklist

When reporting an experiment, record the operating system, Python and package versions, processor and graphics device/driver, model and backend, input source, input size, confidence threshold, warm-up settings, and number of runs. The dependency file does not pin exact package versions, so record installed versions when exact reproducibility matters.
