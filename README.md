\# VisionX — Real-Time Visual Intelligence

<p align="center">
  <img
    src="assets/visionx_demo.gif"
    alt="VisionX real-time object detection demo"
    width="800"
  />
</p>

<p align="center">

&#x20; <strong>Real-time object detection, multi-object tracking, and hardware-aware computer vision.</strong>

&#x20; <br />

&#x20; An experimental computer vision system exploring how modern AI vision models perform on everyday consumer hardware.

</p>



<p align="center">

&#x20; <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge\&logo=python\&logoColor=white" alt="Python" />

&#x20; <img src="https://img.shields.io/badge/YOLO11-Object%20Detection-111827?style=for-the-badge" alt="YOLO11" />

&#x20; <img src="https://img.shields.io/badge/OpenVINO-Optimized%20Inference-0071C5?style=for-the-badge\&logo=intel\&logoColor=white" alt="OpenVINO" />

&#x20; <img src="https://img.shields.io/badge/Streamlit-Interactive%20UI-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white" alt="Streamlit" />

</p>



<p align="center">

&#x20; <a href="#-overview">Overview</a> •

&#x20; <a href="#-features">Features</a> •

&#x20; <a href="#-architecture">Architecture</a> •

&#x20; <a href="#-installation">Installation</a> •

&#x20; <a href="#-experiments">Experiments</a> •

&#x20; <a href="#-roadmap">Roadmap</a>

</p>



\---



\## 🔭 Overview



\*\*VisionX\*\* is a real-time visual intelligence project built to explore object detection and tracking beyond simply running a pretrained model.



The system combines \*\*YOLO11n\*\*, \*\*OpenVINO\*\*, \*\*ByteTrack\*\*, and an interactive \*\*Streamlit\*\* interface to investigate how detection pipelines behave across different inference backends and real-world video scenarios.



A key engineering question behind the project is:



> \*\*What changes when object tracking is added to a real-time object detection pipeline running on consumer hardware?\*\*



VisionX explores inference latency, processing throughput, detection behavior, and track-level consistency while recognizing that speed measurements alone cannot establish detection accuracy.



\### 🎯 Project objectives



\- Build a working computer vision application with image, video, and webcam workflows.

\- Compare PyTorch CPU inference with Intel OpenVINO CPU and GPU execution.

\- Explore multi-object tracking using ByteTrack.

\- Measure latency and processing throughput using repeatable experiments.

\- Develop an evidence-driven understanding of computer vision performance on hardware without a discrete NVIDIA GPU.



\## ✨ Features



| Feature | Description |

|---|---|

| 🖼️ Image detection | Upload images and inspect annotated object detections. |

| 🎥 Video analysis | Process video frames and visualize detected objects. |

| 📷 Webcam inference | Run live object detection using a connected camera. |

| 🧭 Multi-object tracking | Associate detections across frames and display tracking IDs. |

| ⚡ Multiple inference backends | Compare PyTorch CPU, OpenVINO CPU, and OpenVINO Intel GPU. |

| 🎚️ Configurable inference | Adjust confidence threshold and image resolution. |

| 📊 Performance metrics | Inspect latency and estimated processing FPS. |

| 📥 Export results | Download annotated images and save experiment outputs. |

| 🧪 Experiment tooling | Compare tracking-enabled and tracking-disabled pipelines. |



\## 🧠 Technology Stack



<p align="center">

&#x20; <img src="https://skillicons.dev/icons?i=python,opencv,pytorch" alt="Python, OpenCV and PyTorch" />

</p>



\- \*\*Language:\*\* Python

\- \*\*Detection:\*\* Ultralytics YOLO11n

\- \*\*Inference optimization:\*\* OpenVINO

\- \*\*Deep learning runtime:\*\* PyTorch

\- \*\*Computer vision:\*\* OpenCV

\- \*\*Tracking:\*\* ByteTrack through the Ultralytics tracking interface

\- \*\*Dashboard:\*\* Streamlit

\- \*\*Data analysis:\*\* Pandas

\- \*\*Testing:\*\* Python unittest and project-specific tests



\## 🏗️ Architecture



```text

&#x20;                        ┌──────────────────────┐

&#x20;                        │      Streamlit UI    │

&#x20;                        │ Image / Video / Live │

&#x20;                        └──────────┬───────────┘

&#x20;                                   │

&#x20;                        ┌──────────▼───────────┐

&#x20;                        │   VisionX Detector    │

&#x20;                        │  YOLO11n Inference    │

&#x20;                        └──────────┬───────────┘

&#x20;                                   │

&#x20;               ┌───────────────────┼──────────────────┐

&#x20;               │                   │                  │

&#x20;       ┌───────▼────────┐  ┌───────▼────────┐ ┌──────▼─────────┐

&#x20;       │  PyTorch CPU   │  │ OpenVINO CPU   │ │ OpenVINO Intel │

&#x20;       │                │  │                │ │ GPU            │

&#x20;       └────────────────┘  └────────────────┘ └────────────────┘

&#x20;                                   │

&#x20;                        ┌──────────▼───────────┐

&#x20;                        │ Optional ByteTrack   │

&#x20;                        │ Track IDs over time  │

&#x20;                        └──────────┬───────────┘

&#x20;                                   │

&#x20;                        ┌──────────▼───────────┐

&#x20;                        │ Metrics and Results  │

&#x20;                        │ Latency / FPS / Logs │

&#x20;                        └──────────────────────┘

```



\### Repository structure



```text

VisionX/

├── app.py

├── requirement.txt

├── .gitignore

├── assets/

│   ├── sample\_images/

│   └── sample\_videos/

├── configs/

├── docs/

├── experiments/

│   ├── plots/

│   ├── reports/

│   └── results/

├── models/

├── scripts/

│   ├── benchmark.py

│   ├── compare\_tracking.py

│   ├── run\_image.py

│   └── test\_gpu.py

├── src/

│   └── visionx/

│       ├── \_\_init\_\_.py

│       └── detector.py

├── tests/

│   └── test\_detector.py

└── yolov11n.pt

```



\*Note: This tree describes the intended project layout. Keep only paths and filenames that actually exist in the published repository.\*



\## ⚙️ Installation



\### 1. Clone the repository



```powershell

git clone https://github.com/YOUR\_USERNAME/VisionX.git

cd VisionX

```



Replace `YOUR\_USERNAME` with your GitHub username.



\### 2. Create a virtual environment



```powershell

py -3.13 -m venv .venv

```



\### 3. Install dependencies



```powershell

.\\.venv\\Scripts\\python.exe -m pip install --upgrade pip

.\\.venv\\Scripts\\python.exe -m pip install -r requirement.txt

```



If the dependency file has a different name in the published repository, use that exact filename.



\### 4. Launch the dashboard



```powershell

.\\.venv\\Scripts\\python.exe -m streamlit run app.py

```



Streamlit will provide a local URL in the terminal. Open it in your browser to use the VisionX interface.



\### 5. Check available image-inference options



```powershell

.\\.venv\\Scripts\\python.exe scripts\\run\_image.py --help

```



Use the displayed arguments to select an image, backend, confidence threshold, and inference size.



> \*\*Model availability:\*\* VisionX uses pretrained YOLO weights. Make sure the required model weights are available locally or allow the model library to retrieve them when supported. Generated model files and large binary weights may be excluded from the Git repository.



\## 🖥️ Supported Inference Backends



| Backend | Execution target | Purpose |

|---|---|---|

| `pytorch\_cpu` | CPU | Baseline inference using PyTorch |

| `openvino\_cpu` | CPU | Optimized inference using OpenVINO |

| `openvino\_gpu` | Compatible Intel GPU | Hardware-accelerated inference through OpenVINO |



The Intel GPU backend depends on compatible hardware, drivers, and OpenVINO device support. It is not equivalent to CUDA and does not require an NVIDIA GPU.



\### Why multiple backends?



The same detection model can behave differently in latency and throughput depending on the inference runtime, device, input resolution, and execution environment.



VisionX makes these implementation choices visible rather than treating the model as a black box.



\## 📈 Experiments



\### 1. Inference backend comparison



An initial experiment processed 21 local images at three input resolutions.



| Backend | Input size | Mean latency | Estimated FPS |

|---|---:|---:|---:|

| PyTorch CPU | 320 | 50.64 ms | 19.75 |

| PyTorch CPU | 480 | 79.37 ms | 12.60 |

| PyTorch CPU | 640 | 93.19 ms | 10.73 |

| OpenVINO CPU | 320 | 40.06 ms | 24.96 |

| OpenVINO CPU | 480 | 38.12 ms | 26.23 |

| OpenVINO CPU | 640 | 39.47 ms | 25.34 |

| OpenVINO Intel GPU | 320 | 35.38 ms | 28.26 |

| OpenVINO Intel GPU | 480 | 37.08 ms | 26.97 |

| OpenVINO Intel GPU | 640 | 37.67 ms | 26.54 |



\*\*Interpretation:\*\* In this initial sample, OpenVINO had lower measured per-image latency than the PyTorch CPU baseline. The Intel GPU backend also showed promising latency.



These results are preliminary, not a general performance guarantee.



\*\*Important limitations:\*\*



\- The sample contained only 21 images and was not a representative benchmark dataset.

\- Estimated FPS is calculated from single-image latency; it is not a measurement of sustained video-stream throughput.

\- Detection counts do not measure accuracy.

\- Hardware load, warm-up, image content, and runtime conditions can affect measurements.



\### 2. Detection versus tracking



VisionX also includes an experiment comparing two processing modes on the same video:



\- \*\*Tracking OFF:\*\* Run object detection independently on each frame.

\- \*\*Tracking ON:\*\* Run detection with ByteTrack to associate objects across frames.



The experiment records:



\- Mean processing latency

\- Median and 95th-percentile latency

\- Estimated processing throughput

\- Frame-level detection counts

\- Number of unique track IDs

\- A track-label consistency proxy



Tracking can add useful temporal information, but it may also introduce computational overhead. Track IDs and label consistency are not, by themselves, proof of correct tracking or improved detection accuracy.



A rigorous evaluation of detection accuracy would require suitable ground-truth annotations and metrics such as precision, recall, or mAP. Tracking quality would require appropriate identity-aware metrics and annotated sequences.



\### Research direction



\*\*Working research question:\*\*



\*How does adding multi-object tracking affect processing performance and track-level label consistency in a real-time object detection pipeline on consumer hardware?\*



This question motivates further experiments under controlled conditions, including repeated runs, representative videos, and ground-truth-based evaluation.



\## 🔬 Reproducibility and Evaluation



The project is being developed with an emphasis on transparent engineering measurements.



A meaningful comparison should keep the following consistent:



\- Input video and processed frame range

\- Detection model and confidence threshold

\- Input resolution

\- Warm-up procedure

\- Measurement boundaries

\- Hardware and software environment



Repeated runs help reveal variability. Mean latency alone can hide slow frames, which is why median and p95 latency are useful complementary metrics.



When reporting results, distinguish \*\*inference latency\*\*, \*\*processing throughput\*\*, and \*\*end-to-end application FPS\*\*. They measure different things.



\## 🛣️ Roadmap



\- \[x] Build an interactive image-detection interface

\- \[x] Add multiple inference backend options

\- \[x] Integrate video and webcam workflows

\- \[x] Explore object tracking with track IDs

\- \[x] Develop a tracking ON/OFF comparison script

\- \[x] Record initial inference measurements

\- \[ ] Improve automated tests and error handling

\- \[ ] Add clearer experiment reports and performance plots

\- \[ ] Evaluate on a representative, documented video dataset

\- \[ ] Add ground-truth-based evaluation where annotations are available

\- \[ ] Document reproducible setup and deployment procedures



\## 🎓 What This Project Demonstrates



VisionX is an exploration of practical computer vision engineering, combining:



\- Model integration and inference optimization

\- CPU and integrated-GPU deployment considerations

\- Real-time image and video processing

\- Multi-object tracking

\- Performance measurement and experimental design

\- Interactive application development

\- Critical interpretation of benchmark results



The objective is not just to make an AI model run, but to understand how the complete system behaves and how its performance should be evaluated.



\## 🤝 Contributing



Suggestions, bug reports, and improvements are welcome.



1\. Fork the repository.

2\. Create a feature branch.

3\. Make a focused change.

4\. Test the change in your environment.

5\. Open a pull request describing the change and how it was tested.



\## 📜 Acknowledgements



VisionX builds on open-source computer vision and inference technologies, including:



\- \[Ultralytics](https://github.com/ultralytics/ultralytics)

\- \[OpenVINO](https://github.com/openvinotoolkit/openvino)

\- \[PyTorch](https://github.com/pytorch/pytorch)

\- \[OpenCV](https://github.com/opencv/opencv)

\- \[Streamlit](https://github.com/streamlit/streamlit)



Please review the respective licenses and usage terms before redistributing models, weights, or derived assets.



\## 👨‍💻 Author



\*\VAIBHAV BHARDWAJ\*\*



Computer Science \& Data Science | Computer Vision | AI Systems



\- GitHub: \(https://github.com/gradientvb)

\- LinkedIn: \www.linkedin.com/in/vaibhav-bhardwaj-955229295



\---



<p align="center">

&#x20; <strong>VisionX — From object detection to visual intelligence.</strong>

&#x20; <br />

&#x20; Built to experiment. Measured with care. Improved with evidence.

</p>



