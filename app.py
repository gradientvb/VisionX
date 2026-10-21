from pathlib import Path
import io
import os
import tempfile
import time
import threading
from collections import Counter, defaultdict, deque

import av
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

import cv2
import pandas as pd
import streamlit as st
from PIL import Image

from src.visionx.detector import BACKENDS, VisionXDetector

st.set_page_config(page_title="VisionX | Visual Intelligence", page_icon="👁️", layout="wide")
st.title("👁️ VisionX")
st.caption("Real-Time Visual Intelligence · Image and Video Detection")

BACKEND_LABELS = {
    "pytorch_cpu": "PyTorch · CPU",
    "openvino_cpu": "OpenVINO · CPU",
    "openvino_gpu": "OpenVINO · Intel GPU",
}
BACKEND_DEVICES = {
    "pytorch_cpu": "cpu",
    "openvino_cpu": "CPU",
    "openvino_gpu": "intel:gpu",
}


@st.cache_resource
def load_detector(backend: str):
    return VisionXDetector(backend=backend, confidence=0.25, image_size=640)


with st.sidebar:
    st.header("Detection Settings")
    backend = st.selectbox(
        "Inference backend",
        options=list(BACKENDS.keys()),
        format_func=lambda value: BACKEND_LABELS.get(value, value),
    )
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    image_size = st.select_slider("Input resolution", options=[320, 480, 640], value=640)
    st.divider()
    track_objects = st.checkbox(
        "Track objects across frames (webcam)",
        value=True,
        help="Associates detections across frames and smooths class labels for each tracked object.",
    )
    st.divider()
    st.caption("VisionX · Computer Vision Research Project")

class VisionXWebcamProcessor(VideoProcessorBase):
    """Browser webcam processor with on-frame object detection and FPS overlay."""

    def __init__(self, backend: str, confidence: float, image_size: int, track_objects: bool = True):
        self.backend = backend
        self.confidence = confidence
        self.image_size = image_size
        self.track_objects = track_objects
        self.device = BACKEND_DEVICES.get(backend, "cpu")
        self.detector = VisionXDetector(
            backend=backend,
            confidence=confidence,
            image_size=image_size,
        )
        self.lock = threading.Lock()
        self.frame_count = 0
        self.started_at = time.perf_counter()
        self.last_inference_ms = 0.0
        self.last_classes = []
        self.class_history = defaultdict(lambda: deque(maxlen=12))

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        start = time.perf_counter()
        if self.track_objects:
            result = self.detector.model.track(
                source=image,
                conf=self.confidence,
                imgsz=self.image_size,
                device=self.device,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )[0]
        else:
            result = self.detector.model.predict(
                source=image,
                conf=self.confidence,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )[0]

        inference_ms = (time.perf_counter() - start) * 1000
        if self.track_objects and result.boxes is not None and result.boxes.id is not None:
            # Draw tracked boxes ourselves so class labels can be smoothed across frames.
            annotated = image.copy()
            boxes_xyxy = result.boxes.xyxy.cpu().tolist()
            class_ids = result.boxes.cls.int().cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
            track_ids = result.boxes.id.int().cpu().tolist()
            for xyxy, class_id, score, track_id in zip(boxes_xyxy, class_ids, scores, track_ids):
                self.class_history[track_id].append(int(class_id))
                smoothed_class = Counter(self.class_history[track_id]).most_common(1)[0][0]
                x1, y1, x2, y2 = map(int, xyxy)
                label = f"ID {track_id} | {result.names[smoothed_class]} {score:.2f}"
                color = (255, 80, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                label_top = max(y1, text_h + 8)
                cv2.rectangle(annotated, (x1, label_top - text_h - 8), (x1 + text_w + 8, label_top + baseline - 2), color, -1)
                cv2.putText(annotated, label, (x1 + 4, label_top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            # Keep the history bounded for long-running webcam sessions.
            if len(self.class_history) > 300:
                active_ids = set(track_ids)
                for old_id in list(self.class_history.keys()):
                    if old_id not in active_ids:
                        del self.class_history[old_id]
        else:
            annotated = result.plot()

        with self.lock:
            self.frame_count += 1
            self.last_inference_ms = inference_ms
            elapsed = max(time.perf_counter() - self.started_at, 1e-6)
            fps = self.frame_count / elapsed

        cv2.putText(
            annotated,
            f"VisionX | {fps:.1f} FPS | {inference_ms:.0f} ms",
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


image_tab, video_tab, webcam_tab = st.tabs([
    "🖼️ Image Detection",
    "🎬 Video Detection",
    "📷 Live Webcam",
])

with image_tab:
    uploaded_image = st.file_uploader(
        "Upload an image to analyze", type=["jpg", "jpeg", "png", "webp"], key="image_uploader"
    )
    if uploaded_image is None:
        st.info("Upload an image to begin object detection.")
    else:
        original = Image.open(uploaded_image).convert("RGB")
        left, right = st.columns(2)
        with left:
            st.subheader("Input Image")
            st.image(original, use_container_width=True)

        if st.button("Run Image Detection", type="primary", use_container_width=True, key="run_image_detection"):
            temp_path = None
            try:
                with st.spinner("Running image detection..."):
                    detector = load_detector(backend)
                    detector.confidence = confidence
                    detector.image_size = image_size
                    suffix = Path(uploaded_image.name).suffix or ".jpg"
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                        temp_file.write(uploaded_image.getvalue())
                        temp_path = temp_file.name
                    start = time.perf_counter()
                    result = detector.detect_image(temp_path)
                    latency_ms = (time.perf_counter() - start) * 1000
                    annotated_rgb = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
                    annotated_image = Image.fromarray(annotated_rgb)
                    detections = []
                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0].item())
                            detections.append({
                                "Class": result.names[class_id],
                                "Confidence": round(float(box.conf[0].item()), 4),
                            })

                with right:
                    st.subheader("Detection Results")
                    st.image(annotated_image, use_container_width=True)
                fps = 1000 / latency_ms if latency_ms > 0 else 0
                metric1, metric2, metric3 = st.columns(3)
                metric1.metric("Detected Objects", len(detections))
                metric2.metric("Inference Latency", f"{latency_ms:.1f} ms")
                metric3.metric("Estimated FPS", f"{fps:.2f}")
                st.caption(f"Backend: {backend} · Resolution: {image_size} · Confidence threshold: {confidence:.2f}")
                st.subheader("Detected Classes")
                if detections:
                    counts = Counter(item["Class"] for item in detections)
                    st.dataframe(pd.DataFrame([{"Class": name, "Count": count} for name, count in counts.items()]), use_container_width=True, hide_index=True)
                    with st.expander("View individual detections"):
                        st.dataframe(pd.DataFrame(detections), use_container_width=True, hide_index=True)
                else:
                    st.warning("No objects passed the selected confidence threshold.")
                output_buffer = io.BytesIO()
                annotated_image.save(output_buffer, format="PNG")
                st.download_button("Download Annotated Image", data=output_buffer.getvalue(), file_name="visionx_detection.png", mime="image/png", use_container_width=True)
            except Exception as error:
                st.error(f"Image detection failed: {error}")
                st.info("Check that the selected model exists and the backend is supported on this computer.")
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

with video_tab:
    st.markdown("Upload a short video to detect objects frame by frame and download an annotated MP4. Processing speed depends on video resolution, backend, and your computer.")
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"], key="video_uploader")
    if uploaded_video is None:
        st.info("Upload a short video to begin frame-by-frame object detection.")
    else:
        st.video(uploaded_video)
        if st.button("Run Video Detection", type="primary", use_container_width=True, key="run_video_detection"):
            input_path = output_path = None
            capture = writer = None
            try:
                with tempfile.NamedTemporaryFile(suffix=Path(uploaded_video.name).suffix or ".mp4", delete=False) as input_file:
                    input_file.write(uploaded_video.getvalue())
                    input_path = input_file.name
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as output_file:
                    output_path = output_file.name

                capture = cv2.VideoCapture(input_path)
                if not capture.isOpened():
                    raise RuntimeError("OpenCV could not open this video. Try a standard MP4 video.")
                source_fps = capture.get(cv2.CAP_PROP_FPS)
                if not source_fps or source_fps <= 0 or source_fps > 240:
                    source_fps = 25.0
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if width <= 0 or height <= 0:
                    raise RuntimeError("The video has invalid frame dimensions.")

                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError("OpenCV could not create an output MP4 on this system.")

                detector = load_detector(backend)
                detector.confidence = confidence
                detector.image_size = image_size
                progress = st.progress(0, text="Preparing video...")
                status = st.empty()
                frame_count = total_detections = 0
                class_counts = Counter()
                inference_seconds = 0.0
                wall_start = time.perf_counter()

                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    inference_start = time.perf_counter()
                    result = detector.model.predict(
                        source=frame,
                        conf=confidence,
                        imgsz=image_size,
                        device=BACKEND_DEVICES.get(backend, "cpu"),
                        verbose=False,
                    )[0]
                    inference_seconds += time.perf_counter() - inference_start
                    writer.write(result.plot())
                    frame_count += 1
                    if result.boxes is not None:
                        total_detections += len(result.boxes)
                        for box in result.boxes:
                            class_counts[result.names[int(box.cls[0].item())]] += 1
                    if total_frames > 0:
                        progress.progress(min(frame_count / total_frames, 1.0), text=f"Processed {frame_count}/{total_frames} frames")
                    elif frame_count % 10 == 0:
                        status.write(f"Processed {frame_count} frames...")

                capture.release()
                capture = None
                writer.release()
                writer = None
                wall_seconds = time.perf_counter() - wall_start
                progress.progress(1.0, text="Video processing complete.")
                status.empty()
                if frame_count == 0 or not os.path.exists(output_path):
                    raise RuntimeError("No video frames were processed.")
                video_bytes = Path(output_path).read_bytes()
                st.subheader("Annotated Video")
                st.video(video_bytes)
                metric1, metric2, metric3, metric4 = st.columns(4)
                metric1.metric("Frames Processed", frame_count)
                metric2.metric("Processing FPS", f"{frame_count / wall_seconds:.2f}" if wall_seconds > 0 else "—")
                metric3.metric("Mean Inference Time", f"{inference_seconds / frame_count * 1000:.1f} ms")
                metric4.metric("Detections Across Frames", total_detections)
                st.caption(f"Backend: {backend} · Resolution: {image_size} · Confidence threshold: {confidence:.2f} · Source FPS: {source_fps:.2f}")
                if class_counts:
                    st.subheader("Detected Classes Across All Frames")
                    st.caption("Counts are per-frame detections; the same object may be counted in multiple frames.")
                    st.dataframe(pd.DataFrame([{"Class": name, "Frame detections": count} for name, count in class_counts.most_common()]), use_container_width=True, hide_index=True)
                else:
                    st.warning("No objects passed the selected confidence threshold in the processed frames.")
                st.download_button("Download Annotated Video", data=video_bytes, file_name="visionx_annotated_video.mp4", mime="video/mp4", use_container_width=True)
            except Exception as error:
                st.error(f"Video detection failed: {error}")
                st.info("Try a short MP4 video first. If the error mentions the backend or device, share the full error so we can adapt video inference for that backend.")
            finally:
                if capture is not None:
                    capture.release()
                if writer is not None:
                    writer.release()
                for temp_path in (input_path, output_path):
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except OSError:
                            pass


with webcam_tab:
    st.subheader("Live Webcam Detection")
    st.markdown(
        "Allow camera access in your browser to run object detection on live "
        "video. The annotated stream includes an approximate FPS and inference "
        "time overlay. Performance depends on your selected backend and hardware."
    )
    st.warning(
        "Camera permission is requested by your browser. If the camera does not "
        "start, check browser permissions and close other apps using the webcam."
    )
    try:
        webrtc_streamer(
            key=f"visionx-webcam-{backend}-{confidence}-{image_size}-{track_objects}",
            video_processor_factory=lambda: VisionXWebcamProcessor(
                backend=backend,
                confidence=confidence,
                image_size=image_size,
                track_objects=track_objects,
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    except Exception as error:
        st.error(f"Could not start webcam detection: {error}")
    st.caption(
        f"Backend: {backend} · Resolution: {image_size} · "
        f"Confidence threshold: {confidence:.2f} · "
        f"Tracking: {'on' if track_objects else 'off'}"
    )
