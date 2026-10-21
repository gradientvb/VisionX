
from ultralytics import YOLO

image_path = (
    "assets/sample_images/"
    "SUNG JIN WOO WALLPAPER.jpg"
)

model = YOLO("yolo11n_openvino_model")

results = model.predict(
    source=image_path,
    device="intel:gpu",
    imgsz=640,
    verbose=False,
)

result = results[0]
print("GPU inference succeeded.")
print("Detected objects:", len(result.boxes))
