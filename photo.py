from ultralytics import YOLO
import cv2
import numpy as np
import os

# --- 1. CONFIGURATION ---
MODEL_PATH = "models/bags.pt"  # Your Ultralytics .pt model
IMAGE_PATH = "image.png"
CONFIDENCE_THRESHOLD = 0.25 # Ultralytics default is often 0.25

# CLASS_LABELS will be automatically loaded by the YOLO model if it's standard
# If you have custom class names not embedded in the model, you might need them.

# --- 2. LOAD YOUR ULTRALYTICS MODEL ---
print(f"[INFO] Loading Ultralytics YOLO model from: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
    # The model automatically moves to GPU if available, or uses CPU.
    # You can force CPU: model = YOLO(MODEL_PATH).to('cpu')
except Exception as e:
    print(f"[ERROR] Could not load Ultralytics YOLO model: {e}")
    print("Ensure MODEL_PATH is correct and points to a valid Ultralytics .pt file.")
    exit()
print("[INFO] Ultralytics YOLO model loaded successfully.")
if hasattr(model, 'names'):
    CLASS_LABELS = model.names # Get class names from the model
    print(f"[INFO] Model class names: {CLASS_LABELS}")
else:
    print("[WARNING] Could not retrieve class names from model. Please define CLASS_LABELS manually.")
    CLASS_LABELS = [f"class_{i}" for i in range(80)] # Fallback

# --- 3. LOAD IMAGE ---
image_bgr = cv2.imread(IMAGE_PATH)
if image_bgr is None:
    print(f"[ERROR] Could not read image from path: {IMAGE_PATH}")
    exit()
original_height, original_width = image_bgr.shape[:2]

# --- 4. PERFORM INFERENCE ---
print("[INFO] Performing inference...")
# Ultralytics model handles preprocessing internally.
# You can also pass a NumPy array (BGR) directly: results = model(image_bgr)
results = model(IMAGE_PATH, conf=CONFIDENCE_THRESHOLD, verbose=False) # verbose=False to reduce output
print("[INFO] Inference complete.")

# --- 5. POST-PROCESSING ---
# `results` is a list of Results objects. For a single image, we take the first one.
result = results[0]
output_image = result.plot() # Ultralytics provides a convenient plot() method

# If you want to draw manually or access raw data:
# output_image_manual = image_bgr.copy()
# detected_count = 0
# for box in result.boxes:
#     class_id = int(box.cls[0].item())
#     confidence = box.conf[0].item()
#     x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#
#     label_text = f"{CLASS_LABELS[class_id]}: {confidence:.2f}"
#     color = (0, 255, 0) # Example: Green
#
#     cv2.rectangle(output_image_manual, (x1, y1), (x2, y2), color, 2)
#     cv2.putText(output_image_manual, label_text, (x1, y1 - 10 if y1 - 10 > 10 else y1 + 20),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#     print(f"[DETECTED] {label_text} at [{x1}, {y1}, {x2-x1}, {y2-y1}]")
#     detected_count += 1
#
# if detected_count == 0:
#    print("[INFO] No objects detected with sufficient confidence.")

# --- 6. SHOW IMAGE ---
if output_image is not None:
     cv2.imshow(f"Ultralytics Detections - {os.path.basename(IMAGE_PATH)}", output_image)
     # cv2.imshow("Manual Detections", output_image_manual) # If drawing manually
     print("\nPress any key in the image window to close it...")
     cv2.waitKey(0)
     cv2.destroyAllWindows()
else:
    print("[INFO] No detections to display.")


if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        print(f"[WARNING] '{IMAGE_PATH}' not found. Creating a dummy image.")
        dummy_img_np = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_img_np, "Replace Me!", (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
        cv2.imwrite(IMAGE_PATH, dummy_img_np)

    if not os.path.exists(MODEL_PATH) and MODEL_PATH == "path/to/your/model.pt":
        print(f"[ERROR] Model file '{MODEL_PATH}' (default placeholder) not found.")
        print("Please update MODEL_PATH in the script to your custom .pt model file.")
        exit()
    elif not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file '{MODEL_PATH}' not found. Please check the path.")
        exit()
    # Script runs top-down