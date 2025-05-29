from ultralytics import YOLO
import cv2
import os
import time # To calculate FPS

# --- 1. CONFIGURATION ---
MODEL_PATH = "models/best.pt"  # Your Ultralytics .pt model
VIDEO_PATH = "demo-video.mov" # Path to your input video
# OUTPUT_VIDEO_PATH is no longer strictly needed if SAVE_OUTPUT_VIDEO is False

CONFIDENCE_THRESHOLD = 0.25 # Adjust as needed
SAVE_OUTPUT_VIDEO = False   # <<< --- SET TO FALSE: No output video file will be saved
SHOW_DISPLAY = True         # Set to False if you don't want to display (e.g., for benchmarking only)

# Only define OUTPUT_VIDEO_PATH if we intend to save
OUTPUT_VIDEO_PATH = "output_video.mp4" # Default path if SAVE_OUTPUT_VIDEO is True

# --- 2. LOAD YOUR ULTRALYTICS MODEL ---
print(f"[INFO] Loading Ultralytics YOLO model from: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"[ERROR] Could not load Ultralytics YOLO model: {e}")
    print("Ensure MODEL_PATH is correct and points to a valid Ultralytics .pt file.")
    exit()
print("[INFO] Ultralytics YOLO model loaded successfully.")

if hasattr(model, 'names'):
    CLASS_LABELS = model.names
    print(f"[INFO] Model class names: {CLASS_LABELS}")
else:
    print("[WARNING] Could not retrieve class names from model. Using generic labels.")
    CLASS_LABELS = [f"class_{i}" for i in range(80)] # Fallback

# --- 3. INITIALIZE VIDEO STREAM ---
print(f"[INFO] Opening video file: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"[ERROR] Could not open video file: {VIDEO_PATH}")
    exit()

# Get video properties (still useful for display or if saving is re-enabled)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"[INFO] Video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS")

writer = None
if SAVE_OUTPUT_VIDEO:
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
    if writer.isOpened():
        print(f"[INFO] Output video will be saved to: {OUTPUT_VIDEO_PATH}")
    else:
        print(f"[ERROR] Could not open video writer for: {OUTPUT_VIDEO_PATH}. Disabling video saving.")
        SAVE_OUTPUT_VIDEO = False # Force disable if writer fails

# --- 4. PROCESS VIDEO FRAMES ---
frame_count = 0
start_time = time.time()

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        print("[INFO] End of video stream or error reading frame.")
        break

    frame_count += 1

    # Perform inference
    results = model(frame_bgr, conf=CONFIDENCE_THRESHOLD, verbose=False)
    result = results[0]
    annotated_frame = result.plot()

    # Calculate and display FPS (optional)
    current_time = time.time()
    processing_time = current_time - start_time
    if processing_time > 0:
        current_fps = frame_count / processing_time
        cv2.putText(annotated_frame, f"FPS: {current_fps:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Write the frame to the output video file (only if enabled and writer is valid)
    if SAVE_OUTPUT_VIDEO and writer is not None:
        writer.write(annotated_frame)

    # Display the resulting frame (if enabled)
    if SHOW_DISPLAY:
        cv2.imshow("Video Detections", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exiting...")
            break

# --- 5. CLEANUP ---
end_time = time.time()
total_processing_time = end_time - start_time
avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0
print(f"[INFO] Processed {frame_count} frames in {total_processing_time:.2f} seconds.")
print(f"[INFO] Average FPS: {avg_fps:.2f}")

cap.release()
if SAVE_OUTPUT_VIDEO and writer is not None:
    writer.release()
    print(f"[INFO] Output video saved successfully: {OUTPUT_VIDEO_PATH}")
elif SAVE_OUTPUT_VIDEO and writer is None: # Should not happen if logic is correct
    print("[INFO] Video saving was enabled but writer was not initialized or failed.")

if SHOW_DISPLAY:
    cv2.destroyAllWindows()

print("[INFO] Script finished.")


if __name__ == "__main__":
    # Basic checks for paths
    if not os.path.exists(MODEL_PATH) and MODEL_PATH == "path/to/your/model.pt":
        print(f"[ERROR] Default MODEL_PATH '{MODEL_PATH}' used. Please update it.")
        exit()
    elif not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        exit()

    if not os.path.exists(VIDEO_PATH) and VIDEO_PATH == "path/to/your/video.mp4":
        print(f"[ERROR] Default VIDEO_PATH '{VIDEO_PATH}' used. Please update it.")
        exit()
    elif not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Video file not found: {VIDEO_PATH}")
        exit()

    # The script will run when executed directly.