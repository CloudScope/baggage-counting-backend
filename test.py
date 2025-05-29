from ultralytics import YOLO
import cv2
import os
import time # To calculate FPS

# --- 1. CONFIGURATION ---
MODEL_PATH = "models/best.pt"  # Your Ultralytics .pt model
VIDEO_PATH = "demo-video.mov" # Path to your input video

CONFIDENCE_THRESHOLD = 0.35 # Adjust as needed (might need higher for less false positives)
SAVE_OUTPUT_VIDEO = False   # Set to True if you want to save the output
SHOW_DISPLAY = True         # Set to False if you don't want to display

# --- NEW: Define the Region of Interest (ROI) box coordinates ---
# This is a fixed rectangle on the video frame where you want to count objects.
# Format: (x1, y1, x2, y2) representing the top-left (x1, y1) and bottom-right (x2, y2) corners.
# IMPORTANT: You WILL NEED TO ADJUST these coordinates for your specific video dimensions and desired ROI.
# Example values for a 1280x720 frame, creating a box in the middle-ish area:
ROI_BOX = (1830, 150, 2200, 550) # (roi_x1, roi_y1, roi_x2, roi_y2)
ROI_COLOR = (255, 0, 0)         # Blue color for drawing the ROI box
ROI_THICKNESS = 5               # Thickness of the ROI box outline

# Only define OUTPUT_VIDEO_PATH if we intend to save
OUTPUT_VIDEO_PATH = "output_video_with_roi_count.mp4"

# --- 2. LOAD YOUR ULTRALYTICS MODEL ---
print(f"[INFO] Loading Ultralytics YOLO model from: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"[ERROR] Could not load Ultralytics YOLO model: {e}")
    exit()
print("[INFO] Ultralytics YOLO model loaded successfully.")

if hasattr(model, 'names'):
    CLASS_LABELS = model.names
else:
    CLASS_LABELS = [f"class_{i}" for i in range(80)] # Fallback

# --- 3. INITIALIZE VIDEO STREAM ---
print(f"[INFO] Opening video file: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"[ERROR] Could not open video file: {VIDEO_PATH}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"[INFO] Video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS")

writer = None
if SAVE_OUTPUT_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        print(f"[ERROR] Could not open video writer for: {OUTPUT_VIDEO_PATH}. Disabling video saving.")
        SAVE_OUTPUT_VIDEO = False

# --- 4. PROCESS VIDEO FRAMES ---
frame_count = 0 # Counter for current FPS calculation
total_frames_processed = 0 # Counter for total frames processed in the video
processing_start_time = time.time() # Start time for overall video processing FPS

# --- NEW: Unpack ROI coordinates for easier use within the loop ---
roi_x1, roi_y1, roi_x2, roi_y2 = ROI_BOX

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        print("[INFO] End of video stream or error reading frame.")
        break

    total_frames_processed += 1
    # --- NEW: Counter for objects whose center is within the ROI for the current frame ---
    current_frame_objects_in_roi = 0 # Reset this count for each new frame

    # Perform inference
    results = model(frame_bgr, conf=CONFIDENCE_THRESHOLD, verbose=False)
    result = results[0] # Get results for the current frame

    # Get the annotated frame (with YOLO's default drawings like bounding boxes and labels)
    annotated_frame = result.plot() # This returns an RGB frame by default

    # --- NEW: Iterate through detected boxes to check if their center is in the ROI ---
    for box_data in result.boxes: # result.boxes contains data for each detected box
        # Get bounding box coordinates (xyxy format: top-left x, top-left y, bottom-right x, bottom-right y)
        x1, y1, x2, y2 = map(int, box_data.xyxy[0].tolist())

        # Calculate the center point of the detected object's bounding box
        obj_center_x = (x1 + x2) // 2
        obj_center_y = (y1 + y2) // 2

        # Check if the object's center point (obj_center_x, obj_center_y) lies within the defined ROI
        if roi_x1 < obj_center_x < roi_x2 and roi_y1 < obj_center_y < roi_y2:
            # If the center is inside the ROI, increment the counter
            current_frame_objects_in_roi += 1
            # Optional: Draw a small circle at the center of objects within ROI for visual confirmation
            # This helps to see which objects are being counted.
            cv2.circle(annotated_frame, (obj_center_x, obj_center_y), 5, (0, 255, 0), -1) # Green dot

    # --- NEW: Draw the ROI box on the frame ---
    # This is drawn on top of the `annotated_frame` from `result.plot()`
    cv2.rectangle(annotated_frame,          # Image to draw on
                  (roi_x1, roi_y1),         # Top-left corner of ROI
                  (roi_x2, roi_y2),         # Bottom-right corner of ROI
                  ROI_COLOR,                # Color of the ROI box
                  ROI_THICKNESS)            # Thickness of the ROI box line

    # --- NEW: Display the count of objects within the ROI on the frame ---
    count_text = f"Objects in ROI: {current_frame_objects_in_roi}"
    # Position the text slightly above the ROI box's top-left corner
    text_y_position = roi_y1 - 10 if roi_y1 > 20 else roi_y1 + 20 # Adjust to avoid going off-screen
    cv2.putText(annotated_frame,            # Image to draw on
                count_text,                 # Text to display
                (roi_x1, text_y_position),  # Position of the text (bottom-left corner of text)
                cv2.FONT_HERSHEY_SIMPLEX,   # Font type
                0.7,                        # Font scale (size)
                ROI_COLOR,                  # Text color (same as ROI box)
                2)                          # Text thickness

    # Calculate and display FPS (current processing FPS, can fluctuate)
    frame_count += 1
    if frame_count >= 1:
        current_time = time.time()
        elapsed_time_for_fps_calc = current_time - processing_start_time # Use overall start time for smoother FPS
        if elapsed_time_for_fps_calc > 0:
            current_fps = total_frames_processed / elapsed_time_for_fps_calc
            cv2.putText(annotated_frame, f"FPS: {current_fps:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2) # Red color for FPS

    if SAVE_OUTPUT_VIDEO and writer is not None:
        writer.write(annotated_frame)

    if SHOW_DISPLAY:
        cv2.imshow("Video Detections with ROI Count", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit
            print("[INFO] Exiting...")
            break

# --- 5. CLEANUP ---
processing_end_time = time.time()
total_processing_time_for_video = processing_end_time - processing_start_time
avg_fps_for_video = total_frames_processed / total_processing_time_for_video if total_processing_time_for_video > 0 else 0
print(f"[INFO] Processed {total_frames_processed} frames in {total_processing_time_for_video:.2f} seconds.")
print(f"[INFO] Average FPS for entire video: {avg_fps_for_video:.2f}")

cap.release()
if SAVE_OUTPUT_VIDEO and writer is not None:
    writer.release()
    print(f"[INFO] Output video saved successfully: {OUTPUT_VIDEO_PATH}")
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

    # --- NEW: Validate ROI_BOX coordinates (basic check) ---
    if not (isinstance(ROI_BOX, tuple) and len(ROI_BOX) == 4 and all(isinstance(c, int) for c in ROI_BOX)):
        print("[ERROR] ROI_BOX is not properly defined. It should be a tuple of 4 integers: (x1, y1, x2, y2).")
        exit()
    # Ensure x1 < x2 and y1 < y2 for a valid rectangle
    if not (ROI_BOX[0] < ROI_BOX[2] and ROI_BOX[1] < ROI_BOX[3]):
        print("[ERROR] ROI_BOX coordinates are invalid: x1 (left) must be < x2 (right) and y1 (top) must be < y2 (bottom).")
        exit()