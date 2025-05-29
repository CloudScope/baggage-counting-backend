from ultralytics import YOLO
import cv2
import os
import time

from mqtt_handler import MQTTHandler

# --- 1. CONFIGURATION ---
MODEL_PATH = "models/best.pt"  # Your Ultralytics .pt model
VIDEO_PATH = "demo-video.mov" # Path to your input video

CONFIDENCE_THRESHOLD = 0.35
SAVE_OUTPUT_VIDEO = False
SHOW_DISPLAY = True

# Define the target class name to track and count
TARGET_CLASS_NAME = "Baggage" # Or "suitcase", "backpack", "person", etc. Case-sensitive.

# Example for a 1280px wide video, box on the right
ROI_BOX = (1830, 150, 2200, 550)# (roi_x1, roi_y1, roi_x2, roi_y2)
ROI_COLOR = (255, 0, 0)         # Blue color for drawing the ROI box <<< --- SYNTAX ERROR FIXED HERE
ROI_THICKNESS = 10

OUTPUT_VIDEO_PATH = "output_video_with_target_class_roi_pass.mp4"

# Variables for tracking TARGET CLASS objects passing through ROI
total_target_objects_passed_roi = 0
target_object_ids_in_roi_previously = set()


# --- 2. LOAD YOUR ULTRALYTICS MODEL ---
print(f"[INFO] Loading Ultralytics YOLO model from: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"[ERROR] Could not load Ultralytics YOLO model: {e}")
    exit()
print("[INFO] Ultralytics YOLO model loaded successfully.")

CLASS_LABELS = model.names if hasattr(model, 'names') else [f"class_{i}" for i in range(80)]

# --- NEW: Validate if TARGET_CLASS_NAME exists in model's classes (IMPROVED) ---
is_target_class_valid = False
if isinstance(CLASS_LABELS, dict): # model.names is usually a dict {id: name}
    if TARGET_CLASS_NAME in CLASS_LABELS.values():
        is_target_class_valid = True
elif isinstance(CLASS_LABELS, list): # Fallback if it's a list
    if TARGET_CLASS_NAME in CLASS_LABELS:
        is_target_class_valid = True

if not is_target_class_valid:
    print(f"[ERROR] TARGET_CLASS_NAME '{TARGET_CLASS_NAME}' not found in model's class labels.")
    print("Please ensure TARGET_CLASS_NAME matches a class name from your model exactly (case-sensitive).")
    if isinstance(CLASS_LABELS, dict):
        print("Available classes from model:", list(CLASS_LABELS.values()))
    else:
        print("Available classes from model:", CLASS_LABELS)
    exit()
else:
    print(f"[INFO] Successfully targeting class: '{TARGET_CLASS_NAME}' for ROI counting.")


# --- 3. INITIALIZE VIDEO STREAM ---
print(f"[INFO] Opening video file: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"[ERROR] Could not open video file: {VIDEO_PATH}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video = cap.get(cv2.CAP_PROP_FPS)
print(f"[INFO] Video properties: {frame_width}x{frame_height} @ {fps_video:.2f} FPS")

writer = None
if SAVE_OUTPUT_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps_video, (frame_width, frame_height))
    if not writer.isOpened(): SAVE_OUTPUT_VIDEO = False

# --- 4. PROCESS VIDEO FRAMES ---
total_frames_processed = 0
processing_start_time = time.time()
roi_x1, roi_y1, roi_x2, roi_y2 = ROI_BOX

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        print("[INFO] End of video stream or error reading frame.")
        break

    total_frames_processed += 1
    current_frame_target_objects_in_roi_now = 0

    results = model.track(source=frame_bgr, persist=True, conf=CONFIDENCE_THRESHOLD, verbose=False)
    annotated_frame = frame_bgr.copy()

    if results and results[0].boxes is not None and results[0].boxes.id is not None:
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()
        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        for i in range(len(track_ids)):
            x1, y1, x2, y2 = map(int, boxes_xyxy[i])
            track_id = track_ids[i]
            conf = confs[i]
            cls_id = cls_ids[i]

            class_name = ""
            if isinstance(CLASS_LABELS, dict):
                class_name = CLASS_LABELS.get(cls_id, f"CLS_ID_{cls_id}")
            elif isinstance(CLASS_LABELS, list):
                 if 0 <= cls_id < len(CLASS_LABELS):
                    class_name = CLASS_LABELS[cls_id]
                 else:
                    class_name = f"CLS_ID_{cls_id}"

            is_target_class = (class_name == TARGET_CLASS_NAME)

            box_color = (0, 0, 255) if is_target_class else (0, 255, 0) # Red for target, Green for others
            label = f"ID:{track_id} {class_name} {conf:.2f}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            if is_target_class:
                obj_center_x = (x1 + x2) // 2
                obj_center_y = (y1 + y2) // 2

                if roi_x1 < obj_center_x < roi_x2 and roi_y1 < obj_center_y < roi_y2:
                    current_frame_target_objects_in_roi_now += 1

                    if track_id not in target_object_ids_in_roi_previously:
                        total_target_objects_passed_roi += 1
                        target_object_ids_in_roi_previously.add(track_id)
                        print(f"[ROI EVENT] '{TARGET_CLASS_NAME}' (ID: {track_id}) entered ROI. Total '{TARGET_CLASS_NAME}' Passed ROI: {total_target_objects_passed_roi}")
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255,0), 3) # Cyan highlight
                        cv2.circle(annotated_frame, (obj_center_x, obj_center_y), 7, (255, 255,0), -1)
                    else:
                        cv2.circle(annotated_frame, (obj_center_x, obj_center_y), 5, (255, 165, 0), -1) # Orange dot

    cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), ROI_COLOR, ROI_THICKNESS)

    current_count_text = f"Current '{TARGET_CLASS_NAME}' in ROI: {current_frame_target_objects_in_roi_now}"
    total_passed_text = f"Total '{TARGET_CLASS_NAME}' Passed ROI: {total_target_objects_passed_roi}"
    text_y_pos1 = roi_y1 - 30 if roi_y1 > 40 else roi_y1 + 20
    text_y_pos2 = roi_y1 - 10 if roi_y1 > 20 else roi_y1 + 40

    cv2.putText(annotated_frame, current_count_text, (roi_x1, text_y_pos1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, ROI_COLOR, 2)
    cv2.putText(annotated_frame, total_passed_text, (roi_x1, text_y_pos2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)

    current_time = time.time()
    elapsed_time = current_time - processing_start_time
    if elapsed_time > 0:
        fps_current = total_frames_processed / elapsed_time
        cv2.putText(annotated_frame, f"FPS: {fps_current:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if SAVE_OUTPUT_VIDEO and writer is not None:
        writer.write(annotated_frame)

    if SHOW_DISPLAY:
        cv2.imshow(f"Tracking '{TARGET_CLASS_NAME}' in ROI", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exiting...")
            break

# --- 5. CLEANUP ---
processing_end_time = time.time()
total_processing_time_for_video = processing_end_time - processing_start_time
avg_fps_for_video = total_frames_processed / total_processing_time_for_video if total_processing_time_for_video > 0 else 0
print(f"[INFO] Processed {total_frames_processed} frames in {total_processing_time_for_video:.2f} seconds.")
print(f"[INFO] Average FPS for entire video: {avg_fps_for_video:.2f}")
print(f"[INFO] Total unique '{TARGET_CLASS_NAME}' objects passed through ROI: {total_target_objects_passed_roi}")


cap.release()
if SAVE_OUTPUT_VIDEO and writer is not None:
    writer.release()
if SHOW_DISPLAY:
    cv2.destroyAllWindows()
print("[INFO] Script finished.")

if __name__ == "__main__":
    # Path checks
    if not os.path.exists(MODEL_PATH) and MODEL_PATH == "path/to/your/model.pt":
        print(f"[ERROR] Default MODEL_PATH used. Update MODEL_PATH."); exit()
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}"); exit()
    if not os.path.exists(VIDEO_PATH) and VIDEO_PATH == "path/to/your/video.mp4":
        print(f"[ERROR] Default VIDEO_PATH used. Update VIDEO_PATH."); exit()
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Video not found: {VIDEO_PATH}"); exit()
    # ROI checks
    if not (isinstance(ROI_BOX, tuple) and len(ROI_BOX) == 4 and all(isinstance(c, int) for c in ROI_BOX)):
        print(f"[ERROR] ROI_BOX is not properly defined."); exit()
    if not (ROI_BOX[0] < ROI_BOX[2] and ROI_BOX[1] < ROI_BOX[3]):
        print(f"[ERROR] ROI_BOX coordinates are invalid."); exit()