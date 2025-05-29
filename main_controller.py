# main_controller.py
from ultralytics import YOLO
import cv2
import os
import time
from mqtt_handler import MQTTHandler
from dotenv import load_dotenv
import torch # For GPU check
from datetime import datetime # For timestamps
from zoneinfo import ZoneInfo
import json 

# --- 0. Load Environment Variables ---
load_dotenv()

# --- GPU Check ---
print(f"[INFO] PyTorch version: {torch.__version__}")
print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INFO] CUDA device count: {torch.cuda.device_count()}")
    print(f"[INFO] Current CUDA device: {torch.cuda.current_device()}")
    print(f"[INFO] Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    if not torch.cuda.get_device_properties(0).major >= 3: # Basic check for compute capability
        print("[WARNING] Your GPU's compute capability might be too low for efficient YOLO processing.")
else:
    print("[WARNING] CUDA is not available. Model will run on CPU, expect very slow performance.")


# --- 1. CONFIGURATION (Loaded from .env) ---
MODEL_PATH = os.getenv("MODEL_PATH")
VIDEO_PATH = os.getenv("VIDEO_PATH")

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
TARGET_CLASS_NAME = os.getenv("TARGET_CLASS_NAME", "person")
INFERENCE_IMG_SIZE = int(os.getenv("INFERENCE_IMG_SIZE", "320")) # Add to .env: e.g., 320, 416, 640
FRAME_PROCESSING_INTERVAL = int(os.getenv("FRAME_PROCESSING_INTERVAL", "1")) # Add to .env: 1=all, 2=every other

VEHICLE_NO = os.getenv("VEHICLE_NO", "UNKNOWN_VEHICLE")

try:
    ROI_X1 = int(os.getenv("ROI_X1"))
    ROI_Y1 = int(os.getenv("ROI_Y1"))
    ROI_X2 = int(os.getenv("ROI_X2"))
    ROI_Y2 = int(os.getenv("ROI_Y2"))
    ROI_BOX = (ROI_X1, ROI_Y1, ROI_X2, ROI_Y2)
except (TypeError, ValueError) as e:
    print(f"[ERROR] Invalid or missing ROI coordinates in .env file: {e}. Please define ROI_X1, ROI_Y1, ROI_X2, ROI_Y2.")
    exit()

ROI_COLOR = (255, 0, 0)
ROI_THICKNESS = 10

MQTT_BROKER_ADDRESS = os.getenv("MQTT_BROKER_ADDRESS")
MQTT_BROKER_PORT = os.getenv("MQTT_BROKER_PORT")
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
MQTT_CLIENT_ID_PREFIX = os.getenv("MQTT_CLIENT_ID_PREFIX")
MQTT_BASE_TOPIC = os.getenv("MQTT_BASE_TOPIC")

critical_env_vars = ["MODEL_PATH", "VIDEO_PATH", "MQTT_BROKER_ADDRESS", "MQTT_BROKER_PORT",
                     "MQTT_CLIENT_ID_PREFIX", "MQTT_BASE_TOPIC",
                     "ROI_X1", "ROI_Y1", "ROI_X2", "ROI_Y2"]
missing_vars_check = [var for var in critical_env_vars if not os.getenv(var)]
if missing_vars_check:
    print(f"[ERROR] Critical environment variables missing from .env: {', '.join(missing_vars_check)}")
    exit()

MQTT_TOPIC_ROI_COUNT = f"{MQTT_BASE_TOPIC}/{MQTT_CLIENT_ID_PREFIX}/roi/{TARGET_CLASS_NAME.lower()}/passed_count"
SAVE_OUTPUT_VIDEO = os.getenv("SAVE_OUTPUT_VIDEO", "False").lower() == "true"
SHOW_DISPLAY = os.getenv("SHOW_DISPLAY", "True").lower() == "true"

mqtt_handler = None
cap = None
writer = None
processing_start_time = None
total_target_objects_passed_roi = 0
target_object_ids_in_roi_previously = set()
annotated_frame_prev = None # For displaying during skipped frames


def main():
    global total_target_objects_passed_roi, target_object_ids_in_roi_previously
    global mqtt_handler, cap, writer, processing_start_time, annotated_frame_prev

    output_video_filename_prefix_local = os.getenv("OUTPUT_VIDEO_FILENAME_PREFIX", "processed_video")
    output_video_path_local = f"{output_video_filename_prefix_local}_{TARGET_CLASS_NAME.lower()}_roi_pass.mp4"

    print("[INFO] Initializing MQTT Handler...")
    mqtt_handler = MQTTHandler(
        MQTT_BROKER_ADDRESS, MQTT_BROKER_PORT,
        username=MQTT_USERNAME if MQTT_USERNAME else None,
        password=MQTT_PASSWORD if MQTT_PASSWORD else None,
        client_id_prefix=MQTT_CLIENT_ID_PREFIX
    )
    mqtt_handler.connect()
    time.sleep(0.5)

    print(f"[INFO] Loading Ultralytics YOLO model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        if torch.cuda.is_available():
            print("[INFO] Moving model to GPU.")
            # model.to('cuda') # YOLO model automatically moves to CUDA if available during inference
    except Exception as e:
        print(f"[ERROR] Could not load Ultralytics YOLO model: {e}"); return
    print("[INFO] Ultralytics YOLO model loaded successfully.")

    CLASS_LABELS = model.names if hasattr(model, 'names') else [f"class_{i}" for i in range(80)]
    is_target_class_valid = False
    # (Class validation logic as before) ...
    if isinstance(CLASS_LABELS, dict):
        if TARGET_CLASS_NAME in CLASS_LABELS.values(): is_target_class_valid = True
    elif isinstance(CLASS_LABELS, list):
        if TARGET_CLASS_NAME in CLASS_LABELS: is_target_class_valid = True

    if not is_target_class_valid:
        print(f"[ERROR] TARGET_CLASS_NAME '{TARGET_CLASS_NAME}' not found..."); return
    else:
        print(f"[INFO] Targeting class: '{TARGET_CLASS_NAME}'. Publishing to topic: {MQTT_TOPIC_ROI_COUNT}")


    print(f"[INFO] Opening video file: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH) # Try: cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)
    if not cap.isOpened(): print(f"[ERROR] Could not open video file: {VIDEO_PATH}"); return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Video: {frame_width}x{frame_height} @ {fps_video:.2f} FPS")
    print(f"[INFO] Processing every {FRAME_PROCESSING_INTERVAL} frame(s). Inference size: {INFERENCE_IMG_SIZE}px.")


    if SAVE_OUTPUT_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path_local, fourcc, fps_video, (frame_width, frame_height))
        if writer.isOpened(): print(f"[INFO] Output video will be saved to: {output_video_path_local}")
        else: writer = None; print(f"[ERROR] Could not open video writer.")

    total_frames_processed_in_loop = 0 # Frames on which model.track was called
    total_frames_read = 0 # All frames read from video
    processing_start_time = time.time()
    roi_x1, roi_y1, roi_x2, roi_y2 = ROI_BOX
    frame_skip_counter = 0

    while True:
        if not cap.isOpened(): print("[ERROR] Video capture is not open."); break
        ret, frame_bgr = cap.read()
        if not ret: print("[INFO] End of video stream."); break
        
        total_frames_read += 1
        frame_skip_counter += 1
        
        current_loop_annotated_frame = None # Frame to display/save for this iteration

        if frame_skip_counter % FRAME_PROCESSING_INTERVAL == 0 or FRAME_PROCESSING_INTERVAL == 1:
            total_frames_processed_in_loop +=1
            # --- Perform detection and tracking on this frame ---
            results = model.track(source=frame_bgr, persist=True, conf=CONFIDENCE_THRESHOLD,
                                  imgsz=INFERENCE_IMG_SIZE, half=torch.cuda.is_available(), # only use half if cuda is available
                                  verbose=False)
            annotated_frame = frame_bgr.copy() # Start with original for this processed frame

            # (Your existing detection, ROI checking, and drawing logic here)
            # ... (ensure it uses 'annotated_frame')
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                # ... (loop through detections, draw on 'annotated_frame') ...
                boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy()

                current_frame_target_objects_in_roi_now = 0
                for i in range(len(track_ids)):
                    x1_obj, y1_obj, x2_obj, y2_obj = map(int, boxes_xyxy[i])
                    track_id, cls_id_val, conf_val = track_ids[i], cls_ids[i], confs[i] # Unpack for clarity

                    class_name = CLASS_LABELS.get(cls_id_val, f"CLS_{cls_id_val}") if isinstance(CLASS_LABELS, dict) else \
                                 (CLASS_LABELS[cls_id_val] if 0 <= cls_id_val < len(CLASS_LABELS) else f"CLS_{cls_id_val}")
                    is_target_class = (class_name == TARGET_CLASS_NAME)

                    box_color = (0, 0, 255) if is_target_class else (0, 255, 0)
                    label = f"ID:{track_id} {class_name} {conf_val:.2f}"
                    cv2.rectangle(annotated_frame, (x1_obj, y1_obj), (x2_obj, y2_obj), box_color, 2)
                    cv2.putText(annotated_frame, label, (x1_obj, y1_obj - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                    if is_target_class:
                        obj_center_x, obj_center_y = (x1_obj + x2_obj) // 2, (y1_obj + y2_obj) // 2
                        if roi_x1 < obj_center_x < roi_x2 and roi_y1 < obj_center_y < roi_y2:
                            current_frame_target_objects_in_roi_now += 1
                            if track_id not in target_object_ids_in_roi_previously:
                                total_target_objects_passed_roi += 1
                                target_object_ids_in_roi_previously.add(track_id)
                                # --- CONSTRUCT AND PUBLISH JSON PAYLOAD ---
                                #timestamp = datetime.now(ZoneInfo("Asia/Kolkata"))
                                event_payload = {
                                    "timestamp": datetime.now().isoformat(),
                                    "vehicle_no": VEHICLE_NO,
                                    f"{TARGET_CLASS_NAME.lower()}_count": total_target_objects_passed_roi # Cumulative count
                                    # You could also send current_frame_target_objects_in_roi_now
                                    # "current_target_in_roi": current_frame_target_objects_in_roi_now
                                }
                                payload_str = json.dumps(event_payload)
                                print(f"[ROI EVENT] '{TARGET_CLASS_NAME}' (ID: {track_id}) entered. Publishing: {payload_str}")


                                #print(f"[ROI EVENT] '{TARGET_CLASS_NAME}' (ID: {track_id}) entered. Total: {total_target_objects_passed_roi}")
                                if mqtt_handler and mqtt_handler.is_connected():
                                    #pub_info = mqtt_handler.publish(MQTT_TOPIC_ROI_COUNT, total_target_objects_passed_roi)
                                    pub_info = mqtt_handler.publish(MQTT_TOPIC_ROI_COUNT, payload_str)
                                    if pub_info:
                                        print(f"[DEBUG] MQTT publish call info: MID={pub_info.mid}, RC={pub_info.rc}") # ADD THIS
                                    else:
                                        print("[DEBUG] MQTT publish call returned None (likely not connected or error in handler).")
                                cv2.rectangle(annotated_frame, (x1_obj, y1_obj), (x2_obj, y2_obj), (255, 255,0), 3) # Cyan
                                cv2.circle(annotated_frame, (obj_center_x, obj_center_y), 7, (255, 255,0), -1)
                            else:
                                cv2.circle(annotated_frame, (obj_center_x, obj_center_y), 5, (255, 165, 0), -1) # Orange

            # Draw ROI and text on the (potentially) processed frame
            cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), ROI_COLOR, ROI_THICKNESS)
            # (Text drawing for current/total counts as before, using 'annotated_frame')
            current_count_text = f"Current '{TARGET_CLASS_NAME}' in ROI: {current_frame_target_objects_in_roi_now}"
            total_passed_text_video = f"Total '{TARGET_CLASS_NAME}' Passed ROI: {total_target_objects_passed_roi}"
            text_y_pos1 = roi_y1 - 30 if roi_y1 > 40 else roi_y1 + 20
            text_y_pos2 = roi_y1 - 10 if roi_y1 > 20 else roi_y1 + 40
            cv2.putText(annotated_frame, current_count_text, (roi_x1, text_y_pos1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ROI_COLOR, 2)
            cv2.putText(annotated_frame, total_passed_text_video, (roi_x1, text_y_pos2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
            
            annotated_frame_prev = annotated_frame.copy() # Store the latest processed frame
            current_loop_annotated_frame = annotated_frame

            if frame_skip_counter > 10000: frame_skip_counter = 0 # Reset skip counter periodically
        
        else: # This is a skipped frame
            if annotated_frame_prev is not None:
                current_loop_annotated_frame = annotated_frame_prev # Use last good annotated frame
            else:
                current_loop_annotated_frame = frame_bgr # Fallback to raw frame if no prev

        # FPS Calculation (based on frames read for smoother display FPS)
        current_time_loop = time.time()
        elapsed_time_loop = current_time_loop - (processing_start_time if processing_start_time else current_time_loop)
        if elapsed_time_loop > 0 and current_loop_annotated_frame is not None:
            # Display FPS based on all frames read to give a sense of video playback speed
            fps_display = total_frames_read / elapsed_time_loop
            cv2.putText(current_loop_annotated_frame, f"Display FPS: {fps_display:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
            # Display Processing FPS based on frames actually processed by model
            if total_frames_processed_in_loop > 0 :
                fps_processing = total_frames_processed_in_loop / elapsed_time_loop
                cv2.putText(current_loop_annotated_frame, f"Processing FPS: {fps_processing:.2f}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)


        if SAVE_OUTPUT_VIDEO and writer is not None and current_loop_annotated_frame is not None:
            writer.write(current_loop_annotated_frame)
        
        if SHOW_DISPLAY and current_loop_annotated_frame is not None:
            cv2.imshow(f"Tracking '{TARGET_CLASS_NAME}' in ROI", current_loop_annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Exiting loop due to 'q' press...")
                break
    
    # Summary from main
    processing_end_time = time.time()
    total_processing_time_for_video = processing_end_time - (processing_start_time if processing_start_time else processing_end_time)
    avg_processing_fps = total_frames_processed_in_loop / total_processing_time_for_video if total_processing_time_for_video > 0 and total_frames_processed_in_loop > 0 else 0
    avg_display_fps = total_frames_read / total_processing_time_for_video if total_processing_time_for_video > 0 and total_frames_read > 0 else 0


    print(f"\n[INFO] --- Video Processing Summary (from main) ---")
    print(f"[INFO] Total frames read from video: {total_frames_read}")
    print(f"[INFO] Total frames processed by model: {total_frames_processed_in_loop}")
    if total_frames_read > 0 :
        print(f"[INFO] Total processing time: {total_processing_time_for_video:.2f} seconds.")
        print(f"[INFO] Average Display FPS: {avg_display_fps:.2f}")
        if total_frames_processed_in_loop > 0:
            print(f"[INFO] Average Model Processing FPS: {avg_processing_fps:.2f}")


if __name__ == "__main__":
    # (Critical env var checks as before) ...
    required_env_vars = ["MODEL_PATH", "VIDEO_PATH", "MQTT_BROKER_ADDRESS", "MQTT_BROKER_PORT",
                           "MQTT_CLIENT_ID_PREFIX", "MQTT_BASE_TOPIC",
                           "ROI_X1", "ROI_Y1", "ROI_X2", "ROI_Y2",
                           "INFERENCE_IMG_SIZE", "FRAME_PROCESSING_INTERVAL"] # Added new ones
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"[ERROR] Missing critical environment variables: {', '.join(missing_vars)}. Check .env file.")
        exit()
    
    try:
        main()
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received by main process. Exiting gracefully...")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred in main execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\n[INFO] --- Final Script Cleanup ---")
        if cap is not None and cap.isOpened(): cap.release(); print("[INFO] Video capture released.")
        if writer is not None: writer.release(); print("[INFO] Video writer released.")
        if SHOW_DISPLAY: cv2.destroyAllWindows(); print("[INFO] OpenCV windows destroyed.")
        if mqtt_handler is not None: mqtt_handler.disconnect()
        print(f"[INFO] Final total unique '{TARGET_CLASS_NAME}' objects passed: {total_target_objects_passed_roi}")
        print("[INFO] Script finished.")