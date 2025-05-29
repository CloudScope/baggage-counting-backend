# process_video.py
from ultralytics import YOLO
import cv2
import os
import time
from mqtt_handler import MQTTHandler
from dotenv import load_dotenv
import torch
import json
from datetime import datetime, timezone

# --- 0. Load Environment Variables ---
load_dotenv()

# --- GPU Check ---
# ... (GPU check code) ...
print(f"[INFO] PyTorch version: {torch.__version__}")
print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INFO] CUDA device count: {torch.cuda.device_count()}")
    print(f"[INFO] Current CUDA device: {torch.cuda.current_device()}")
    print(f"[INFO] Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("[WARNING] CUDA is not available. Model will run on CPU, expect slow performance.")


# --- 1. CONFIGURATION (Loaded from .env) ---
MODEL_PATH = os.getenv("MODEL_PATH")
VIDEO_PATH = os.getenv("VIDEO_PATH") # RTSP URL or file path

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
TARGET_CLASS_NAME = os.getenv("TARGET_CLASS_NAME", "person")
INFERENCE_IMG_SIZE = int(os.getenv("INFERENCE_IMG_SIZE", "320"))
FRAME_PROCESSING_INTERVAL = int(os.getenv("FRAME_PROCESSING_INTERVAL", "1"))
VEHICLE_NO = os.getenv("VEHICLE_NO", "UNKNOWN_VEHICLE")

# --- Single ROI Configuration ---
ROI_NAME = os.getenv("ROI_NAME", "DefaultROI")
try:
    ROI_X1 = int(os.getenv("ROI_X1"))
    ROI_Y1 = int(os.getenv("ROI_Y1"))
    ROI_X2 = int(os.getenv("ROI_X2"))
    ROI_Y2 = int(os.getenv("ROI_Y2"))
    if not (ROI_X1 < ROI_X2 and ROI_Y1 < ROI_Y2):
        print(f"[ERROR] Invalid ROI coordinates for '{ROI_NAME}'. Check .env. Exiting.")
        exit()
    ROI_BOX = (ROI_X1, ROI_Y1, ROI_X2, ROI_Y2)
except (TypeError, ValueError) as e:
    print(f"[ERROR] Invalid or missing ROI coordinates in .env file: {e}. Exiting.")
    exit()
print(f"[INFO] Loaded ROI: '{ROI_NAME}' with box {ROI_BOX}")

ROI_COLOR = (255, 0, 0) # Blue for the single ROI
ROI_THICKNESS = 10

MQTT_BROKER_ADDRESS = os.getenv("MQTT_BROKER_ADDRESS")
MQTT_BROKER_PORT = os.getenv("MQTT_BROKER_PORT")
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
MQTT_CLIENT_ID_PREFIX = os.getenv("MQTT_CLIENT_ID_PREFIX")
MQTT_BASE_TOPIC = os.getenv("MQTT_BASE_TOPIC")

critical_env_vars = ["MODEL_PATH", "VIDEO_PATH", "MQTT_BROKER_ADDRESS", "MQTT_BROKER_PORT",
                     "MQTT_CLIENT_ID_PREFIX", "MQTT_BASE_TOPIC", "VEHICLE_NO", "ROI_NAME",
                     "ROI_X1", "ROI_Y1", "ROI_X2", "ROI_Y2", # ROI vars now direct
                     "INFERENCE_IMG_SIZE", "FRAME_PROCESSING_INTERVAL"]
missing_vars_check = [var for var in critical_env_vars if not os.getenv(var)]
if missing_vars_check:
    print(f"[ERROR] Critical .env variables missing: {', '.join(missing_vars_check)}")
    exit()

MQTT_EVENT_TOPIC = f"{MQTT_BASE_TOPIC}/{MQTT_CLIENT_ID_PREFIX}/vehicle_event"

SAVE_OUTPUT_VIDEO = os.getenv("SAVE_OUTPUT_VIDEO", "False").lower() == "true"
SHOW_DISPLAY = os.getenv("SHOW_DISPLAY", "True").lower() == "true"
OUTPUT_VIDEO_FPS = int(os.getenv("OUTPUT_VIDEO_FPS", "10"))

mqtt_handler = None
cap = None
writer = None
processing_start_time = None
annotated_frame_prev = None

# --- ROI State Variables (for the single ROI) ---
total_target_objects_passed_roi = 0
object_ids_previously_in_roi = set()


# --- RTSP Connection Attempt Logic (same as before) ---
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY_SECONDS = 5
def attempt_rtsp_connect(rtsp_url):
    global cap
    print(f"[INFO] Attempting to connect to stream: {rtsp_url}")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|analyzeduration;1M|probesize;1M"
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"[WARNING] Failed to open stream: {rtsp_url}")
        return False
    print(f"[INFO] Successfully opened stream: {rtsp_url}")
    return True


def main():
    global total_target_objects_passed_roi, object_ids_previously_in_roi # Modified global state
    global mqtt_handler, cap, writer, processing_start_time, annotated_frame_prev

    output_video_filename_prefix_local = os.getenv("OUTPUT_VIDEO_FILENAME_PREFIX", "processed_rtsp_stream")
    output_video_path_local = f"{output_video_filename_prefix_local}_{TARGET_CLASS_NAME.lower()}_single_roi_pass.mp4"

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
    try: model = YOLO(MODEL_PATH)
    except Exception as e: print(f"[ERROR] Could not load YOLO model: {e}"); return
    print("[INFO] Ultralytics YOLO model loaded successfully.")

    CLASS_LABELS = model.names if hasattr(model, 'names') else [f"class_{i}" for i in range(80)]
    is_target_class_valid = False
    if isinstance(CLASS_LABELS, dict):
        if TARGET_CLASS_NAME in CLASS_LABELS.values(): is_target_class_valid = True
    elif isinstance(CLASS_LABELS, list):
        if TARGET_CLASS_NAME in CLASS_LABELS: is_target_class_valid = True
    if not is_target_class_valid: print(f"[ERROR] TARGET_CLASS_NAME '{TARGET_CLASS_NAME}' not found..."); return
    else: print(f"[INFO] Targeting class: '{TARGET_CLASS_NAME}'. Publishing events to: {MQTT_EVENT_TOPIC}")

    connected_to_stream = False
    for attempt in range(MAX_RECONNECT_ATTEMPTS):
        if attempt_rtsp_connect(VIDEO_PATH): connected_to_stream = True; break
        print(f"[WARNING] Stream connection attempt {attempt + 1}/{MAX_RECONNECT_ATTEMPTS} failed. Retrying..."); time.sleep(RECONNECT_DELAY_SECONDS)
    if not connected_to_stream or cap is None or not cap.isOpened(): print(f"[ERROR] Failed to connect to stream. Exiting."); return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if frame_width == 0 or frame_height == 0:
        ret_test, frame_test = cap.read()
        if ret_test and frame_test is not None: frame_height, frame_width, _ = frame_test.shape; print(f"[INFO] Inferred dims: {frame_width}x{frame_height}")
        else: print("[ERROR] Still could not get frame dimensions. Exiting."); return
    print(f"[INFO] Stream: {frame_width}x{frame_height} @ {fps_video if fps_video > 0 else 'N/A'} FPS")
    print(f"[INFO] Processing every {FRAME_PROCESSING_INTERVAL} frame(s). Inference size: {INFERENCE_IMG_SIZE}px.")

    if SAVE_OUTPUT_VIDEO:
        effective_fps_out = fps_video if fps_video > 0 else OUTPUT_VIDEO_FPS
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path_local, fourcc, effective_fps_out, (frame_width, frame_height))
        if writer.isOpened(): print(f"[INFO] Output video: {output_video_path_local} @ {effective_fps_out} FPS")
        else: writer = None; print(f"[ERROR] Could not open video writer.")

    total_frames_processed_in_loop = 0
    total_frames_read = 0
    processing_start_time = time.time()
    frame_skip_counter = 0
    roi_x1, roi_y1, roi_x2, roi_y2 = ROI_BOX # Unpack for the single ROI

    while True:
        if not cap.isOpened():
            print("[ERROR] Stream disconnected. Attempting reconnect..."); cap.release()
            reconnected = False
            for rec_attempt in range(MAX_RECONNECT_ATTEMPTS):
                if attempt_rtsp_connect(VIDEO_PATH): reconnected = True; frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); break
                print(f"[WARNING] Stream reconnect attempt {rec_attempt + 1} failed. Retrying..."); time.sleep(RECONNECT_DELAY_SECONDS)
            if not reconnected: print("[ERROR] Failed to reconnect. Exiting loop."); break
            else: print("[INFO] Reconnected to stream."); processing_start_time = time.time(); total_frames_read = 0; total_frames_processed_in_loop = 0; frame_skip_counter = 0

        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None: print("[WARNING] Could not read frame."); time.sleep(0.1); continue
        
        total_frames_read += 1
        frame_skip_counter += 1
        current_loop_annotated_frame = None

        if frame_skip_counter % FRAME_PROCESSING_INTERVAL == 0 or FRAME_PROCESSING_INTERVAL == 1:
            total_frames_processed_in_loop +=1
            results = model.track(source=frame_bgr, persist=True, conf=CONFIDENCE_THRESHOLD,
                                  imgsz=INFERENCE_IMG_SIZE, half=torch.cuda.is_available(),
                                  verbose=False)
            annotated_frame = frame_bgr.copy()
            current_target_in_roi_now_display = 0 # For display of current objects in ROI

            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy()

                for i in range(len(track_ids)):
                    x1_obj, y1_obj, x2_obj, y2_obj = map(int, boxes_xyxy[i])
                    track_id, cls_id_val, conf_val = track_ids[i], cls_ids[i], confs[i]
                    class_name = CLASS_LABELS.get(cls_id_val, f"CLS_{cls_id_val}") if isinstance(CLASS_LABELS, dict) else \
                                 (CLASS_LABELS[cls_id_val] if 0 <= cls_id_val < len(CLASS_LABELS) else f"CLS_{cls_id_val}")
                    is_target_class = (class_name == TARGET_CLASS_NAME)

                    box_color_obj = (0, 0, 255) if is_target_class else (0, 255, 0)
                    label_obj = f"ID:{track_id} {class_name} {conf_val:.2f}"
                    cv2.rectangle(annotated_frame, (x1_obj, y1_obj), (x2_obj, y2_obj), box_color_obj, 2)
                    cv2.putText(annotated_frame, label_obj, (x1_obj, y1_obj - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color_obj, 2)

                    if is_target_class:
                        obj_center_x, obj_center_y = (x1_obj + x2_obj) // 2, (y1_obj + y2_obj) // 2
                        if roi_x1 < obj_center_x < roi_x2 and roi_y1 < obj_center_y < roi_y2: # Check against the single ROI
                            current_target_in_roi_now_display += 1
                            if track_id not in object_ids_previously_in_roi: # Use single ROI's set
                                total_target_objects_passed_roi += 1 # Use single ROI's counter
                                object_ids_previously_in_roi.add(track_id)
                                
                                timestamp_iso = datetime.now(timezone.utc).isoformat()
                                event_payload = {
                                    "timestamp": timestamp_iso, "vehicle_no": VEHICLE_NO,
                                    "roi_name": ROI_NAME, # Use the single ROI_NAME from .env
                                    f"{TARGET_CLASS_NAME.lower()}_count": total_target_objects_passed_roi,
                                    "triggering_object_id": track_id
                                }
                                payload_str = json.dumps(event_payload)
                                print(f"[ROI EVENT] '{TARGET_CLASS_NAME}' (ID: {track_id}) entered ROI '{ROI_NAME}'. Pub: {payload_str[:100]}...")
                                if mqtt_handler and mqtt_handler.is_connected():
                                    mqtt_handler.publish(MQTT_EVENT_TOPIC, payload_str)
                                cv2.rectangle(annotated_frame, (x1_obj, y1_obj), (x2_obj, y2_obj), ROI_COLOR, 3) # Use single ROI_COLOR
                                cv2.circle(annotated_frame, (obj_center_x, obj_center_y), 7, ROI_COLOR, -1)
                            else:
                                cv2.circle(annotated_frame, (obj_center_x, obj_center_y), 5, ROI_COLOR, -1)
            
            # Draw the single ROI box and its counts
            cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), ROI_COLOR, ROI_THICKNESS)
            current_count_text = f"Current '{ROI_NAME}': {current_target_in_roi_now_display} now"
            total_text = f"Total Passed '{ROI_NAME}': {total_target_objects_passed_roi}"
            text_y_pos_current = roi_y1 - 30 if roi_y1 > 40 else roi_y1 + 20
            text_y_pos_total = roi_y1 - 10 if roi_y1 > 20 else roi_y1 + 40
            cv2.putText(annotated_frame, current_count_text, (roi_x1, text_y_pos_current), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ROI_COLOR, 2)
            cv2.putText(annotated_frame, total_text, (roi_x1, text_y_pos_total), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ROI_COLOR, 2)

            annotated_frame_prev = annotated_frame.copy()
            current_loop_annotated_frame = annotated_frame
            if frame_skip_counter > 10000: frame_skip_counter = 0
        else:
            current_loop_annotated_frame = annotated_frame_prev if annotated_frame_prev is not None else frame_bgr.copy()

        # (FPS display, video writing/showing as before)
        # ... (same display logic as before) ...
        current_time_loop = time.time()
        elapsed_time_loop = current_time_loop - (processing_start_time if processing_start_time else current_time_loop)
        if elapsed_time_loop > 0 and current_loop_annotated_frame is not None:
            fps_display = total_frames_read / elapsed_time_loop
            cv2.putText(current_loop_annotated_frame, f"Display FPS: {fps_display:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
            if total_frames_processed_in_loop > 0 :
                fps_processing = total_frames_processed_in_loop / elapsed_time_loop
                cv2.putText(current_loop_annotated_frame, f"Proc. FPS: {fps_processing:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)

        if SAVE_OUTPUT_VIDEO and writer is not None and current_loop_annotated_frame is not None:
            writer.write(current_loop_annotated_frame)
        if SHOW_DISPLAY and current_loop_annotated_frame is not None:
            cv2.imshow(f"Tracking '{TARGET_CLASS_NAME}' in ROI '{ROI_NAME}' (RTSP)", current_loop_annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Exiting loop due to 'q' press..."); break
    
    # (Summary print logic as before, adapted for single ROI)
    # ... (same summary print structure as before, but referring to single ROI count) ...
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
    
    print(f"\n[INFO] --- Final ROI Count for '{ROI_NAME}' ---")
    print(f"[INFO] Total '{TARGET_CLASS_NAME}' passed = {total_target_objects_passed_roi}")


if __name__ == "__main__":
    required_env_vars = ["MODEL_PATH", "VIDEO_PATH", "MQTT_BROKER_ADDRESS", "MQTT_BROKER_PORT",
                           "MQTT_CLIENT_ID_PREFIX", "MQTT_BASE_TOPIC", "VEHICLE_NO", "ROI_NAME",
                           "ROI_X1", "ROI_Y1", "ROI_X2", "ROI_Y2",
                           "INFERENCE_IMG_SIZE", "FRAME_PROCESSING_INTERVAL"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"[ERROR] Missing critical environment variables: {', '.join(missing_vars)}. Check .env file.")
        exit()
    
    try:
        main()
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received. Exiting gracefully...")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\n[INFO] --- Final Script Cleanup ---")
        if cap is not None and cap.isOpened(): cap.release(); print("[INFO] Video capture released.")
        if writer is not None: writer.release(); print("[INFO] Video writer released.")
        if SHOW_DISPLAY: cv2.destroyAllWindows(); print("[INFO] OpenCV windows destroyed.")
        if mqtt_handler is not None: mqtt_handler.disconnect()
        print("[INFO] Script finished.")