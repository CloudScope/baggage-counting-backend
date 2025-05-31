# process_video.py
from ultralytics import YOLO
import cv2
import os
import time
from dotenv import load_dotenv
import torch
import json
from datetime import datetime, timezone
from typing import List, Dict, Any
from dataclasses import dataclass, field
from typing import Tuple, Set, List, Optional, Dict, Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import threading
import uvicorn

from logger_setup import setup_logger # Import the setup function
logger = setup_logger(logger_name='BaggageProcessor') # Initialize the main logger for this module

import mqtt_handler as mh # Alias to avoid name collision if any

# --- 0. Load Environment Variables ---
logger.info("Application starting. Loading environment variables...")
load_dotenv()
logger.info("Environment variables loaded.")

# --- GPU Check ---
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 0:
        try:
            current_device = torch.cuda.current_device()
            logger.info(f"Current CUDA device: {current_device}")
            logger.info(f"Device name: {torch.cuda.get_device_name(current_device)}")
        except Exception as e:
            logger.error(f"Could not get CUDA device details: {e}", exc_info=True)
else:
    logger.warning("CUDA is not available. Model will run on CPU, expect significantly slower performance.")


# --- 1. CONFIGURATION (Loaded from .env) ---
#logger.info("Loading application configuration from environment variables.")
#MODEL_PATH = os.getenv("MODEL_PATH")
#VIDEO_PATH = os.getenv("VIDEO_PATH") 
#
#CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
#TARGET_CLASS_NAME = os.getenv("TARGET_CLASS_NAME", "baggage") # Changed default for context
#INFERENCE_IMG_SIZE = int(os.getenv("INFERENCE_IMG_SIZE", "320"))
#FRAME_PROCESSING_INTERVAL = int(os.getenv("FRAME_PROCESSING_INTERVAL", "1")) # Process every Nth frame
#VEHICLE_NO = os.getenv("VEHICLE_NO", "UNKNOWN_VEHICLE")

#ROIS_CONFIG = []
#MAX_ROIS_TO_CHECK = 1 #3 
#for i in range(1, MAX_ROIS_TO_CHECK + 1):
#    roi_name = os.getenv(f"ROI_{i}_NAME")
#    if roi_name and roi_name.strip(): 
#        try:
#            x1 = int(os.getenv(f"ROI_{i}_X1"))
#            y1 = int(os.getenv(f"ROI_{i}_Y1"))
#            x2 = int(os.getenv(f"ROI_{i}_X2"))
#            y2 = int(os.getenv(f"ROI_{i}_Y2"))
#            if not (x1 < x2 and y1 < y2): 
#                logger.warning(f"Invalid coordinates for ROI_{i} ('{roi_name}'). Skipping this ROI.", extra={"roi_id": f"roi_{i}", "roi_name": roi_name})
#                continue
#            ROIS_CONFIG.append({
#                "name": roi_name, "id": f"roi_{i}", "box": (x1, y1, x2, y2),
#                "color": (0, 255, 0), # Green colors for ROIs
#                "passed_count": 0, 
#                "object_ids_previously_in": set()
#            })
#            logger.info(f"Loaded ROI_{i}: '{roi_name}' with box ({x1},{y1},{x2},{y2})", extra={"roi_id": f"roi_{i}", "roi_name": roi_name})
#        except (TypeError, ValueError) as e:
#            logger.warning(f"Invalid or missing coordinates for ROI_{i} ('{roi_name}'). Error: {e}. Skipping this ROI.", exc_info=False, extra={"roi_id": f"roi_{i}", "roi_name": roi_name})
#            continue
#if not ROIS_CONFIG:
#    logger.critical("No valid ROIs configured in .env file. At least ROI_1_NAME and its coordinates must be set. Exiting.")
#    exit(1) # Exit with error code
#logger.info(f"Successfully loaded {len(ROIS_CONFIG)} ROIs: {[roi['name'] for roi in ROIS_CONFIG]}")

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/reset_counts")
def reset_counts():
    for roi in ROIS_CONFIG:
        roi.passed_count = 0
        roi.object_ids_previously_in.clear()
    logger.info("All ROI counts and tracked IDs have been reset via API.")
    return JSONResponse(content={"status": "reset", "message": "All ROI counts have been reset."})

def run_api():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

@dataclass
class ROI:
    name: str
    id: str
    box: Tuple[int, int, int, int]
    color: Tuple[int, int, int] = (0, 255, 0)
    passed_count: int = 0
    object_ids_previously_in: Set[int] = field(default_factory=set)

def load_rois_from_env(max_rois: int) -> List[ROI]:
    """
    Loads ROI configurations from environment variables.
    Returns a list of ROI dataclass instances.
    """
    rois = []
    for i in range(1, max_rois + 1):
        roi_name = os.getenv(f"ROI_{i}_NAME")
        if not roi_name or not roi_name.strip():
            continue
        try:
            x1 = int(os.getenv(f"ROI_{i}_X1"))
            y1 = int(os.getenv(f"ROI_{i}_Y1"))
            x2 = int(os.getenv(f"ROI_{i}_X2"))
            y2 = int(os.getenv(f"ROI_{i}_Y2"))
            if not (x1 < x2 and y1 < y2):
                logger.warning(
                    f"Invalid coordinates for ROI_{i} ('{roi_name}'). Skipping.",
                    extra={"roi_id": f"roi_{i}", "roi_name": roi_name}
                )
                continue
            roi = ROI(
                name=roi_name,
                id=f"roi_{i}",
                box=(x1, y1, x2, y2)
            )
            rois.append(roi)
            logger.info(
                f"Loaded ROI_{i}: '{roi_name}' with box ({x1},{y1},{x2},{y2})",
                extra={"roi_id": f"roi_{i}", "roi_name": roi_name}
            )
        except (TypeError, ValueError) as e:
            logger.warning(
                f"Invalid or missing coordinates for ROI_{i} ('{roi_name}'). Error: {e}. Skipping.",
                exc_info=False,
                extra={"roi_id": f"roi_{i}", "roi_name": roi_name}
            )
    return rois

# Usage in config section:
MAX_ROIS_TO_CHECK = int(os.getenv("MAX_ROIS_TO_CHECK", "1"))
ROIS_CONFIG = load_rois_from_env(MAX_ROIS_TO_CHECK)
if not ROIS_CONFIG:
    logger.critical("No valid ROIs configured in .env file. At least ROI_1_NAME and its coordinates must be set. Exiting.")
    exit(1)
logger.info(f"Successfully loaded {len(ROIS_CONFIG)} ROIs: {[roi.name for roi in ROIS_CONFIG]}")

ROI_THICKNESS = 10

@dataclass
class AppConfig:
    model_path: str
    video_path: str
    mqtt_broker_address: str
    mqtt_broker_port: str
    mqtt_client_id_prefix: str
    mqtt_base_topic: str
    vehicle_no: str
    target_class_name: str
    inference_img_size: int
    confidence_threshold: float
    frame_processing_interval: int
    save_output_video: bool
    show_display: bool
    output_video_fps: int
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[str] = None
    mqtt_ca_certs: Optional[str] = None


def load_app_config() -> AppConfig:
    """Load and validate all critical configuration from environment variables."""
    required_vars = [
        "MODEL_PATH", "VIDEO_PATH", "MQTT_BROKER_ADDRESS", "MQTT_BROKER_PORT",
        "MQTT_CLIENT_ID_PREFIX", "MQTT_BASE_TOPIC", "VEHICLE_NO"
    ]
    env = {var: os.getenv(var) for var in required_vars}
    missing = [k for k, v in env.items() if not v]
    if missing:
        logger.critical(f"Critical environment variables missing: {', '.join(missing)}. Application cannot start.", extra={"missing_vars": missing})
        exit(1)
    if not os.getenv("ROI_1_NAME"):
        logger.critical("ROI_1_NAME is a critical configuration and is missing. Application cannot start.")
        exit(1)
    return AppConfig(
        model_path=env["MODEL_PATH"],
        video_path=env["VIDEO_PATH"],
        mqtt_broker_address=env["MQTT_BROKER_ADDRESS"],
        mqtt_broker_port=env["MQTT_BROKER_PORT"],
        mqtt_client_id_prefix=env["MQTT_CLIENT_ID_PREFIX"],
        mqtt_base_topic=env["MQTT_BASE_TOPIC"],
        vehicle_no=env["VEHICLE_NO"],
        target_class_name=os.getenv("TARGET_CLASS_NAME", "baggage"),
        inference_img_size=int(os.getenv("INFERENCE_IMG_SIZE", "320")),
        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.35")),
        frame_processing_interval=int(os.getenv("FRAME_PROCESSING_INTERVAL", "1")),
        save_output_video=os.getenv("SAVE_OUTPUT_VIDEO", "False").lower() == "true",
        show_display=os.getenv("SHOW_DISPLAY", "True").lower() == "true",
        output_video_fps=int(os.getenv("OUTPUT_VIDEO_FPS", "10")),
        mqtt_username=os.getenv("MQTT_USERNAME"),
        mqtt_password=os.getenv("MQTT_PASSWORD"),
        mqtt_ca_certs=os.getenv("MQTT_CA_CERTS"),
    )

# Usage:
config = load_app_config()
logger.info(f"Configuration loaded: TargetClass='{config.target_class_name}', Confidence={config.confidence_threshold}, VehicleNo='{config.vehicle_no}'")
logger.info(f"MQTT: Broker={config.mqtt_broker_address}:{config.mqtt_broker_port}, BaseTopic='{config.mqtt_base_topic}'")
logger.info(f"Display: Show={config.show_display}, SaveVideo={config.save_output_video} (FPS: {config.output_video_fps if config.save_output_video else 'N/A'})")

MQTT_EVENT_TOPIC = f"{config.mqtt_base_topic}/{config.mqtt_client_id_prefix}/roi_event"

SAVE_OUTPUT_VIDEO = config.save_output_video
SHOW_DISPLAY = config.show_display
OUTPUT_VIDEO_FPS = config.output_video_fps

logger.info(
    f"Configuration loaded: TargetClass='{config.target_class_name}', Confidence={config.confidence_threshold}, VehicleNo='{config.vehicle_no}'"
)
logger.info(
    f"MQTT: Broker={config.mqtt_broker_address}:{config.mqtt_broker_port}, BaseTopic='{config.mqtt_base_topic}', EventTopic='{MQTT_EVENT_TOPIC}'"
)
logger.info(
    f"Display: Show={SHOW_DISPLAY}, SaveVideo={SAVE_OUTPUT_VIDEO} (FPS: {OUTPUT_VIDEO_FPS if SAVE_OUTPUT_VIDEO else 'N/A'})"
)


# --- Global variables for video processing state ---
mqtt_handler_instance = None
cap = None
writer = None
processing_start_time_global = None # To track overall processing time
annotated_frame_prev = None

# --- RTSP Connection Attempt Logic ---
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY_SECONDS = 5

def attempt_rtsp_connect(rtsp_url):
    global cap
    logger.info(f"Attempting to connect to stream: {rtsp_url}", extra={"rtsp_url": rtsp_url})
    # Environment variable for OpenCV FFMPEG options (RTSP specific)
    # Try UDP first (lower latency but less reliable), TCP is often the fallback
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|analyzeduration;1M|probesize;1M"
    # Alternatively, for TCP only: "rtsp_transport;tcp|analyzeduration;1M|probesize;1M"
    
    cap = cv2.VideoCapture(rtsp_url) # cv2.CAP_FFMPEG can be used to force FFMPEG
    
    if not cap.isOpened():
        logger.warning(f"Failed to open stream on attempt: {rtsp_url}", extra={"rtsp_url": rtsp_url})
        return False
    logger.info(f"Successfully opened stream: {rtsp_url}", extra={"rtsp_url": rtsp_url})
    return True


def main():
    global mqtt_handler_instance, cap, writer, processing_start_time_global, annotated_frame_prev
    # ROIS_CONFIG is global and modified for counts

    output_video_filename_prefix_local = os.getenv("OUTPUT_VIDEO_FILENAME_PREFIX", "processed_rtsp_stream")
    output_video_path_local = f"{output_video_filename_prefix_local}_{config.target_class_name.lower()}_multi_roi_pass.mp4"

     # Variable assignments OUTSIDE the try block
    total_frames_processed_in_loop = 0
    total_frames_read_current_session = 0
    processing_start_time_global = time.time()
    current_session_start_time = time.time()
    frame_skip_counter = 0
    last_log_time = time.time()

    try:
        logger.info("Initializing MQTT Handler...")
        mqtt_handler_instance = mh.MQTTHandler(
            config.mqtt_broker_address, config.mqtt_broker_port,
            username=config.mqtt_username if config.mqtt_username else None,
            password=config.mqtt_password if config.mqtt_password else None,
            client_id_prefix=config.mqtt_client_id_prefix,
        )
        if not mqtt_handler_instance.connect():
            logger.error("Failed to connect to MQTT broker initially. Proceeding without MQTT publishing if it cannot reconnect later.")

        logger.info(f"Loading Ultralytics YOLO model from: {config.model_path}", extra={"model_path": config.model_path})
        try:
            model = YOLO(config.model_path)
            logger.info("Ultralytics YOLO model loaded successfully.")
        except Exception as e:
            logger.critical(f"Could not load Ultralytics YOLO model: {e}. Application cannot continue.", exc_info=True, extra={"model_path": config.model_path})
            return # Exit main

        CLASS_LABELS = model.names if hasattr(model, 'names') else {i: f"class_{i}" for i in range(80)} # Ensure dict for .get

        is_target_class_valid = False
        if isinstance(CLASS_LABELS, dict): # model.names is a dict {idx: name}
            if config.target_class_name in CLASS_LABELS.values(): is_target_class_valid = True
        elif isinstance(CLASS_LABELS, list): # Should not happen with recent ultralytics
            if config.target_class_name in CLASS_LABELS: is_target_class_valid = True

        if not is_target_class_valid:
            logger.critical(
                f"TARGET_CLASS_NAME '{config.target_class_name}' not found in model's class labels. Available: {CLASS_LABELS.values() if isinstance(CLASS_LABELS, dict) else CLASS_LABELS}. Exiting.",
                extra={"target_class": config.target_class_name, "available_classes": list(CLASS_LABELS.values()) if isinstance(CLASS_LABELS, dict) else CLASS_LABELS}
            )
            return
        else:
            logger.info(
                f"Targeting class: '{config.target_class_name}'. Publishing ROI events to: {MQTT_EVENT_TOPIC}",
                extra={"target_class": config.target_class_name, "mqtt_topic": MQTT_EVENT_TOPIC}
            )

        connected_to_stream = False
        for attempt in range(MAX_RECONNECT_ATTEMPTS):
            if attempt_rtsp_connect(config.video_path):
                connected_to_stream = True
                break
            logger.warning(f"Stream connection attempt {attempt + 1}/{MAX_RECONNECT_ATTEMPTS} failed. Retrying in {RECONNECT_DELAY_SECONDS}s...", extra={"attempt": attempt+1})
            time.sleep(RECONNECT_DELAY_SECONDS)

        if not connected_to_stream or cap is None or not cap.isOpened():
            logger.critical(f"Failed to connect to video stream '{config.video_path}' after {MAX_RECONNECT_ATTEMPTS} attempts. Exiting.", extra={"video_path": config.video_path})
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_video = cap.get(cv2.CAP_PROP_FPS)

        if frame_width == 0 or frame_height == 0:
            logger.warning("Could not get frame dimensions from RTSP stream properties. Attempting to read one frame to infer dimensions...")
            ret_test, frame_test = cap.read()
            if ret_test and frame_test is not None:
                frame_height, frame_width, _ = frame_test.shape
                logger.info(f"Inferred frame dimensions from first frame: {frame_width}x{frame_height}")
            else:
                logger.error("Still could not get frame dimensions after reading a test frame. Output video saving might fail. Continuing...", extra={"video_path": config.video_path})

        logger.info(f"Stream properties: {frame_width}x{frame_height} @ {fps_video if fps_video > 0 else 'N/A (RTSP)'} FPS",
                    extra={"width": frame_width, "height": frame_height, "fps": fps_video})
        logger.info(f"Processing every {config.frame_processing_interval} frame(s). Inference image size: {config.inference_img_size}px.",
                    extra={"frame_interval": config.frame_processing_interval, "inference_size": config.inference_img_size})

        if SAVE_OUTPUT_VIDEO:
            if frame_width > 0 and frame_height > 0:
                effective_fps_out = fps_video if fps_video > 0 else OUTPUT_VIDEO_FPS
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') # For .mp4
                try:
                    writer = cv2.VideoWriter(output_video_path_local, fourcc, effective_fps_out, (frame_width, frame_height))
                    if writer.isOpened():
                        logger.info(f"Output video will be saved to: {output_video_path_local} @ {effective_fps_out} FPS", extra={"output_path": output_video_path_local, "output_fps": effective_fps_out})
                    else:
                        writer = None
                        logger.error(f"Could not open video writer for path: {output_video_path_local}. Video will not be saved.", extra={"output_path": output_video_path_local})
                except Exception as e:
                    writer = None
                    logger.error(f"Exception opening video writer: {e}", exc_info=True, extra={"output_path": output_video_path_local})
            else:
                logger.error("Cannot save output video due to invalid frame dimensions (0x0).", extra={"save_video_enabled": SAVE_OUTPUT_VIDEO})


        total_frames_processed_in_loop = 0
        total_frames_read_current_session = 0 # Frames read since last (re)connect
        processing_start_time_global = time.time() # For overall app uptime FPS
        current_session_start_time = time.time() # For current stream session FPS
        frame_skip_counter = 0
        last_log_time = time.time()

        logger.info("Starting main video processing loop...")
        while True:
            try: # Main processing loop try-catch
                if not cap.isOpened():
                    logger.error("RTSP stream disconnected or not open. Attempting to reconnect...", extra={"video_path": config.video_path})
                    if cap: cap.release() 
                    reconnected = False
                    for rec_attempt in range(MAX_RECONNECT_ATTEMPTS):
                        logger.info(f"Reconnect attempt {rec_attempt + 1}/{MAX_RECONNECT_ATTEMPTS} for {config.video_path}")
                        if attempt_rtsp_connect(config.video_path):
                            reconnected = True
                            # Re-fetch frame dimensions if they could have changed (unlikely for fixed RTSP)
                            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            # Reset session counters upon successful reconnect
                            total_frames_read_current_session = 0
                            total_frames_processed_in_loop = 0 # Or decide if this should be cumulative
                            frame_skip_counter = 0
                            current_session_start_time = time.time()
                            break
                        logger.warning(f"RTSP reconnect attempt {rec_attempt + 1} failed. Retrying in {RECONNECT_DELAY_SECONDS}s...")
                        time.sleep(RECONNECT_DELAY_SECONDS)
                    if not reconnected:
                        logger.critical(f"Failed to reconnect to RTSP stream '{config.video_path}' after {MAX_RECONNECT_ATTEMPTS} attempts. Exiting loop.", extra={"video_path": config.video_path})
                        break 
                    else:
                        logger.info(f"Successfully reconnected to RTSP stream: {config.video_path}", extra={"video_path": config.video_path})

                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None: 
                    logger.warning("Could not read frame from RTSP stream (ret=False or frame is None). Stream might be closing or temporarily unavailable. Continuing to next read attempt.", extra={"video_path": config.video_path})
                    time.sleep(0.1) # Small delay before trying to read again or letting reconnect logic handle
                    continue 

                total_frames_read_current_session += 1
                frame_skip_counter += 1
                current_loop_annotated_frame = None # Frame to display/save

                # --- Model Inference and Object Tracking ---
                if frame_skip_counter >= config.frame_processing_interval: # Check if it's time to process
                    frame_skip_counter = 0 # Reset counter for next interval
                    total_frames_processed_in_loop +=1

                    # Perform tracking
                    results = model.track(
                        source=frame_bgr,
                        persist=True,
                        conf=config.confidence_threshold,
                        imgsz=config.inference_img_size,
                        half=torch.cuda.is_available(),
                        verbose=False,
                    )
                    annotated_frame_processing = frame_bgr.copy() # Annotate on a copy for this processed frame
                    current_target_in_roi_counts_display = {roi_cfg.id: 0 for roi_cfg in ROIS_CONFIG}


                    if results and results[0].boxes is not None and results[0].boxes.id is not None:
                        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
                        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int) # Class indices
                        confs = results[0].boxes.conf.cpu().numpy()

                        for i in range(len(track_ids)):
                            x1_obj, y1_obj, x2_obj, y2_obj = map(int, boxes_xyxy[i])
                            track_id, cls_id_val, conf_val = track_ids[i], cls_ids[i], confs[i]

                            # Get class name from CLASS_LABELS dictionary
                            class_name = CLASS_LABELS.get(cls_id_val, f"CLS_ID_{cls_id_val}")
                            is_target_class = (class_name == config.target_class_name)

                            # Draw bounding box and label for all detected objects
                            box_color_obj = (0, 0, 255) if is_target_class else (0, 128, 0) # Red for target, Green for others
                            label_obj = f"ID:{track_id} {class_name} {conf_val:.2f}"
                            cv2.rectangle(annotated_frame_processing, (x1_obj, y1_obj), (x2_obj, y2_obj), box_color_obj, 2)
                            cv2.putText(annotated_frame_processing, label_obj, (x1_obj, y1_obj - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color_obj, 2)

                            if is_target_class:
                                obj_center_x, obj_center_y = (x1_obj + x2_obj) // 2, (y1_obj + y2_obj) // 2

                                for roi_cfg in ROIS_CONFIG: # Iterate through defined ROIs
                                    roi_x1_cfg, roi_y1_cfg, roi_x2_cfg, roi_y2_cfg = roi_cfg.box

                                    # Check if object center is within this ROI
                                    if roi_x1_cfg < obj_center_x < roi_x2_cfg and roi_y1_cfg < obj_center_y < roi_y2_cfg:
                                        current_target_in_roi_counts_display[roi_cfg.id] += 1

                                        if track_id not in roi_cfg.object_ids_previously_in:
                                            roi_cfg.passed_count += 1
                                            roi_cfg.object_ids_previously_in.add(track_id)

                                            logger.info(
                                                f"ROI '{roi_cfg.name}': Total '{config.target_class_name}' passed = {roi_cfg.passed_count}",
                                                extra={"roi_name": roi_cfg.name, "final_count": roi_cfg.passed_count}
                                            )

                                            timestamp = int(datetime.now().timestamp())
                                            event_payload = {
                                                "timestamp": timestamp,
                                                "vehicle_no": config.vehicle_no,
                                                "roi_name": roi_cfg.name,
                                                f"{config.target_class_name.lower()}_count_in_roi": roi_cfg.passed_count
                                            }
                                            payload_str = json.dumps(event_payload)

                                            logger.info(
                                                f"ROI Event: '{config.target_class_name}' (ID: {track_id}) entered ROI '{roi_cfg.name}'. Total in ROI: {roi_cfg.passed_count}",
                                                extra={
                                                    "event_type": "roi_entry",
                                                    "target_class": config.target_class_name,
                                                    "object_id": track_id,
                                                    "roi_name": roi_cfg.name,
                                                    "roi_total_count": roi_cfg.passed_count,
                                                    "vehicle_no": config.vehicle_no,
                                                    "payload_preview": payload_str[:150]
                                                }
                                            )
                                            logger.debug(f"Full MQTT payload for ROI event: {payload_str}", extra={"mqtt_payload": event_payload}) # Full payload at debug

                                            if mqtt_handler_instance and mqtt_handler_instance.is_connected():
                                                mqtt_handler_instance.publish(MQTT_EVENT_TOPIC, payload_str)
                                                logger.info("MQTT event published successfully.", extra={"mqtt_topic": MQTT_EVENT_TOPIC, "payload_preview": payload_str[:150]})
                                            else:
                                                logger.warning("MQTT handler is not connected. Event will not be published.", extra={"mqtt_topic": MQTT_EVENT_TOPIC, "payload_preview": payload_str[:150]})                                      
                    # --- Draw all ROI boxes and their respective counts ---
                    text_y_start_offset = 30 
                    for roi_cfg in ROIS_CONFIG:
                        roi_x1_cfg, roi_y1_cfg, roi_x2_cfg, roi_y2_cfg = roi_cfg.box
                        cv2.rectangle(annotated_frame_processing, (roi_x1_cfg, roi_y1_cfg), (roi_x2_cfg, roi_y2_cfg), roi_cfg.color, ROI_THICKNESS)

                        current_text = f"'{roi_cfg.name}': {current_target_in_roi_counts_display.get(roi_cfg.id, 0)} now"
                        total_text = f"Passed '{roi_cfg.name}': {roi_cfg.passed_count}"

                        text_y_current = roi_y1_cfg - text_y_start_offset if roi_y1_cfg > text_y_start_offset + 20 else roi_y1_cfg + 20
                        text_y_total = text_y_current + 20

                        cv2.putText(annotated_frame_processing, current_text, (roi_x1_cfg, text_y_current), cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi_cfg.color, 2)
                        cv2.putText(annotated_frame_processing, total_text, (roi_x1_cfg, text_y_total), cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi_cfg.color, 2)

                    annotated_frame_prev = annotated_frame_processing.copy() # Store this processed frame
                    current_loop_annotated_frame = annotated_frame_processing # Use this for current display/save

                else: # Frame was skipped for processing, use previous annotated frame or raw current if none
                    current_loop_annotated_frame = annotated_frame_prev if annotated_frame_prev is not None else frame_bgr.copy()

                # --- FPS display and video writing/showing ---
                current_time_loop = time.time()
                # Calculate FPS for the current session (since last connect/reconnect)
                elapsed_time_session = current_time_loop - current_session_start_time
                if elapsed_time_session > 0 and current_loop_annotated_frame is not None:
                    # display_fps_session = total_frames_read_current_session / elapsed_time_session
                    processing_fps_session = total_frames_processed_in_loop / elapsed_time_session if total_frames_processed_in_loop > 0 else 0

                    # cv2.putText(current_loop_annotated_frame, f"Session Display FPS: {display_fps_session:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
                    cv2.putText(current_loop_annotated_frame, f"Session Proc. FPS: {processing_fps_session:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)

                    # Log FPS periodically
                    if current_time_loop - last_log_time > 60: # Log every 60 seconds
                        logger.info(f"Current session processing FPS: {processing_fps_session:.2f}", 
                                    extra={"processing_fps": round(processing_fps_session,2), "frames_processed_session": total_frames_processed_in_loop})
                        last_log_time = current_time_loop


                if SAVE_OUTPUT_VIDEO and writer is not None and current_loop_annotated_frame is not None:
                    try:
                        writer.write(current_loop_annotated_frame)
                    except Exception as e:
                        logger.error(f"Error writing frame to video file: {e}", exc_info=True)
                        # Potentially stop trying to write if errors persist

                if SHOW_DISPLAY and current_loop_annotated_frame is not None:
                    try:
                        cv2.imshow(f"Tracking '{config.target_class_name}' in ROIs (RTSP) - Press 'q' to quit", current_loop_annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logger.info("User pressed 'q'. Exiting main processing loop...")
                            break 
                    except Exception as e: # Catch errors like window closed unexpectedly
                        logger.error(f"OpenCV display error: {e}", exc_info=True)
                        # If display fails catastrophically, maybe disable SHOW_DISPLAY or break
                        # SHOW_DISPLAY = False # Example
                        # break

            except KeyboardInterrupt: # Catch Ctrl+C within the loop for graceful exit
                logger.info("KeyboardInterrupt received during main loop. Exiting...")
                break
            except Exception as e: # Catch any other unexpected errors in the loop
                logger.error(f"An unexpected error occurred in the main processing loop: {e}", exc_info=True)
                logger.info("Attempting to continue processing loop after error, or will exit on next failure.")
                time.sleep(1) # Brief pause before trying to continue

        # --- End of main processing loop ---
        logger.info("Main processing loop finished.")

        # --- Summary ---
        processing_end_time_global = time.time()
        total_processing_duration = processing_end_time_global - (processing_start_time_global if processing_start_time_global else processing_end_time_global)

        # Calculate overall average FPS based on total frames processed throughout the app's run
        # This might need adjustment if total_frames_processed_in_loop is reset on reconnect
        # For now, assume total_frames_processed_in_loop is cumulative for this summary.
        # If it's per session, then this average is for the last session.
        avg_overall_processing_fps = total_frames_processed_in_loop / total_processing_duration if total_processing_duration > 0 and total_frames_processed_in_loop > 0 else 0

        logger.info(f"--- Video Processing Summary ---")
        logger.info(f"Total frames processed by model (last session/overall): {total_frames_processed_in_loop}")
        if total_frames_processed_in_loop > 0 and total_processing_duration > 0:
            logger.info(f"Total application processing time: {total_processing_duration:.2f} seconds.")
            logger.info(f"Average Model Processing FPS (last session/overall): {avg_overall_processing_fps:.2f} FPS", extra={"avg_processing_fps": round(avg_overall_processing_fps,2)})

        logger.info("--- Final ROI Counts ---")
        for roi_cfg in ROIS_CONFIG:
            roi_x1_cfg, roi_y1_cfg, roi_x2_cfg, roi_y2_cfg = roi_cfg.box
            cv2.rectangle(annotated_frame_processing, (roi_x1_cfg, roi_y1_cfg), (roi_x2_cfg, roi_y2_cfg), roi_cfg.color, ROI_THICKNESS)

            current_text = f"'{roi_cfg.name}': {current_target_in_roi_counts_display.get(roi_cfg.id, 0)} now"
            total_text = f"Passed '{roi_cfg.name}': {roi_cfg.passed_count}"

            text_y_current = roi_y1_cfg - text_y_start_offset if roi_y1_cfg > text_y_start_offset + 20 else roi_y1_cfg + 20
            text_y_total = text_y_current + 20

            cv2.putText(annotated_frame_processing, current_text, (roi_x1_cfg, text_y_current), cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi_cfg.color, 2)
            cv2.putText(annotated_frame_processing, total_text, (roi_x1_cfg, text_y_total), cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi_cfg.color, 2)

            logger.info(
                f"ROI '{roi_cfg.name}': Total '{config.target_class_name}' passed = {roi_cfg.passed_count}",
                extra={"roi_name": roi_cfg.name, "final_count": roi_cfg.passed_count}
            )
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received during main loop. Exiting...")
    except Exception as e:
        logger.critical(f"An unhandled critical error occurred in main(): {e}", exc_info=True)
    finally:
        logger.info("--- Initiating Final Script Cleanup ---")
        if cap is not None and cap.isOpened():
            cap.release()
            logger.info("Video capture released.")
        if writer is not None:
            writer.release()
            logger.info("Video writer released.")
        if SHOW_DISPLAY:
            try:
                cv2.destroyAllWindows()
                logger.info("OpenCV windows destroyed.")
            except Exception as e:
                logger.warning(f"Error destroying OpenCV windows: {e}", exc_info=False)
        if mqtt_handler_instance is not None:
            logger.info("Disconnecting MQTT handler...")
            mqtt_handler_instance.disconnect()
            logger.info("MQTT handler disconnect called.")
        logger.info("Application finished and cleaned up.")
        import logging
        logging.shutdown()

if __name__ == "__main__":
    # Initial critical environment variable checks are done globally now.
    # The logger is also initialized globally.
    logger.info("Application instance started.")
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    main()
