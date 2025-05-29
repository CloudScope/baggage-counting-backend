# edge_device_app/main_controller.py
import cv2
import time
from datetime import datetime
import os
import signal # For graceful shutdown
import traceback # For detailed error logging

import config # Loads .env_edge
from anpr_module import extract_plate_number
from bag_counter_module import BagCounter
from local_logger import log_event_to_csv
from mqtt_handler import MQTTHandler
# S3Uploader is defined within mqtt_handler.py if boto3 is present and S3_ENABLED
if config.S3_ENABLED and hasattr(config, 'S3_BUCKET_NAME_LOGS'):
    try:
        from mqtt_handler import S3Uploader
        if S3Uploader is None and config.S3_ENABLED:
            print("System WARNING: S3_ENABLED is True, but S3Uploader could not be imported (boto3 likely missing).")
    except ImportError:
        S3Uploader = None
        if config.S3_ENABLED:
            print("System WARNING: S3_ENABLED is True, but S3Uploader could not be imported from mqtt_handler.")
else:
    S3Uploader = None


class BaggageCountingSystem:
    def __init__(self):
        print("System: Initializing Baggage Counting System...")
        self.running = True

        # Camera related attributes
        self.camera = None
        self.is_video_file = False
        self.original_fps = 30.0 # Default FPS
        self.video_playback_speed_factor = config.VIDEO_PLAYBACK_SPEED_FACTOR # Loaded from config
        self.delay_per_frame_ms = int(1000 / self.original_fps) # Base delay, will be adjusted
        self.video_loop_count = 0
        self.max_video_loops = 1 # 0 for infinite, N for N loops (can also be made configurable)

        # ANPR and Bag Counting state attributes
        self.current_vehicle_plate = None
        self.vehicle_detected_time = None
        self.processing_vehicle = False
        self.plate_consecutive_detections = 0

        self._init_single_camera() # This will set self.running = False on critical camera error

        if not self.running:
            print("System: Halting initialization due to camera failure.")
            return

        # Initialize AI Modules
        print(f"System: Bag Counter Model Path: {config.YOLO_MODEL_PATH}")
        if not os.path.exists(config.YOLO_MODEL_PATH):
            print(f"System CRITICAL ERROR: YOLO Model not found at {config.YOLO_MODEL_PATH}. Exiting.")
            self.running = False; return
        self.bag_counter = BagCounter(model_path=config.YOLO_MODEL_PATH,
                                      confidence_threshold=config.YOLO_CONFIDENCE_THRESHOLD)
        if self.bag_counter.model is None:
            print("System CRITICAL ERROR: BagCounter model failed to load. Exiting.")
            self.running = False; return

        # Initialize MQTT Handler
        print("System: Initializing MQTT Handler for NATS...")
        self.mqtt_handler = MQTTHandler(
            host=config.NATS_MQTT_BROKER_HOST,
            port=config.NATS_MQTT_BROKER_PORT,
            client_id=config.MQTT_CLIENT_ID_EDGE,
            username=config.NATS_MQTT_USERNAME_EDGE,
            password=config.NATS_MQTT_PASSWORD_EDGE,
            topic_template=config.MQTT_TOPIC_TELEMETRY_TEMPLATE
        )
        if not self.mqtt_handler.connect():
             print("System WARNING: Failed to connect to MQTT broker on init. Will retry.")

        # Initialize S3 Uploader
        self.s3_uploader = None
        if config.S3_ENABLED and S3Uploader:
            print(f"System: Initializing S3 Uploader for bucket: {config.S3_BUCKET_NAME_LOGS}")
            self.s3_uploader = S3Uploader(bucket_name=config.S3_BUCKET_NAME_LOGS)
            if self.s3_uploader and not self.s3_uploader.s3_client:
                print("System WARNING: S3Uploader initialized, but S3 client connection failed.")
                self.s3_uploader = None

        # Ensure base log directory exists
        if not os.path.exists(config.LOG_BASE_DIR):
            try:
                os.makedirs(config.LOG_BASE_DIR)
                print(f"System: Created base log directory: {config.LOG_BASE_DIR}")
            except OSError as e:
                print(f"System ERROR: Could not create base log directory {config.LOG_BASE_DIR}: {e}")
                self.running = False

        signal.signal(signal.SIGINT, self.shutdown_handler)
        signal.signal(signal.SIGTERM, self.shutdown_handler)
        
        if self.running:
            print("System: Initialization complete. Press Ctrl+C in terminal or 'q' in display window to exit.")
        else:
            print("System: Initialization failed. Please check logs above.")


    def _init_single_camera(self):
        print(f"System: Camera Source (from config): {config.CAMERA_SOURCE}")
        camera_source_path = str(config.CAMERA_SOURCE)
        
        try:
            camera_idx = int(camera_source_path)
            self.camera = cv2.VideoCapture(camera_idx)
            self.is_video_file = False
            print(f"System: Initializing live camera with index: {camera_idx}")
            self.original_fps = 30.0 # Assume 30 FPS for live cameras for delay calc if needed
            self.delay_per_frame_ms = max(1, int(config.PROCESS_LOOP_DELAY * 1000)) # Use config delay for live
        except ValueError:
            if os.path.exists(camera_source_path):
                self.is_video_file = True
                print(f"System: Initializing video file input from: {camera_source_path}")
            elif camera_source_path.startswith(("rtsp://", "http://", "https://")):
                self.is_video_file = False
                print(f"System: Initializing live camera stream from: {camera_source_path}")
                self.original_fps = 30.0 # Assume 30 FPS for live streams
                self.delay_per_frame_ms = max(1, int(config.PROCESS_LOOP_DELAY * 1000))
            else:
                print(f"System ERROR: Video file not found at '{camera_source_path}' and not a known stream URL format.")
                self.running = False; return
            
            self.camera = cv2.VideoCapture(camera_source_path)
        
        if not self.camera or not self.camera.isOpened():
            print(f"System ERROR: Cannot open camera/video source: {camera_source_path}")
            self.running = False; return
        
        if self.is_video_file:
            read_fps = self.camera.get(cv2.CAP_PROP_FPS)
            if read_fps > 0:
                self.original_fps = read_fps
            else:
                print(f"System WARNING: Video file FPS is reported as {read_fps}. Using default {self.original_fps} FPS.")
            
            frame_count = int(self.camera.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / self.original_fps if self.original_fps > 0 else 0
            print(f"System: Video file details - Original FPS: {self.original_fps:.2f}, Frames: {frame_count}, Duration: {duration:.2f}s")

            base_delay_ms_for_realtime = int(1000 / self.original_fps) if self.original_fps > 0 else 33
            self.delay_per_frame_ms = max(1, int(base_delay_ms_for_realtime / self.video_playback_speed_factor))
            print(f"System: Playback Speed Factor: {self.video_playback_speed_factor}x. Target delay per frame: {self.delay_per_frame_ms} ms")

            if frame_count <= 0:
                print("System WARNING: Video file seems to have no frames or invalid metadata.")
        else: # Live camera (index or stream)
            print(f"System: Live camera initialized. Target processing delay: {self.delay_per_frame_ms} ms (from PROCESS_LOOP_DELAY).")

    def get_csv_log_path(self, vehicle_no):
        today_str = datetime.now().strftime("%Y-%m-%d")
        daily_log_dir = os.path.join(config.LOG_BASE_DIR, today_str)
        if not os.path.exists(daily_log_dir):
            try: os.makedirs(daily_log_dir)
            except OSError as e: print(f"System ERROR creating daily log dir {daily_log_dir}: {e}"); return None
        safe_vehicle_no = "".join(c if c.isalnum() else "_" for c in vehicle_no)
        return os.path.join(daily_log_dir, f"{safe_vehicle_no}.csv")

    def shutdown_handler(self, signum, frame):
        print(f"\nSystem: Shutdown signal {signal.Signals(signum).name} received. Cleaning up...")
        self.running = False

    def cleanup(self):
        print("System: Releasing resources...")
        if self.camera and self.camera.isOpened():
            self.camera.release()
            print("System: Camera released.")
        cv2.destroyAllWindows()
        if hasattr(self, 'mqtt_handler') and self.mqtt_handler:
            self.mqtt_handler.disconnect()
        print("System: Cleanup complete. Exiting.")

    def run(self):
        if not self.running:
            print("System: Cannot run due to initialization errors.")
            self.cleanup()
            return

        frame_counter = 0
        window_name = "Baggage AI Debug View"

        try:
            while self.running:
                loop_start_time = time.time() # For precise delay calculation

                ret, frame = self.camera.read()
                
                if not ret:
                    if self.is_video_file:
                        self.video_loop_count += 1
                        loops_display = f"{self.max_video_loops}" if self.max_video_loops > 0 else "infinite"
                        print(f"System: End of video file. Loop count: {self.video_loop_count}/{loops_display}")
                        if self.max_video_loops == 0 or self.video_loop_count < self.max_video_loops:
                            print("System: Looping video...")
                            self.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            frame_counter = 0
                            ret, frame = self.camera.read()
                            if not ret: self.running = False; break
                        else:
                            print("System: Maximum video loops reached. Exiting.")
                            self.running = False; break
                    else: # Live camera error
                        print("System Warning: Failed to get frame from live camera. Attempting re-init...")
                        time.sleep(1)
                        if self.camera and self.camera.isOpened(): self.camera.release()
                        self._init_single_camera()
                        if not self.camera or not self.camera.isOpened(): self.running = False; break
                        continue
                
                if frame is None: continue

                frame_counter += 1
                current_frame_for_processing = frame.copy()
                display_frame = frame.copy()

                # --- ANPR Phase ---
                if not self.processing_vehicle:
                    anpr_roi_frame = current_frame_for_processing # Or specific ROI
                    plate = extract_plate_number(anpr_roi_frame)
                    if plate:
                        cv2.putText(display_frame, f"ANPR: {plate}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
                        if self.current_vehicle_plate == plate:
                            self.plate_consecutive_detections += 1
                        else:
                            self.current_vehicle_plate = plate
                            self.plate_consecutive_detections = 1
                        
                        if self.plate_consecutive_detections >= config.ANPR_TRIGGER_THRESHOLD:
                            print(f"System: Frame {frame_counter}: Vehicle '{self.current_vehicle_plate}' confirmed. Switching to bag counting.")
                            self.processing_vehicle = True
                            self.vehicle_detected_time = datetime.now()
                    else: 
                        if self.current_vehicle_plate is not None:
                            self.current_vehicle_plate = None
                            self.plate_consecutive_detections = 0
                
                # --- Bag Counting Phase ---
                elif self.processing_vehicle and self.current_vehicle_plate:
                    bag_roi_frame = current_frame_for_processing # Or specific ROI
                    bag_count, annotated_bag_frame_from_module = self.bag_counter.count_bags(bag_roi_frame)
                    
                    if annotated_bag_frame_from_module is not None:
                        display_frame = annotated_bag_frame_from_module
                    cv2.putText(display_frame, f"Bags: {bag_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                    # --- LOGGING AND REPORTING ---
                    csv_filepath = self.get_csv_log_path(self.current_vehicle_plate)
                    if csv_filepath:
                        log_success = log_event_to_csv(self.current_vehicle_plate, bag_count, self.vehicle_detected_time, csv_filepath, config.LOG_BASE_DIR)
                        if log_success and self.s3_uploader:
                            s3_log_key = f"csv_logs/{self.vehicle_detected_time.strftime('%Y-%m-%d')}/{os.path.basename(csv_filepath)}"
                            self.s3_uploader.upload_file(csv_filepath, s3_log_key)
                    else:
                        print("System ERROR: Could not determine CSV log path. Logging skipped.")

                    telemetry_data = {
                        "timestamp": self.vehicle_detected_time.isoformat() + "Z",
                        "vehicle_number": self.current_vehicle_plate,
                        "bag_count": int(bag_count),
                        "bay_id": config.BAY_ID
                    }
                    self.mqtt_handler.publish_telemetry(config.BAY_ID, telemetry_data)
                    
                    print(f"System: Frame {frame_counter}: Logged/Reported for '{self.current_vehicle_plate}' (Bags: {bag_count}). Resetting.")
                    self.current_vehicle_plate = None
                    self.processing_vehicle = False
                    self.plate_consecutive_detections = 0

                # --- Display Information on Frame ---
                state_text = "ANPR Mode"
                if self.processing_vehicle and self.current_vehicle_plate:
                    state_text = f"Processing: {self.current_vehicle_plate}"
                elif self.current_vehicle_plate:
                    state_text = f"ANPR Tracking: {self.current_vehicle_plate} ({self.plate_consecutive_detections})"
                
                cv2.putText(display_frame, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Frame: {frame_counter} | Speed: {self.video_playback_speed_factor:.1f}x", (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                
                cv2.imshow(window_name, cv2.resize(display_frame, (960, 540)))

                # --- Calculate delay for cv2.waitKey() ---
                calculated_delay_for_waitkey = 1 # Default minimal delay

                if self.is_video_file:
                    # self.delay_per_frame_ms already accounts for the speed factor
                    processing_time_ms = (time.time() - loop_start_time) * 1000
                    wait_time_ms = self.delay_per_frame_ms - processing_time_ms
                    calculated_delay_for_waitkey = max(1, int(wait_time_ms))
                else: # Live camera
                    calculated_delay_for_waitkey = self.delay_per_frame_ms # Use pre-calculated delay for live (from PROCESS_LOOP_DELAY)
                
                key = cv2.waitKey(calculated_delay_for_waitkey) & 0xFF
                if key == ord('q'):
                    print("System: 'q' pressed in window, initiating shutdown.")
                    self.running = False; break
            
        except Exception as e:
            print(f"System CRITICAL ERROR in main loop: {e}")
            traceback.print_exc()
            self.running = False
        finally:
            self.cleanup()

if __name__ == "__main__":
    system = BaggageCountingSystem()
    if system.running:
        system.run()
    else:
        print("System: Exiting due to critical initialization failure(s). Please check logs.")