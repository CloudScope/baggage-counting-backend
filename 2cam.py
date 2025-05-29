# edge_device_app/main_controller.py
import cv2
import time
from datetime import datetime
import os
import signal # For graceful shutdown

# Use relative imports if running as part of a package structure
# from . import config
# from .anpr_module import extract_plate_number
# from .bag_counter_module import BagCounter
# from .local_logger import log_event_to_csv, ensure_log_dir_exists
# from .mqtt_handler import MQTTHandler
# if config.S3_ENABLED:
#     from .mqtt_handler import S3Uploader

# Use direct imports if running this script directly from edge_device_app/
# and other .py files are in the same directory or PYTHONPATH is configured.
import config
from anpr_module import extract_plate_number
from bag_counter_module import BagCounter
from local_logger import log_event_to_csv # ensure_log_dir_exists is used internally by get_csv_log_path
from mqtt_handler import MQTTHandler
if config.S3_ENABLED and hasattr(config, 'S3_BUCKET_NAME_LOGS'): # Check if S3Uploader is available
    # S3Uploader is defined within mqtt_handler.py if boto3 is present
    from mqtt_handler import S3Uploader
else:
    S3Uploader = None


class BaggageCountingSystem:
    def __init__(self):
        print("System: Initializing Baggage Counting System...")
        self.running = True # Flag for graceful shutdown

        # Initialize Cameras
        self.anpr_cam = None
        self.bag_cam = None
        self._init_cameras()

        # Initialize AI Modules
        print(f"System: Bag Counter Model Path: {config.YOLO_MODEL_PATH}")
        if not os.path.exists(config.YOLO_MODEL_PATH):
            print(f"System CRITICAL ERROR: YOLO Model not found at {config.YOLO_MODEL_PATH}. Exiting.")
            self.running = False # Prevent run loop if model is missing
            return
        self.bag_counter = BagCounter(model_path=config.YOLO_MODEL_PATH,
                                      confidence_threshold=config.YOLO_CONFIDENCE_THRESHOLD)
        if self.bag_counter.model is None: # Check if model actually loaded in BagCounter
            print("System CRITICAL ERROR: BagCounter model failed to load. Exiting.")
            self.running = False
            return

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
        if not self.mqtt_handler.connect(): # Attempt initial connection
             print("System WARNING: Failed to connect to MQTT broker on init. Will retry during operations.")


        # Initialize S3 Uploader (if enabled and available)
        self.s3_uploader = None
        if config.S3_ENABLED and S3Uploader:
            print(f"System: Initializing S3 Uploader for bucket: {config.S3_BUCKET_NAME_LOGS}")
            self.s3_uploader = S3Uploader(bucket_name=config.S3_BUCKET_NAME_LOGS)
            if self.s3_uploader and not self.s3_uploader.s3_client: # Check if S3 client initialized successfully
                print("System WARNING: S3Uploader initialized, but S3 client connection failed. S3 uploads might not work.")
                self.s3_uploader = None # Disable if client init failed

        self.current_vehicle_plate = None
        self.vehicle_detected_time = None
        self.processing_vehicle = False
        self.plate_consecutive_detections = 0

        # Ensure base log directory exists using the path from config
        if not os.path.exists(config.LOG_BASE_DIR):
            try:
                os.makedirs(config.LOG_BASE_DIR)
                print(f"System: Created base log directory: {config.LOG_BASE_DIR}")
            except OSError as e:
                print(f"System ERROR: Could not create base log directory {config.LOG_BASE_DIR}: {e}")
                self.running = False # Potentially critical if logging is essential

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown_handler)
        signal.signal(signal.SIGTERM, self.shutdown_handler)
        print("System: Initialization complete. Press Ctrl+C to exit gracefully.")

    def _init_cameras(self):
        print(f"System: ANPR Camera Source: {config.ANPR_CAMERA_SOURCE}")
        try:
            anpr_src = int(config.ANPR_CAMERA_SOURCE) # Check if it's an int (webcam index)
        except ValueError:
            anpr_src = config.ANPR_CAMERA_SOURCE # Assume it's an RTSP URL or file path
        self.anpr_cam = cv2.VideoCapture(anpr_src)
        if not self.anpr_cam.isOpened():
            print(f"System ERROR: Cannot open ANPR camera: {anpr_src}")
            self.running = False; return

        print(f"System: Bag Camera Source: {config.BAG_CAMERA_SOURCE}")
        try:
            bag_src = int(config.BAG_CAMERA_SOURCE)
        except ValueError:
            bag_src = config.BAG_CAMERA_SOURCE
        self.bag_cam = cv2.VideoCapture(bag_src)
        if not self.bag_cam.isOpened():
            print(f"System ERROR: Cannot open Bag camera: {bag_src}")
            self.running = False; return
        print("System: Cameras initialized.")


    def get_csv_log_path(self, vehicle_no):
        # local_logger.ensure_log_dir_exists will use config.LOG_BASE_DIR
        # For this specific file, we need the daily subdirectory and the filename
        today_str = datetime.now().strftime("%Y-%m-%d")
        daily_log_dir = os.path.join(config.LOG_BASE_DIR, today_str)
        
        # Ensure daily log directory exists
        if not os.path.exists(daily_log_dir):
            try:
                os.makedirs(daily_log_dir)
            except OSError as e:
                print(f"System ERROR creating daily log dir {daily_log_dir}: {e}")
                return None # Cannot form path if dir creation fails

        safe_vehicle_no = "".join(c if c.isalnum() else "_" for c in vehicle_no)
        return os.path.join(daily_log_dir, f"{safe_vehicle_no}.csv")


    def shutdown_handler(self, signum, frame):
        print(f"\nSystem: Shutdown signal {signal.Signals(signum).name} received. Cleaning up...")
        self.running = False

    def cleanup(self):
        print("System: Releasing resources...")
        if self.anpr_cam and self.anpr_cam.isOpened():
            self.anpr_cam.release()
            print("System: ANPR camera released.")
        if self.bag_cam and self.bag_cam.isOpened():
            self.bag_cam.release()
            print("System: Bag camera released.")
        cv2.destroyAllWindows()
        if self.mqtt_handler:
            self.mqtt_handler.disconnect()
        print("System: Cleanup complete. Exiting.")


    def run(self):
        if not self.running: # Check if init failed
            print("System: Cannot run due to initialization errors.")
            self.cleanup()
            return

        try:
            while self.running:
                ret_anpr, anpr_frame = self.anpr_cam.read()
                ret_bag, bag_frame_raw = self.bag_cam.read()

                if not ret_anpr:
                    print("System Warning: Failed to get frame from ANPR camera. Retrying...")
                    time.sleep(1) # Brief pause before retrying or re-initializing
                    # Consider re-initializing camera if persistent failure
                    continue
                
                # For POC, display camera feeds (optional)
                # cv2.imshow("ANPR Camera", cv2.resize(anpr_frame, (640,480)))
                # if ret_bag and bag_frame_raw is not None:
                #     cv2.imshow("Bag Camera (Raw)", cv2.resize(bag_frame_raw, (640,480)))

                # --- ANPR Logic ---
                if not self.processing_vehicle:
                    plate = extract_plate_number(anpr_frame)
                    if plate:
                        if self.current_vehicle_plate == plate:
                            self.plate_consecutive_detections +=1
                        else:
                            self.current_vehicle_plate = plate
                            self.plate_consecutive_detections = 1
                        
                        print(f"System: Potential plate '{plate}', detection count: {self.plate_consecutive_detections}/{config.ANPR_TRIGGER_THRESHOLD}")

                        if self.plate_consecutive_detections >= config.ANPR_TRIGGER_THRESHOLD:
                            print(f"System: Vehicle '{self.current_vehicle_plate}' confirmed. Processing baggage...")
                            self.processing_vehicle = True
                            self.vehicle_detected_time = datetime.now()
                            
                            # --- Bag Counting Logic (Triggered) ---
                            if ret_bag and bag_frame_raw is not None:
                                bag_count, annotated_bag_frame = self.bag_counter.count_bags(bag_frame_raw.copy()) # Use a copy for annotation
                                print(f"System: Vehicle '{self.current_vehicle_plate}', Bag Count: {bag_count}")

                                # if annotated_bag_frame is not None:
                                #    cv2.imshow("Detected Bags", cv2.resize(annotated_bag_frame, (640,480)))

                                # Log locally
                                csv_filepath = self.get_csv_log_path(self.current_vehicle_plate)
                                if csv_filepath:
                                    log_success = log_event_to_csv(self.current_vehicle_plate, bag_count, self.vehicle_detected_time, csv_filepath) # base_log_dir is implicit via config
                                    if log_success and self.s3_uploader:
                                        s3_log_key = f"csv_logs/{self.vehicle_detected_time.strftime('%Y-%m-%d')}/{os.path.basename(csv_filepath)}"
                                        self.s3_uploader.upload_file(csv_filepath, s3_log_key)
                                else:
                                    print("System ERROR: Could not determine CSV log path. Logging skipped.")


                                # Send to MQTT/NATS
                                telemetry_data = {
                                    "timestamp": self.vehicle_detected_time.isoformat() + "Z", # ISO 8601 with Z for UTC
                                    "vehicle_number": self.current_vehicle_plate,
                                    "bag_count": int(bag_count), # Ensure it's an int
                                    "bay_id": config.BAY_ID
                                }
                                self.mqtt_handler.publish_telemetry(config.BAY_ID, telemetry_data)
                                
                                print(f"System: Processing for vehicle '{self.current_vehicle_plate}' complete. Resetting for next vehicle.")
                                # Consider a cooldown period or a "vehicle left" trigger for a real system
                                self.current_vehicle_plate = None
                                self.processing_vehicle = False
                                self.plate_consecutive_detections = 0
                                # time.sleep(5) # Optional cooldown before looking for new plate

                            else: # No bag frame available
                                print("System Error: Bag camera frame not available for counting when vehicle was detected.")
                                self.processing_vehicle = False # Allow re-trigger if ANPR is still active
                    else: # No plate detected in this frame
                        if self.current_vehicle_plate is not None: # If we were tracking a plate
                            print(f"System: Plate '{self.current_vehicle_plate}' lost. Resetting active plate.")
                            self.current_vehicle_plate = None
                            self.plate_consecutive_detections = 0
                
                # Key press for manual quit (mainly for when not using SIGINT/SIGTERM directly)
                key = cv2.waitKey(max(1, int(config.PROCESS_LOOP_DELAY * 1000))) & 0xFF
                if key == ord('q'):
                    print("System: 'q' pressed, initiating shutdown.")
                    self.running = False
                    break 
            
        except Exception as e: # Catch-all for unexpected errors in the main loop
            print(f"System CRITICAL ERROR in main loop: {e}")
            import traceback
            traceback.print_exc()
            self.running = False # Trigger shutdown on major error
        finally:
            self.cleanup()

if __name__ == "__main__":
    # This check ensures that if this script is run directly,
    # and it's inside edge_device_app/, Python might need help finding modules
    # if they are not in the current dir. Better to run from parent or set PYTHONPATH.
    # However, with the current flat structure within edge_device_app/, direct imports should work.
    
    # Add edge_device_app to sys.path if running from baggage_ai_poc/
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # if script_dir not in sys.path:
    #    sys.path.insert(0, script_dir)

    system = BaggageCountingSystem()
    if system.running: # Only run if init was successful
        system.run()
    else:
        print("System: Exiting due to initialization failure.")