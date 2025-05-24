# edge_device_app/main_controller.py
import cv2
import time
from datetime import datetime
import os
import signal

import config # Direct import assuming flat structure for now
from anpr_module import extract_plate_number
from bag_counter_module import BagCounter
from local_logger import log_event_to_csv
from mqtt_handler import MQTTHandler
if config.S3_ENABLED and hasattr(config, 'S3_BUCKET_NAME_LOGS'):
    from mqtt_handler import S3Uploader
else:
    S3Uploader = None

class BaggageCountingSystem:
    def __init__(self):
        print("System: Initializing Baggage Counting System (Single Camera Mode)...")
        self.running = True

        # --- SINGLE CAMERA INITIALIZATION ---
        self.camera = None
        self._init_single_camera() # New method

        # Initialize AI Modules (same as before)
        # ... (ensure self.running is set to False if model loading fails) ...
        print(f"System: Bag Counter Model Path: {config.YOLO_MODEL_PATH}")
        if not os.path.exists(config.YOLO_MODEL_PATH) or not self.running: # Check self.running from camera init
            print(f"System CRITICAL ERROR: YOLO Model not found at {config.YOLO_MODEL_PATH} or camera init failed. Exiting.")
            self.running = False
            return
        self.bag_counter = BagCounter(model_path=config.YOLO_MODEL_PATH,
                                      confidence_threshold=config.YOLO_CONFIDENCE_THRESHOLD)
        if self.bag_counter.model is None:
            print("System CRITICAL ERROR: BagCounter model failed to load. Exiting.")
            self.running = False
            return

        # Initialize MQTT Handler (same as before)
        # ... (ensure mqtt_handler.connect() is called) ...
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


        # Initialize S3 Uploader (same as before)
        # ...
        self.s3_uploader = None
        if config.S3_ENABLED and S3Uploader:
            print(f"System: Initializing S3 Uploader for bucket: {config.S3_BUCKET_NAME_LOGS}")
            self.s3_uploader = S3Uploader(bucket_name=config.S3_BUCKET_NAME_LOGS)
            if self.s3_uploader and not self.s3_uploader.s3_client:
                print("System WARNING: S3Uploader initialized, but S3 client connection failed.")
                self.s3_uploader = None


        self.current_vehicle_plate = None
        self.vehicle_detected_time = None
        self.processing_vehicle = False # True when ANPR confirmed, now counting bags
        self.plate_consecutive_detections = 0
        # self.bag_counting_active_for_vehicle = False # Could be an additional state

        if not os.path.exists(config.LOG_BASE_DIR) and self.running:
            try:
                os.makedirs(config.LOG_BASE_DIR)
                print(f"System: Created base log directory: {config.LOG_BASE_DIR}")
            except OSError as e:
                print(f"System ERROR: Could not create base log directory {config.LOG_BASE_DIR}: {e}")
                self.running = False

        signal.signal(signal.SIGINT, self.shutdown_handler)
        signal.signal(signal.SIGTERM, self.shutdown_handler)
        print("System: Initialization complete. Press Ctrl+C to exit gracefully.")

    def _init_single_camera(self): # New method
        print(f"System: Camera Source: {config.CAMERA_SOURCE}")
        try:
            camera_src = int(config.CAMERA_SOURCE)
        except ValueError:
            camera_src = config.CAMERA_SOURCE
        
        self.camera = cv2.VideoCapture(camera_src)
        if not self.camera.isOpened():
            print(f"System ERROR: Cannot open camera: {camera_src}")
            self.running = False # Critical failure
            return
        print("System: Single camera initialized successfully.")

    def get_csv_log_path(self, vehicle_no): # Same as before
        # ...
        today_str = datetime.now().strftime("%Y-%m-%d")
        daily_log_dir = os.path.join(config.LOG_BASE_DIR, today_str)
        if not os.path.exists(daily_log_dir):
            try: os.makedirs(daily_log_dir)
            except OSError as e: print(f"System ERROR creating daily log dir {daily_log_dir}: {e}"); return None
        safe_vehicle_no = "".join(c if c.isalnum() else "_" for c in vehicle_no)
        return os.path.join(daily_log_dir, f"{safe_vehicle_no}.csv")


    def shutdown_handler(self, signum, frame): # Same as before
        # ...
        print(f"\nSystem: Shutdown signal {signal.Signals(signum).name} received. Cleaning up...")
        self.running = False

    def cleanup(self): # Modified for single camera
        print("System: Releasing resources...")
        if self.camera and self.camera.isOpened():
            self.camera.release()
            print("System: Camera released.")
        cv2.destroyAllWindows()
        if self.mqtt_handler:
            self.mqtt_handler.disconnect()
        print("System: Cleanup complete. Exiting.")

    def run(self):
        if not self.running:
            print("System: Cannot run due to initialization errors.")
            self.cleanup()
            return

        annotated_display_frame = None # For displaying combined annotations

        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    print("System Warning: Failed to get frame from camera. Retrying...")
                    time.sleep(1)
                    # Consider re-initializing camera if persistent
                    if not self.camera.isOpened(): self._init_single_camera()
                    if not self.camera.isOpened(): self.running = False; break
                    continue
                
                current_frame_for_processing = frame.copy() # Work on a copy
                display_frame = frame.copy() # Frame for final display

                # --- ANPR Phase ---
                if not self.processing_vehicle: # If we haven't confirmed a vehicle yet
                    # Define ANPR ROI (if specific, otherwise uses whole frame)
                    # Example: h, w, _ = current_frame_for_processing.shape
                    # anpr_roi_frame = current_frame_for_processing[0:h//2, 0:w] # Top half for ANPR
                    anpr_roi_frame = current_frame_for_processing # Or process the whole frame for ANPR

                    plate = extract_plate_number(anpr_roi_frame)
                    if plate:
                        if self.current_vehicle_plate == plate:
                            self.plate_consecutive_detections += 1
                        else:
                            self.current_vehicle_plate = plate
                            self.plate_consecutive_detections = 1
                        
                        print(f"System: Potential plate '{plate}', count: {self.plate_consecutive_detections}/{config.ANPR_TRIGGER_THRESHOLD}")

                        if self.plate_consecutive_detections >= config.ANPR_TRIGGER_THRESHOLD:
                            print(f"System: Vehicle '{self.current_vehicle_plate}' confirmed. Switching to bag counting mode.")
                            self.processing_vehicle = True # Vehicle confirmed, now focus on bags for this vehicle
                            self.vehicle_detected_time = datetime.now()
                            # No need to do bag counting in this same iteration immediately,
                            # The next loop iteration will enter the `self.processing_vehicle` block.
                    else: # No plate detected
                        if self.current_vehicle_plate is not None:
                            print(f"System: Plate '{self.current_vehicle_plate}' lost. Resetting active plate.")
                        self.current_vehicle_plate = None
                        self.plate_consecutive_detections = 0
                
                # --- Bag Counting Phase (if a vehicle is confirmed) ---
                elif self.processing_vehicle and self.current_vehicle_plate:
                    print(f"System: Bag counting for vehicle '{self.current_vehicle_plate}'...")
                    # Define Bag Counting ROI (if specific)
                    # Example: h, w, _ = current_frame_for_processing.shape
                    # bag_roi_frame = current_frame_for_processing[h//2:h, 0:w] # Bottom half for bags
                    bag_roi_frame = current_frame_for_processing # Or process whole frame for bags

                    bag_count, annotated_bag_frame = self.bag_counter.count_bags(bag_roi_frame)
                    print(f"System: Vehicle '{self.current_vehicle_plate}', Bag Count: {bag_count}")
                    
                    if annotated_bag_frame is not None:
                        display_frame = annotated_bag_frame # Show bag detections

                    # --- LOGGING AND REPORTING ---
                    # This happens ONCE after bag counting for the identified vehicle.
                    # You might want a trigger to "finalize" the count (e.g., vehicle leaves, button press, timeout)
                    # For this POC, let's assume we log/report immediately after the first bag count for the vehicle.
                    
                    csv_filepath = self.get_csv_log_path(self.current_vehicle_plate)
                    if csv_filepath:
                        log_success = log_event_to_csv(self.current_vehicle_plate, bag_count, self.vehicle_detected_time, csv_filepath)
                        if log_success and self.s3_uploader:
                            s3_log_key = f"csv_logs/{self.vehicle_detected_time.strftime('%Y-%m-%d')}/{os.path.basename(csv_filepath)}"
                            self.s3_uploader.upload_file(csv_filepath, s3_log_key)
                    else:
                        print("System ERROR: Could not get CSV log path.")

                    telemetry_data = {
                        "timestamp": self.vehicle_detected_time.isoformat() + "Z",
                        "vehicle_number": self.current_vehicle_plate,
                        "bag_count": int(bag_count),
                        "bay_id": config.BAY_ID
                    }
                    self.mqtt_handler.publish_telemetry(config.BAY_ID, telemetry_data)
                    
                    print(f"System: Processing for '{self.current_vehicle_plate}' complete. Resetting.")
                    # Reset for the next vehicle
                    self.current_vehicle_plate = None
                    self.processing_vehicle = False
                    self.plate_consecutive_detections = 0
                    # Optional: Add a small delay before looking for a new plate
                    # time.sleep(config.PROCESS_LOOP_DELAY * 10) # e.g., 1 second cooldown

                # --- Display Logic ---
                # Add text overlays for current state if desired
                state_text = "Detecting ANPR"
                if self.processing_vehicle and self.current_vehicle_plate:
                    state_text = f"Counting Bags for: {self.current_vehicle_plate}"
                elif self.current_vehicle_plate:
                    state_text = f"Tracking Plate: {self.current_vehicle_plate} ({self.plate_consecutive_detections})"
                
                cv2.putText(display_frame, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # cv2.imshow("Baggage AI System", cv2.resize(display_frame, (960, 540))) # Adjust size as needed

                key = cv2.waitKey(max(1, int(config.PROCESS_LOOP_DELAY * 1000))) & 0xFF
                if key == ord('q'):
                    print("System: 'q' pressed, initiating shutdown.")
                    self.running = False
                    break
        
        except Exception as e:
            print(f"System CRITICAL ERROR in main loop: {e}")
            import traceback
            traceback.print_exc()
            self.running = False
        finally:
            self.cleanup()

if __name__ == "__main__":
    system = BaggageCountingSystem()
    if system.running:
        system.run()
    else:
        print("System: Exiting due to initialization failure(s).")