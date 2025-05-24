# edge_device_app/bag_counter_module.py
import cv2
from ultralytics import YOLO
import os # To check model path

class BagCounter:
    def __init__(self, model_path, confidence_threshold=0.5):
        self.model = None
        self.confidence_threshold = confidence_threshold
        if not os.path.exists(model_path):
            print(f"BagCounter ERROR: YOLOv8 model file not found at {model_path}")
            return # Model remains None

        try:
            self.model = YOLO(model_path)
            # You can check model.names to see loaded classes if needed
            print(f"BagCounter: YOLOv8 model loaded successfully from {model_path}. Classes: {self.model.names if self.model else 'N/A'}")
        except Exception as e:
            print(f"BagCounter ERROR: Failed to load YOLO model from {model_path}: {e}")
            self.model = None # Ensure model is None on failure

    def count_bags(self, frame):
        if not self.model:
            # print("BagCounter: Model not loaded or failed to load. Returning 0 bags.")
            return 0, frame # Return original frame if model not loaded

        if frame is None:
            print("BagCounter: Input frame is None.")
            return 0, None

        try:
            # verbose=False to reduce console spam from YOLO
            results = self.model.predict(source=frame, conf=self.confidence_threshold, verbose=False)
            
            bag_count = 0
            annotated_frame = results[0].plot() # Gets the frame with bounding boxes and labels

            # Iterate through detected objects
            for result in results: # result is one image's worth of detections
                for box in result.boxes:
                    # class_id = int(box.cls[0])
                    # class_name = self.model.names[class_id]
                    # For POC, let's assume any detection from this model is a "bag"
                    # If your model detects multiple classes, you'd filter here:
                    # if class_name.lower() == 'bag':
                    #    bag_count += 1
                    bag_count += 1 # Increment for each detected box

            # print(f"BagCounter: Detected {bag_count} bags.")
            return bag_count, annotated_frame

        except Exception as e:
            print(f"BagCounter ERROR: Error during bag detection: {e}")
            return 0, frame # Return original frame on error