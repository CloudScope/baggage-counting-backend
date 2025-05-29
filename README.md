# Real-time Baggage Counting from RTSP Stream with YOLO and NATS MQTT

This project implements a real-time object detection and counting system using YOLO models. It processes an RTSP video stream, identifies target objects within defined Regions of Interest (ROIs), and publishes event data (including object counts, timestamps, and vehicle information) to a NATS MQTT broker.

## Features

-   Processes RTSP video streams or local video files.
-   Utilizes Ultralytics YOLO models for object detection and tracking.
-   Supports configuration of one or more Regions of Interest (ROIs).
-   Tracks specific target objects (e.g., "baggage", "person") passing through each ROI.
-   Publishes structured JSON event data to a NATS MQTT broker for each detected object entering an ROI.
-   MQTT payload includes timestamp, vehicle identifier, ROI name, and cumulative count for the target object in that ROI.
-   Configuration driven by a `.env` file for easy setup and modification.
-   Includes basic RTSP stream reconnection logic.
-   Optimized for potential GPU acceleration.

## Project Structure

```
baggage-counting-backend/
├── .env # Environment configuration (sensitive data, paths)
├── mqtt_handler.py # Python class for handling MQTT communication
├── main_controller.py # Main Python script for video processing and detection
├── README.md # This file
└── models/ # (Optional) If your model is local
    └── bags.pt
└── requirements.txt
```

## Prerequisites

-   Python 3.8+
-   NATS Server (with MQTT listener enabled and JetStream recommended)
-   An RTSP camera stream or a video file.
-   A YOLO model file (e.g., `yolov8n.pt`).

## Setup Instructions

### 1. Clone the Repository (if applicable) or Create Project Directory

```
git clone <your_repository_url> # Or mkdir my_yolo_nats_counter
cd baggage-counting-backend
```
2. Create Python Virtual Environment (Recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Python Dependencies
```
pip install ultralytics opencv-python paho-mqtt python-dotenv uuid torch torchvision torchaudio
```
If using GPU, ensure PyTorch is installed with CUDA support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX
(Replace cuXXX with your CUDA version, e.g., cu118 or cu121)

    OR
```
pip install -r requirements.txt
```

4. Configure NATS Server

Ensure you have NATS Server installed.

Create a NATS configuration file (e.g., nats_mqtt.conf - an example is provided in this repository).
Key sections in nats_mqtt.conf:

# nats_mqtt.conf
```
port: 4222
http_port: 8222 # Optional monitoring

jetstream {
  store_dir: ./nats_jetstream_data # For QoS > 0 MQTT
}

mqtt {
  port: 1883 # NATS MQTT listener port
}

authorization {
  users = [
    {
      user: "your_nats_mqtt_user",
      password: "your_nats_mqtt_password",
      permissions {
        publish: { allow: ["vision_events.>"] } # Adjust base topic if needed
        subscribe: { allow: ["$SYS.>"] }
      }
    }
  ]
}
```
Important:

Update user, password, and permissions in the authorization block.

The publish permission (e.g., vision_events.>) must match the MQTT_BASE_TOPIC you will set in the .env file.

Start NATS Server:
```
nats-server -c nats_mqtt.conf
```

Verify from NATS logs that the MQTT listener on port 1883 is active.

5. Configure the Application (.env file)

Create a .env file in the project root with the following content. Modify the placeholder values to match your setup.

.env
```
# --- YOLO Model and Video Paths ---
MODEL_PATH="yolov8n.pt" # Path to your YOLO model file
VIDEO_PATH="rtsp://admin:admin%404321@12.23.34.56/cam/realmonitor?channel=1&subtype=0" # Your RTSP URL or local video path

# --- Detection and ROI Configuration ---
CONFIDENCE_THRESHOLD="0.35"
TARGET_CLASS_NAME="baggage" # e.g., "person", "car", "baggage" (must exist in model)
INFERENCE_IMG_SIZE="640"    # Inference image size (e.g., 320, 640, 1280)
FRAME_PROCESSING_INTERVAL="1" # Process every Nth frame (1 = every frame)

# --- Region of Interest (ROI) Configurations ---
# Define up to 2 ROIs. Leave ROI_X_NAME blank to disable that ROI.
# ROI_1
ROI_1_NAME="EntryGate"
ROI_1_X1="100"
ROI_1_Y1="150"
ROI_1_X2="400"
ROI_1_Y2="550"

# ROI_2
ROI_2_NAME="ExitScan"
ROI_2_X1="700"
ROI_2_Y1="150"
ROI_2_X2="1000"
ROI_2_Y2="550"

# --- MQTT Broker Configuration (for NATS) ---
MQTT_BROKER_ADDRESS="localhost"     # NATS server IP or hostname
MQTT_BROKER_PORT="1883"             # NATS MQTT listener port (from nats_mqtt.conf)
MQTT_USERNAME="your_nats_mqtt_user" # NATS user (from nats_mqtt.conf)
MQTT_PASSWORD="your_nats_mqtt_password" # NATS password (from nats_mqtt.conf)
MQTT_CLIENT_ID_PREFIX="edge_cam_01" # Prefix for unique client ID (e.g., camera identifier)
MQTT_BASE_TOPIC="vision_events"     # Base topic for message organization

# --- Device/Context Specific Information ---
VEHICLE_NO="OD02BN1234" # Vehicle identifier

# --- Output Configuration ---
SAVE_OUTPUT_VIDEO="False" # "True" or "False"
SHOW_DISPLAY="True"       # "True" or "False"
OUTPUT_VIDEO_FILENAME_PREFIX="processed_stream"
OUTPUT_VIDEO_FPS="10" # Default FPS for saved video if stream FPS is unreliable (e.g., for RTSP)
```

Notes on .env configuration:

MODEL_PATH: Path to your downloaded or custom-trained YOLO model (e.g., yolov8s.pt).

VIDEO_PATH: Your RTSP stream URL or the path to a local video file.

ROI_X_NAME, ROI_X_X1, etc.: Define the name and coordinates for each ROI. The script will load ROIs where ROI_X_NAME is present and coordinates are valid.

MQTT_USERNAME and MQTT_PASSWORD must match the credentials configured in your nats_mqtt.conf.

MQTT_BASE_TOPIC and MQTT_CLIENT_ID_PREFIX are used to construct the final MQTT topic.

6. Run the Application

Ensure your NATS server is running. Then, execute the main script:

python process_video.py


The script will:

Connect to the RTSP stream (or open the video file).

Load the YOLO model.

Process frames, detect objects, and track them.

When a TARGET_CLASS_NAME object enters a defined ROI, it will:

Increment the count for that ROI.

Publish a JSON message to the NATS MQTT broker.

Display the video with detections and ROI information (if SHOW_DISPLAY="True").

7. Subscribing to MQTT Messages

Use an MQTT client (e.g., MQTT Explorer, mosquitto_sub, or a NATS client like nats sub) to subscribe to the event topic and view the published messages. The topic will be structured like:
vision_events/edge_cam_01/roi_event (based on default .env values).

Example using mosquitto_sub (if your NATS server is on localhost with the example credentials):

mosquitto_sub -h localhost -p 1883 -u "your_nats_mqtt_user" -P "your_nats_mqtt_password" -t "vision_events/edge_cam_01/roi_event" -v


Example using nats sub (if nats CLI is installed and NATS auth allows this user to subscribe):

nats sub "vision_events.edge_cam_01.roi_event" --user your_nats_mqtt_user --password your_nats_mqtt_password --server nats://localhost:4222

(Note: MQTT topic a/b/c maps to NATS subject a.b.c)

An example MQTT payload will look like:
```
{
  "timestamp": "2023-10-28T10:30:45.123456+00:00",
  "vehicle_no": "OD02BN1234",
  "roi_name": "EntryGate",
  "baggage_count_in_roi": 1,
  "triggering_object_id": 42
}
```
# AWS Deployment (EC2 Instance Recommendations)

For deploying this application on AWS, consider the following EC2 instance types:

GPU Instances (Recommended for Real-time Performance)

g4dn series (NVIDIA T4 GPUs): Good balance of performance and cost.

g4dn.xlarge: Often a good starting point for a single stream with a medium YOLO model.

g5 series (NVIDIA A10G GPUs): Higher performance if g4dn is insufficient.

g5.xlarge: Powerful option.

AMI Choice: Use an AWS Deep Learning AMI (DLAMI) for Ubuntu or Amazon Linux 2. These AMIs come with NVIDIA drivers, CUDA, and cuDNN pre-installed, significantly simplifying setup.

CPU-only Instances (If GPU is not an option; expect performance limitations)

c6i / c7i series (Compute Optimized): Good CPU power.

c6i.xlarge: A starting point, but real-time will be challenging.

m6i / m7i series (General Purpose): Balanced resources.

Considerations for CPU-only:

Use very small YOLO models (e.g., yolov8n.pt).

Set INFERENCE_IMG_SIZE to a small value (e.g., 320).

Increase FRAME_PROCESSING_INTERVAL significantly (e.g., 5, 10+).

General AWS Setup Steps:

Launch EC2 Instance: Choose an appropriate instance type and AMI (DLAMI for GPU).

Configure Security Group:

Inbound: Allow SSH (port 22 from your IP), NATS MQTT (port 1883, restrict source if possible), NATS Core (port 4222 if needed), NATS Monitoring (port 8222 from your IP).

Outbound: Allow all or be specific (e.g., for RTSP stream access if external).

Connect via SSH.

Install NATS Server (if running on the same instance) and start it with your nats_mqtt.conf.

Set up Python Environment and install dependencies (as listed in Step 2 & 3 of local setup).

Transfer Application Files (.env, mqtt_handler.py, process_video.py, model) to the instance.

Update .env with correct paths and any AWS-specific configurations (e.g., if NATS runs on a different instance, update MQTT_BROKER_ADDRESS).

Run the application (consider using nohup, screen, tmux, or a process manager like systemd or supervisor for long-running operation).

Monitor CPU/GPU utilization, memory, and network traffic.

# Troubleshooting

Slow Performance / Low FPS:

If using GPU, ensure torch.cuda.is_available() is True in logs and nvidia-smi shows GPU utilization.

Try a smaller INFERENCE_IMG_SIZE in .env.

Increase FRAME_PROCESSING_INTERVAL to process fewer frames.

Use a smaller YOLO model (e.g., yolov8s.pt or yolov8n.pt).

Consider a more powerful EC2 instance if hardware is the bottleneck.

Cannot Connect to RTSP Stream:

Verify the RTSP URL is correct and accessible from the machine running the script (e.g., test with VLC).

Check network connectivity and firewalls.

The OPENCV_FFMPEG_CAPTURE_OPTIONS environment variable in process_video.py can sometimes help with problematic streams.

# MQTT Connection Issues:

Verify NATS server is running and the MQTT listener is active (check NATS logs).

Ensure MQTT_BROKER_ADDRESS, MQTT_BROKER_PORT, MQTT_USERNAME, MQTT_PASSWORD in .env are correct and match your NATS configuration.

Check NATS authorization logs for any permission denials for the MQTT user.

Ensure the EC2 instance's Security Group allows outbound connections to the NATS broker on the MQTT port if NATS is external, or inbound if NATS is on the same EC2.

UnboundLocalError or NameError: Ensure all globally accessed variables in process_video.py (especially those used in finally blocks like cap, writer, mqtt_handler, SHOW_DISPLAY) are initialized to None at the global scope before the main() function.

# Future Enhancements

More sophisticated RTSP stream error handling and recovery.

Dynamic ROI configuration (e.g., via API or configuration file updates without script restart).

Storing processed data (counts, event details) in a database.

Web interface for viewing counts and stream.

Containerization with Docker for easier deployment.


```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ python main_controller.py
```