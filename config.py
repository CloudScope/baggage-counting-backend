import os
from dotenv import load_dotenv

# Base directory of the edge_device_app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(BASE_DIR, '.env')

if os.path.exists(DOTENV_PATH):
    print(f"EdgeApp Config: Loading environment variables from: {DOTENV_PATH}")
    load_dotenv(dotenv_path=DOTENV_PATH)
else:
    print(f"EdgeApp Config Warning: .env file not found at {DOTENV_PATH}. Using defaults or system env vars.")

# Camera Configuration
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
#ANPR_CAMERA_SOURCE = os.getenv("ANPR_CAMERA_SOURCE", "0")
#BAG_CAMERA_SOURCE = os.getenv("BAG_CAMERA_SOURCE", "1")

# YOLO Model
yolo_model_path_env = os.getenv("YOLO_MODEL_PATH")
if not os.path.isabs(yolo_model_path_env):
    YOLO_MODEL_PATH = os.path.join(BASE_DIR, yolo_model_path_env)
else:
    YOLO_MODEL_PATH = yolo_model_path_env
YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", 0.5))

# NATS/MQTT Broker Configuration
NATS_MQTT_BROKER_HOST = os.getenv("NATS_MQTT_BROKER_HOST")
NATS_MQTT_BROKER_PORT = int(os.getenv("NATS_MQTT_BROKER_PORT"))
NATS_MQTT_USERNAME_EDGE = os.getenv("NATS_MQTT_USERNAME_EDGE")
NATS_MQTT_PASSWORD_EDGE = os.getenv("NATS_MQTT_PASSWORD_EDGE")
MQTT_TOPIC_TELEMETRY_TEMPLATE = "baggage.telemetry.{bay_id}" # Hardcoded template
MQTT_CLIENT_ID_EDGE = os.getenv("MQTT_CLIENT_ID_EDGE")
MQTT_RETRY_DELAY = int(os.getenv("MQTT_RETRY_DELAY", 5))

# NATS TLS Configuration (Optional)
NATS_MQTT_USE_TLS = os.getenv("NATS_MQTT_USE_TLS", "false").lower() == "true"
nats_mqtt_ca_certs_path_env = os.getenv("NATS_MQTT_CA_CERTS_PATH", "certs/ca.pem")
if not os.path.isabs(nats_mqtt_ca_certs_path_env):
    NATS_MQTT_CA_CERTS_PATH = os.path.join(BASE_DIR, nats_mqtt_ca_certs_path_env)
else:
    NATS_MQTT_CA_CERTS_PATH = nats_mqtt_ca_certs_path_env


# Local Logging
log_base_dir_env = os.getenv("LOG_BASE_DIR", "logs")
if not os.path.isabs(log_base_dir_env):
    LOG_BASE_DIR = os.path.join(BASE_DIR, log_base_dir_env)
else:
    LOG_BASE_DIR = log_base_dir_env
BAY_ID = os.getenv("BAY_ID", "DefaultBay")

# ANPR
ANPR_TRIGGER_THRESHOLD = int(os.getenv("ANPR_TRIGGER_THRESHOLD", 3))

# Operational
PROCESS_LOOP_DELAY = float(os.getenv("PROCESS_LOOP_DELAY", 0.1)) # seconds

# S3 Configuration (Optional)
S3_ENABLED = os.getenv("S3_ENABLED", "False").lower() == "true"
S3_BUCKET_NAME_LOGS = os.getenv("S3_BUCKET_NAME_LOGS")

# Print some key loaded configs for verification
print(f"EdgeApp Config: NATS Host       = {NATS_MQTT_BROKER_HOST}")
print(f"EdgeApp Config: YOLO Model Path = {YOLO_MODEL_PATH}")
print(f"EdgeApp Config: Log Base Dir    = {LOG_BASE_DIR}")
if NATS_MQTT_USE_TLS:
    print(f"EdgeApp Config: NATS TLS CA Path= {NATS_MQTT_CA_CERTS_PATH}")
if S3_ENABLED:
    print(f"EdgeApp Config: S3 Bucket Logs = {S3_BUCKET_NAME_LOGS}")
print(f"EdgeApp Config: Camera Source   = {CAMERA_SOURCE}")