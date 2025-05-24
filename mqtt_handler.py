# edge_device_app/mqtt_handler.py
import paho.mqtt.client as mqtt
import json
import time
import os # For S3 path checks and NATS TLS cert path checks
import socket # For specific connection error types

# Import config which should have loaded .env_edge
# This makes config variables available globally within this module
try:
    from . import config # If mqtt_handler is part of a package imported by main_controller
except ImportError:
    import config # If main_controller and mqtt_handler are in the same dir, run as script


# --- Optional S3 Uploader Class ---
if config.S3_ENABLED:
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
        print("MQTT Handler: boto3 imported for S3.")

        class S3Uploader:
            def __init__(self, bucket_name):
                if not bucket_name:
                    print("S3Uploader Error: Bucket name not provided. S3 uploads will be disabled.")
                    self.s3_client = None
                    self.bucket_name = None
                    return
                try:
                    self.s3_client = boto3.client('s3')
                    self.bucket_name = bucket_name
                    self.s3_client.head_bucket(Bucket=self.bucket_name)
                    print(f"S3Uploader: Successfully connected to S3 and bucket '{self.bucket_name}'.")
                except (NoCredentialsError, PartialCredentialsError):
                    print("S3Uploader Error: AWS credentials not found. S3 uploads disabled.")
                    self.s3_client = None
                except ClientError as e:
                    error_code = e.response.get("Error", {}).get("Code")
                    if error_code == 'NoSuchBucket':
                        print(f"S3Uploader Error: Bucket '{self.bucket_name}' does not exist. S3 uploads disabled.")
                    elif error_code == 'AccessDenied':
                         print(f"S3Uploader Error: Access denied to bucket '{self.bucket_name}'. S3 uploads disabled.")
                    else:
                        print(f"S3Uploader Error: S3 client error for bucket '{self.bucket_name}': {e}. S3 uploads disabled.")
                    self.s3_client = None
                except Exception as e:
                    print(f"S3Uploader Error: Unexpected error initializing S3: {e}. S3 uploads disabled.")
                    self.s3_client = None

            def upload_file(self, file_path, s3_key):
                if not self.s3_client or not self.bucket_name:
                    # print(f"S3Uploader: S3 not configured. Cannot upload {file_path}.")
                    return False
                if not os.path.exists(file_path):
                    print(f"S3Uploader Error: File not found {file_path}")
                    return False
                try:
                    self.s3_client.upload_file(file_path, self.bucket_name, s3_key)
                    print(f"S3Uploader: Successfully uploaded {file_path} to s3://{self.bucket_name}/{s3_key}")
                    return True
                except ClientError as e:
                    print(f"S3Uploader Error: Failed to upload {file_path} to S3: {e}")
                    return False
                except Exception as e:
                    print(f"S3Uploader Error: Unexpected S3 upload error for {file_path}: {e}")
                    return False
    except ImportError:
        print("MQTT Handler WARNING: boto3 library not found, S3Uploader will not be available even if S3_ENABLED=True.")
        S3Uploader = None 
else:
    S3Uploader = None
# --- End Optional S3 Uploader Class ---


class MQTTHandler:
    def __init__(self, host, port, client_id, username, password, topic_template):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.username = username
        self.password = password
        self.topic_template = topic_template
        self.connected = False
        self.client = None

        try:
            self.client = mqtt.Client(client_id=self.client_id) # , protocol=mqtt.MQTTv311 can be specified
            if self.username: # NATS might not require username/password if other auth is used or none
                self.client.username_pw_set(self.username, self.password)
            
            if config.NATS_MQTT_USE_TLS:
                if os.path.exists(config.NATS_MQTT_CA_CERTS_PATH):
                    self.client.tls_set(ca_certs=config.NATS_MQTT_CA_CERTS_PATH)
                    print(f"MQTT Edge: TLS configured using CA: {config.NATS_MQTT_CA_CERTS_PATH}")
                else:
                    print(f"MQTT Edge WARNING: NATS_MQTT_USE_TLS is true, but CA cert not found at {config.NATS_MQTT_CA_CERTS_PATH}.")
                    # For testing with broker that has TLS but no CA validation (e.g. self-signed and client doesn't verify)
                    # self.client.tls_set() 
                    # self.client.tls_insecure_set(True) # DANGEROUS for production

            self.client.on_connect = self.on_connect
            self.client.on_disconnect = self.on_disconnect
        except Exception as e:
            print(f"MQTT Edge FATAL ERROR: Could not initialize MQTT client: {e}")

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"MQTT Edge: Connected to NATS/MQTT Broker ({self.host}:{self.port}) successfully.")
            self.connected = True
        else:
            print(f"MQTT Edge ERROR: Failed to connect, return code {rc}. Check NATS server, network, and credentials.")
            self.connected = False

    def on_disconnect(self, client, userdata, rc):
        print(f"MQTT Edge: Disconnected from NATS/MQTT, result code {rc}.")
        self.connected = False

    def connect(self):
        if not self.client: return False
        if self.connected: return True
        try:
            print(f"MQTT Edge: Attempting to connect to {self.host}:{self.port}...")
            self.client.connect(self.host, self.port, keepalive=60)
            self.client.loop_start()
            # It's better to rely on on_connect callback to set self.connected
            # but for initial connect, give it a small window
            for _ in range(5): # Wait up to ~1 second for on_connect
                if self.connected: break
                time.sleep(0.2)
            return self.connected
        except (ConnectionRefusedError, socket.gaierror, socket.timeout, TimeoutError) as e: # More specific network errors
            print(f"MQTT Edge ERROR: Connection to {self.host}:{self.port} failed: {e}")
        except Exception as e:
            print(f"MQTT Edge ERROR: MQTT connection error: {e}")
        self.connected = False
        return False

    def disconnect(self):
        if self.client:
            if self.connected: print("MQTT Edge: Disconnecting...")
            self.client.loop_stop()
            self.client.disconnect() # on_disconnect will set self.connected = False

    def publish_telemetry(self, bay_id, data_payload, max_retries=config.MQTT_RETRY_DELAY): # Use config for default retries
        if not self.client: return False
        if not isinstance(data_payload, dict):
            print("MQTT Edge Error: Payload must be a dictionary."); return False

        payload_json = json.dumps(data_payload)
        topic = self.topic_template.format(bay_id=bay_id)
        
        for attempt in range(max_retries):
            if not self.connected:
                print(f"MQTT Edge: Not connected. Attempting connect (publish attempt {attempt+1}/{max_retries})...")
                if not self.connect():
                    if attempt < max_retries - 1:
                        print(f"MQTT Edge: Connection failed. Retrying publish in {config.MQTT_RETRY_DELAY}s...")
                        time.sleep(config.MQTT_RETRY_DELAY)
                        continue
                    else:
                        print(f"MQTT Edge ERROR: Max connection attempts reached. Cannot publish to {topic}.")
                        return False
            
            if self.connected:
                result_info = self.client.publish(topic, payload_json, qos=1)
                if result_info.rc == mqtt.MQTT_ERR_SUCCESS:
                    # For paho-mqtt v2+, publish with QoS 1 blocks until PUBACK. rc is the check.
                    # For v1.x, you might need result_info.wait_for_publish() and check is_published()
                    # This simplified check assumes if no error, it's queued/sent.
                    print(f"MQTT Edge: Successfully published to {topic} (MID: {result_info.mid})")
                    return True
                else:
                    print(f"MQTT Edge WARN: Publish returned code {result_info.rc} (attempt {attempt+1}).")
                    # If publish fails, connection might be lost or other issue
                    if result_info.rc == mqtt.MQTT_ERR_NO_CONN: self.connected = False

            if attempt < max_retries - 1:
                print(f"MQTT Edge: Retrying publish to {topic} in {config.MQTT_RETRY_DELAY}s...")
                time.sleep(config.MQTT_RETRY_DELAY)
            else:
                print(f"MQTT Edge ERROR: Failed to publish to {topic} after {max_retries} attempts.")
                return False
        return False