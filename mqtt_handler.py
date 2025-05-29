# mqtt_handler.py
import paho.mqtt.client as mqtt
import time
import os
import uuid
from dotenv import load_dotenv

load_dotenv()


class MQTTHandler:
    def __init__(self, broker_address, broker_port,
                 username=None, password=None,
                 client_id_prefix="edge_device", keepalive=60):
        self.broker_address = broker_address
        self.broker_port = int(broker_port)
        self.username = username
        self.password = password
        self.keepalive = int(keepalive)

        device_specific_id = str(uuid.uuid4()).split('-')[0]
        self.client_id = f"{client_id_prefix}_{device_specific_id}"
        self.is_connected_flag = False

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self.client_id)

        if self.username:
            self.client.username_pw_set(username=self.username, password=self.password)
            # print(f"[MQTT HANDLER INFO] MQTT authentication will be used for user: {self.username}")
        elif self.password:
             print("[MQTT HANDLER WARNING] MQTT password provided, but no username. This is unusual.")

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"[MQTT HANDLER INFO] Connected to MQTT Broker: {self.broker_address} as client ID: {self.client_id}")
            self.is_connected_flag = True
        else:
            error_msg = f"[MQTT HANDLER ERROR] Failed to connect, return code {rc} ("
            if rc == 1: error_msg += "incorrect protocol version"
            elif rc == 2: error_msg += "invalid client identifier"
            elif rc == 3: error_msg += "server unavailable"
            elif rc == 4: error_msg += "bad username or password" # Common for NATS auth issues
            elif rc == 5: error_msg += "not authorised" # Common for NATS auth issues
            else: error_msg += "Unknown reason"
            error_msg += ")"
            print(error_msg)
            self.is_connected_flag = False

    def _on_disconnect(self, client, userdata, rc, properties=None):
        print(f"[MQTT HANDLER INFO] Disconnected from MQTT Broker (client ID: {self.client_id}) RC: {rc}")
        self.is_connected_flag = False

    def _on_publish(self, client, userdata, mid, properties=None, rc=None):
        # print(f"[MQTT HANDLER DEBUG] Message Published MID: {mid}, RC: {rc}")
        pass

    def connect(self):
        if self.is_connected_flag:
            # print("[MQTT HANDLER INFO] Already connected.")
            return
        try:
            print(f"[MQTT HANDLER INFO] Attempting to connect to MQTT broker: {self.broker_address}:{self.broker_port} with client ID: {self.client_id}")
            self.client.connect(self.broker_address, self.broker_port, self.keepalive)
            self.client.loop_start()
            # Give a brief moment for the on_connect callback to fire
            # For production, a more robust connection check might be needed (e.g., waiting on an event)
            time.sleep(1) # Simple delay
        except ConnectionRefusedError as e:
            print(f"[MQTT HANDLER ERROR] MQTT Connection Refused. Broker at {self.broker_address}:{self.broker_port} might not be running, or check credentials/NATS config. Details: {e}")
            self.is_connected_flag = False
        except Exception as e:
            print(f"[MQTT HANDLER ERROR] Could not connect to MQTT broker: {e}")
            self.is_connected_flag = False

    def is_connected(self):
        return self.is_connected_flag

    def publish(self, topic, payload, qos=0, retain=False):
        if not self.is_connected():
            print("[MQTT HANDLER WARNING] Not connected to MQTT broker. Cannot publish.")
            return None

        if not isinstance(payload, str):
            payload = str(payload)

        # print(f"[MQTT HANDLER DEBUG] Publishing to Topic: {topic}, Payload: {payload}")
        result = self.client.publish(topic, payload, qos=qos, retain=retain)
        return result

    def disconnect(self):
        loop_was_running = hasattr(self.client, '_thread') and self.client._thread is not None and self.client._thread.is_alive()
        paho_is_connected = self.client.is_connected() if hasattr(self.client, 'is_connected') else False

        if loop_was_running or paho_is_connected or self.is_connected_flag:
            print(f"[MQTT HANDLER INFO] Disconnecting MQTT client ID: {self.client_id}...")
            if loop_was_running:
                self.client.loop_stop(force=False)

            if paho_is_connected or self.is_connected_flag:
                self.client.disconnect()
                # Wait for on_disconnect to update our flag
                timeout = time.time() + 2 # Shorter timeout for disconnect
                while self.is_connected_flag and time.time() < timeout:
                    time.sleep(0.05)
            self.is_connected_flag = False
        # else:
            # print("[MQTT HANDLER INFO] MQTT client already disconnected or loop not running.")

if __name__ == '__main__':
    # Example for testing mqtt_handler.py directly
    print("Testing MQTTHandler (ensure NATS server is running with config)...")
    TEST_BROKER = os.getenv("MQTT_BROKER_ADDRESS", "localhost")
    TEST_PORT = int(os.getenv("MQTT_BROKER_PORT", 1883))
    TEST_USER = os.getenv("MQTT_USERNAME", "your_nats_mqtt_user") # Use same user as in nats_mqtt.conf
    TEST_PASS = os.getenv("MQTT_PASSWORD", "your_nats_mqtt_password") # Use same pass as in nats_mqtt.conf
    TEST_TOPIC = f"test/mqtthandler/{str(uuid.uuid4())[:8]}" # Unique test topic

    handler = MQTTHandler(TEST_BROKER, TEST_PORT,
                          username=TEST_USER, password=TEST_PASS,
                          client_id_prefix="direct_test_client")
    handler.connect()
    time.sleep(2) # Allow time for connection

    if handler.is_connected():
        print(f"Publishing test message to {TEST_TOPIC}...")
        pub_info = handler.publish(TEST_TOPIC, "Hello from MQTTHandler test!")
        if pub_info:
            # pub_info.wait_for_publish(timeout=2) # Useful for QoS > 0
            print(f"  Published, MID: {pub_info.mid}, RC: {pub_info.rc}")
        time.sleep(0.5)
        #handler.disconnect()
    else:
        print(f"Failed to connect to {TEST_BROKER}:{TEST_PORT} for testing. Check NATS server, config, and credentials.")
    print("MQTTHandler test finished.")