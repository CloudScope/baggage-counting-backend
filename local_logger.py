# edge_device_app/local_logger.py
import csv
import os
from datetime import datetime

def ensure_log_dir_exists(vehicle_no_for_subfolder, base_log_dir_from_config):
    """Ensures directory structure /base_log_dir/YYYY-MM-DD/ exists."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    log_dir_for_today = os.path.join(base_log_dir_from_config, today_str)

    try:
        if not os.path.exists(base_log_dir_from_config):
            os.makedirs(base_log_dir_from_config)
            print(f"LocalLogger: Created base log directory: {base_log_dir_from_config}")
        if not os.path.exists(log_dir_for_today):
            os.makedirs(log_dir_for_today)
            print(f"LocalLogger: Created daily log directory: {log_dir_for_today}")
        return log_dir_for_today
    except OSError as e:
        print(f"LocalLogger ERROR: Could not create log directory {log_dir_for_today}: {e}")
        return None


def log_event_to_csv(vehicle_no, bag_count, timestamp, csv_full_filepath):
    """Logs event to the specified CSV file path."""
    if csv_full_filepath is None:
        print("LocalLogger ERROR: CSV file path not provided for logging.")
        return False

    file_exists = os.path.isfile(csv_full_filepath)
    
    try:
        # Ensure directory for the specific file exists (redundant if ensure_log_dir_exists was called, but safe)
        os.makedirs(os.path.dirname(csv_full_filepath), exist_ok=True)

        with open(csv_full_filepath, 'a', newline='') as csvfile:
            fieldnames = ['timestamp', 'vehicle_number', 'bag_count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists or os.path.getsize(csv_full_filepath) == 0: # Check if file is empty
                writer.writeheader()
            
            writer.writerow({
                'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'vehicle_number': vehicle_no,
                'bag_count': bag_count
            })
        # print(f"LocalLogger: Event logged to {csv_full_filepath}")
        return True
    except IOError as e:
        print(f"LocalLogger ERROR: Could not write to CSV {csv_full_filepath}: {e}")
        return False
    except Exception as e:
        print(f"LocalLogger ERROR: Unexpected error writing CSV {csv_full_filepath}: {e}")
        return False