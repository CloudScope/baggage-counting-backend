# logger_setup.py
import logging
import os
from pythonjsonlogger import jsonlogger
import logstash_async.handler # Keep for future use, but can be conditionally imported
from logging.handlers import RotatingFileHandler # For rotating log files

def setup_logger(logger_name='app_logger', log_level_str=None):
    """
    Sets up a logger with console, optional file, and optional Logstash handlers using JSON format.
    """
    log_level_env = os.getenv("LOG_LEVEL", "INFO").upper()
    effective_log_level_str = log_level_str if log_level_str else log_level_env
    log_level = getattr(logging, effective_log_level_str, logging.INFO)

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()

    #formatter = jsonlogger.JsonFormatter(
    #    '%(asctime)s %(levelname)s %(name)s %(module)s %(funcName)s %(lineno)d %(message)s',
    #    timestamp=True
    #)

    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(levelname)s %(name)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
        #timestamp=True
    )

    # Console Handler (optional)
    enable_console_logging = os.getenv("ENABLE_CONSOLE_LOGGING", "True").lower() == 'true'
    if enable_console_logging:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter) # JSON to console
        logger.addHandler(console_handler)
        # print(f"DEBUG: Console logging enabled for {logger_name}") # Simple print for bootstrap debugging

    # --- NEW: File Handler (optional) ---
    enable_file_logging = os.getenv("ENABLE_FILE_LOGGING", "False").lower() == 'true'
    log_file_path = os.getenv("LOG_FILE_PATH", "app.log") # Default file name if not set

    if enable_file_logging and log_file_path:
        try:
            # Ensure directory for log file exists
            log_dir = os.path.dirname(log_file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                # print(f"DEBUG: Created log directory {log_dir}")

            # Use RotatingFileHandler for better log management
            # Rotates when file reaches maxBytes, keeps backupCount old files
            file_handler = RotatingFileHandler(
                log_file_path,
                maxBytes=10*1024*1024,  # 10 MB per file
                backupCount=5,          # Keep 5 backup files (e.g., app.log, app.log.1, ...)
                encoding='utf-8'        # Explicitly set encoding
            )
            file_handler.setFormatter(formatter) # Apply JSON formatter
            logger.addHandler(file_handler)
            # print(f"DEBUG: File logging enabled for {logger_name} to {log_file_path}")
            logger.info(f"File logging enabled: writing to {log_file_path}") # Log this through the configured handlers
        except Exception as e:
            # If file logging setup fails, log to console (if enabled) or basicConfig
            error_msg = f"Failed to initialize FileHandler for {log_file_path}: {e}. File logging disabled."
            if enable_console_logging:
                logger.error(error_msg, exc_info=True) # Use logger if console handler is already there
            else:
                logging.basicConfig(level=logging.ERROR) # Fallback if no handlers yet
                logging.error(error_msg, exc_info=True)
            # print(f"ERROR: {error_msg}") # Simple print for bootstrap debugging
    elif enable_file_logging:
        warn_msg = "File logging is set to be enabled, but LOG_FILE_PATH is missing or invalid. File logging disabled."
        if enable_console_logging:
            logger.warning(warn_msg)
        else:
            logging.basicConfig(level=logging.WARNING)
            logging.warning(warn_msg)
        # print(f"WARNING: {warn_msg}")

    # Logstash Handler (optional, unchanged, but now explicitly depends on .env)
    enable_logstash_logging = os.getenv("ENABLE_LOGSTASH_LOGGING", "False").lower() == 'true'
    logstash_host = os.getenv("LOGSTASH_HOST")
    logstash_port_str = os.getenv("LOGSTASH_PORT")
    logstash_db_path = os.getenv("LOGSTASH_DATABASE_PATH", None)

    if enable_logstash_logging and logstash_host and logstash_port_str:
        try:
            logstash_port = int(logstash_port_str)
            logstash_tcp_handler = logstash_async.handler.AsynchronousLogstashHandler(
                host=logstash_host, port=logstash_port, database_path=logstash_db_path,
                enable_local_storage=bool(logstash_db_path), ssl_enable=False
            )
            logstash_tcp_handler.setFormatter(formatter)
            logger.addHandler(logstash_tcp_handler)
            logger.info(f"Logstash logging enabled: sending to {logstash_host}:{logstash_port}. Disk buffering: {'Enabled at ' + logstash_db_path if logstash_db_path else 'Disabled'}")
        except ValueError:
            msg = f"Invalid LOGSTASH_PORT: {logstash_port_str}. Logstash logging disabled."
            if enable_console_logging or enable_file_logging: logger.error(msg)
            else: logging.basicConfig(level=logging.ERROR); logging.error(msg)
        except Exception as e:
            msg = f"Failed to initialize Logstash handler: {e}. Logstash logging disabled."
            if enable_console_logging or enable_file_logging: logger.error(msg, exc_info=True)
            else: logging.basicConfig(level=logging.ERROR); logging.error(msg, exc_info=True)
    elif enable_logstash_logging:
        msg = "Logstash logging is set to be enabled, but LOGSTASH_HOST or LOGSTASH_PORT is missing/invalid. Logstash logging disabled."
        if enable_console_logging or enable_file_logging: logger.warning(msg)
        else: logging.basicConfig(level=logging.WARNING); logging.warning(msg)

    # Fallback if no handlers were configured at all
    if not logger.hasHandlers():
        fallback_handler = logging.StreamHandler()
        fallback_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fallback_handler.setFormatter(fallback_formatter)
        logger.addHandler(fallback_handler)
        logger.warning("No primary log handlers (Console/File/Logstash) were configured. Using basic console output.")
        # print("WARNING: No log handlers configured. Using basic stdout.")

    return logger