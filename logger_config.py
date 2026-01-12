import logging
import os
import sys

def setup_logger():
    """
    Configures a professional, clean logger for the application.
    Suppresses noisy libraries and handles known benign errors.
    """
    # 1. Environment & Library Suppression
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL only for TensorFlow
    
    # Check for DEBUG mode
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"
    log_level = logging.DEBUG if debug_mode else logging.INFO

    # 2. Configure Main Logger
    logger = logging.getLogger("storyteller")
    logger.setLevel(log_level)
    
    # Clean Format
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        logger.addHandler(handler)

    # 3. Suppress Noisy Libraries
    # These libraries are very chatty; silence them unless critical
    noisy_libs = [
        "mediapipe", "tensorflow", "absl", "h5py", 
        "numexpr", "urllib3", "httpx", "httpcore"
    ]
    for lib in noisy_libs:
        logging.getLogger(lib).setLevel(logging.ERROR)

    # 4. Handle Asyncio Noise (WinError 10054)
    # This error often spams on Windows/Gradio shutdown or reload
    asyncio_logger = logging.getLogger("asyncio")
    asyncio_logger.setLevel(logging.CRITICAL) 

    return logger

def get_logger():
    return logging.getLogger("storyteller")
