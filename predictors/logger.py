import logging
import sys

def get_logger():
    logger = logging.getLogger('predictors')
    logger.setLevel(logging.DEBUG)  

    # Create handlers for different logging levels
    info_handler = logging.StreamHandler(sys.stdout)
    error_handler = logging.StreamHandler(sys.stderr)

    # Set levels for handlers
    info_handler.setLevel(logging.INFO)
    error_handler.setLevel(logging.ERROR)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set formatter for handlers
    info_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(info_handler)
    logger.addHandler(error_handler)

    return logger