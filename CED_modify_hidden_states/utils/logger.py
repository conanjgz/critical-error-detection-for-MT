import logging
import sys

formatter = logging.Formatter(
    "| %(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
)


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as multiple loggers"""
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    return logger
