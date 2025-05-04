import logging


def setup_logger(log_file_path):
    format = "%(asctime)s - %(levelname)s - %(message)s"
    file_handler = logging.FileHandler(log_file_path, "w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(format))
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(format))
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
