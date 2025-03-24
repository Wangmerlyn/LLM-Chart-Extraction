import logging

def setup_logger(name="my_logger", log_file=None, level=logging.INFO):
    """
    Set up a logger that logs messages at the INFO level and higher.

    Parameters:
        name (str): Name of the logger (default is 'my_logger').
        log_file (str): If provided, log messages will be written to this file.
        level (int): The logging level (default is logging.INFO).

    Returns:
        logging.Logger: The configured logger.
    """
    # Create a logger
    logger = logging.getLogger(name)

    # Set the logging level
    logger.setLevel(level)

    # Create a formatter for the log messages
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a console handler to output logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # If a log file is provided, create a file handler to log to a file
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
