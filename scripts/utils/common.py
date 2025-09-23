# DiffGesture/scripts/utils/common.py
import os
import random
import torch
import datetime
import numpy as np
import logging
from logging.handlers import RotatingFileHandler


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# --- Color setup ---
class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[37m",  # white/gray
        logging.INFO: "\033[34m",  # blue
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[41m"  # red background
    }
    RESET = "\033[0m"

    def format(self, record):
        log_color = self.COLORS.get(record.levelno, self.RESET)
        message = super().format(record)
        return f"{log_color}{message}{self.RESET}"


def set_logger(log_path=None, log_filename='log'):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_formatter = ColorFormatter("%(asctime)s [%(levelname)s]: %(message)s")
    console_handler.setFormatter(console_formatter)
    handlers = [console_handler]

    # File handler without colors (plain text)
    if log_path is not None:
        os.makedirs(log_path, exist_ok=True)
        file_handler = RotatingFileHandler(
            os.path.join(log_path, log_filename),
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        )
        file_formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    logging.basicConfig(level=logging.DEBUG, handlers=handlers)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def setup_logger(log_dir='logs'):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    script_name = os.path.basename(__file__).replace(".py", "")
    log_filename = f"{script_name}_{timestamp}.log"
    set_logger(log_dir, log_filename)
