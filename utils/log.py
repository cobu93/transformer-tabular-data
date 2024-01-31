import logging
import sys
from config import LOGGING_LEVEL

def get_logger():
    logging.basicConfig(level=LOGGING_LEVEL, stream=sys.stdout)
    return logging.getLogger()