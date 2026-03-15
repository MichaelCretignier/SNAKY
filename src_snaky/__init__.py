from . import run as run
from . import snaky_variables as myv
from . import snaky_functions as myf
from . import snaky_classes as myc

import logging

# FIXME: shouldn't be there when publishing. Otherwise it's going to override the logging system of users.
INFO_BEGIN_LEVEL_NUM = 9
logging.addLevelName(INFO_BEGIN_LEVEL_NUM, "BEGIN")

with_time_format = '%(asctime)s [%(levelname)-8s] %(name)s — %(message)s'
without_time_format = '[%(levelname)s] — %(message)s'

COLORS = {
    logging.DEBUG:    '\033[36m',
    logging.INFO:     '\033[32m',
    25:               '\033[34m',   # blue for SECTION
    logging.WARNING:  '\033[33m',
    logging.ERROR:    '\033[31m',
    logging.CRITICAL: '\033[41m',
}

RESET = '\033[0m'

logging.basicConfig(
    level=logging.WARNING,
    format=without_time_format,
)

class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = COLORS.get(record.levelno, RESET)
        return f'{color}{super().format(record)}{RESET}'

handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter(
    fmt=without_time_format,
    datefmt='%H:%M:%S',
))

logger = logging.getLogger('snaky')

logger.addHandler(logging.NullHandler())
logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.DEBUG)
