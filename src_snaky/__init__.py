import logging
from . import run as run
from . import snaky_variables as myv
from . import snaky_functions as myf
from . import snaky_classes as myc

from .utils import logger

# Root logger — silence everything by default
logging.getLogger().setLevel(logging.WARNING)
logger.setup(logging.DEBUG)
