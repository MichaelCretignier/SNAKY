from dataclasses import dataclass
import logging

logger = logging.getLogger('snaky')

@dataclass
class StellarParams:
    rv_sys: float          # km/s
    fwhm: float            # km/s
    beta_gnd: float = 2.0

    def __post_init__(self):
        logger.info(f'RV sys : {self.rv_sys:.2f} [km/s]')
        self.rv_sys *= 1000
