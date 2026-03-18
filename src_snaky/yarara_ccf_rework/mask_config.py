from dataclasses import InitVar, dataclass, field
from typing import Union
import numpy as np
import pandas as pd
import logging

from src_snaky import myv

logger = logging.getLogger('snaky')

@dataclass
class MaskConfig:
    mask_input: InitVar[Union[str, pd.DataFrame, np.ndarray]]

    mask_col: str = 'weight_rv'
    weighted: bool = True
    squared: bool = True
    wave_min: float = 4000.0
    wave_max: float = 10000.0
    delta_window: int = 5

    # Derived from post_init
    name: str = field(init=False)
    mask: np.ndarray = field(init=False)


    def __post_init__(self, mask_input: Union[str, pd.DataFrame, np.ndarray]):
        if type(mask_input) is str:
            # ccf_name = self.mask
            self.name = mask_input
            mask_loc = f'{myv.MATERIAL_DIR}/MASK_CCF/{mask_input}.txt'
            mask = np.genfromtxt(mask_loc)
            self.mask = np.array([0.5*(mask[:,0]+mask[:,1]),mask[:,2]]).T
            logger.info(f'CCF mask selected : {mask_loc}s')
        elif isinstance(mask_input, pd.DataFrame):
            self.name = "unknown"
            self.mask = np.array([np.array(mask_input['freq_mask0']).astype('float'),np.array(mask_input[self.mask_col]).astype('float')]).T
            self.name = 'ManualDF'
