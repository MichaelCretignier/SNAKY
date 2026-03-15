from dataclasses import dataclass

@dataclass
class CCFConfig:
    rv_range: float
    rv_borders: float
    bis_range: float
    ccf_oversampling: int = 1
    analytical_model: str = 'auto'
    continuum_method: str = 'flux'
    normalisation: str = 'left'
    del_outside_max: bool = False
    check_non_transform: bool = True
