from dataclasses import dataclass

@dataclass
class OutputConfig:
    save: bool = True
    return_ccf: bool = False
    ccf_tag: int = 0
    debug: bool = False
