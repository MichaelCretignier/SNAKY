from dataclasses import InitVar, dataclass, field
from typing import Any
import numpy as np
import pandas as pd

from src_snaky import myf

@dataclass
class ObservationContext:
    dir_root: str
    files: list[Any]
    rv_shift_input: InitVar[np.ndarray[Any] | None] = None
    spectra: tuple | None = None
    sub_dico: str = 'matching_diff'

    # Derived from post_init
    ins: str = field(init=False)
    jdb: np.ndarray[Any] = field(init=False)
    rv_shift: np.ndarray[Any] = field(init=False)

    def __post_init__(self, rv_shift_input: np.ndarray | None):
        self.ins = self.dir_root.split('/')[-2]
        self.jdb = get_jdb(self.files[-1], self.dir_root)
        self.rv_shift = rv_shift_input or np.zeros(len(self.files[-1]))


def get_jdb(files,dir_root):
    try:
        summary = import_summary(dir_root)
        mask = myf.in1d(np.array(summary['filename']),files)
        jdb = np.array(summary.loc[mask,'jdb'])
        if np.sum(jdb!=jdb)!=0:
            jdb = np.arange(len(files))
    except:
        jdb = np.arange(len(files))
    return jdb


def import_summary(dir_root):
    material = pd.read_csv(dir_root+'WORKSPACE/Analyse_summary.csv',index_col=0)
    return material
