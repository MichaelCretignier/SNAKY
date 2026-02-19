import time
import unittest

from numpy import array

from src_snaky.snaky_functions import ccf


class TestCCF(unittest.TestCase):
    wave = []
    spec1 = array([[0.8433243 , 0.85322002, 0.8709996 , ..., 0.99549936, 0.99716879,
            0.99714447]], shape=(1, 174761))
    spec2 = []
    extended = 1500
    rv_range = 300
    oversampling = 1
    spec1_std = None

    def test_ccf_speed(self):
        startTime = time.perf_counter()

        ccf(
            self.wave,
            self.spec1,
            self.spec2,
            self.extended,
            self.rv_range,
            self.oversampling,
            self.spec1_std,
        )
        endTime = time.perf_counter()
        average_time_ms = endTime - startTime
        self.assertLess(average_time_ms, 180000)
