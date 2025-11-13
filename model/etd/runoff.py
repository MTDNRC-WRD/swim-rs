"""runoff.py
called by compute_crop_et.py

"""
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def runoff_infiltration_excess(foo, foo_day):
    # This doesn't split snow vs rain?
    foo.sro = np.maximum(foo_day.hr_precip - foo.ksat_hourly,
                         np.zeros_like(foo_day.hr_precip)).sum(axis=0).reshape(1, -1)


def runoff_infiltration_excess_daily(foo):
    foo.sro = np.maximum((foo.melt + foo.rain) - foo.ksat, 0)  # still need reshape?
