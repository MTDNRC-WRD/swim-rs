"""kcb_daily.py
Defines kcb_daily function
Called by crop_cycle.py

"""

import datetime
import logging

import numpy as np

from numba import jit


def kcb_daily(config, et_cell, foo, foo_day):
    """Compute basal ET

    Parameters
    ---------
        config :

        et_cell :
        crop :
        foo :
        foo_day :

        debug_flag : boolean
            True : write debug level comments to debug.txt
            False

    Returns
    -------
    None

    Notes
    -----

    """

    # Set MAD to MADmid universally atstart.
    # Value will be changed later.  R.Allen 12/14/2011
    # dgetkchum remove this
    # foo.mad = foo.mad_mid

    gs_start, gs_end = datetime.datetime(foo_day.year, 4, 1), datetime.datetime(foo_day.year, 10, 31)
    gs_start_doy, gs_end_doy = int(gs_start.strftime('%j')), int(gs_end.strftime('%j'))

    # if gs_start_doy < foo_day.doy < gs_end_doy:
    foo.kc_bas = foo_day.ndvi * foo.ndvi_beta + foo.ndvi_alpha

    # if et_cell.input[foo_day.dt_string][capture_flag] > 0.:
    #     foo.capture = 1
    # else:
    #     foo.capture = 0

    # Save kcb value for use tomorrow in case curve needs to be extended until frost
    # Save kc_bas_prev prior to CO2 adjustment to avoid double correction
    foo.kc_bas_prev = foo.kc_bas

    # Water has only 'kcb'

    foo.kc_act = foo.kc_bas
    foo.kc_pot = foo.kc_bas

    # ETr changed to ETref 12/26/2007
    foo.etc_act = foo.kc_act * foo_day.refet
    foo.etc_pot = foo.kc_pot * foo_day.refet
    foo.etc_bas = foo.kc_bas * foo_day.refet

    # Save kcb value for use tomorrow in case curve needs to be extended until frost
    # Save kc_bas_prev prior to CO2 adjustment to avoid double correction

    # dgketchum mod to 'turn on' root growth
    condition1 = (foo_day.doy > gs_start_doy) & (0.10 <= foo.kc_bas)
    condition2 = (foo_day.doy < gs_start_doy) | (foo_day.doy > gs_end_doy)

    foo.grow_root = np.where(condition1, True, np.where(condition2, False, foo.grow_root))

    foo.kc_bas_prev = foo.kc_bas

    foo.height = np.maximum(foo.height, 0.05)


# This one is also bad, don't use it. Throwing an error saying config doesn't have ndvi_alpha. :(
@jit
def kcb_daily_1(year, ndvi, refet, doy, ndvi_alpha, ndvi_beta, grow_root, height):
    """Compute basal ET

    The first four inputs are variables from foo_day, next 2 are from config, the last 2 are from tracker.

    Parameters
    ---------
        config :

        et_cell :
        crop :
        foo :
        foo_day :

        debug_flag : boolean
            True : write debug level comments to debug.txt
            False

    Returns
    -------
    list

    Notes
    -----

    """

    # Set MAD to MADmid universally atstart.
    # Value will be changed later.  R.Allen 12/14/2011
    # dgetkchum remove this
    # foo.mad = foo.mad_mid

    gs_start, gs_end = datetime.datetime(year, 4, 1), datetime.datetime(year, 10, 31)
    gs_start_doy, gs_end_doy = int(gs_start.strftime('%j')), int(gs_end.strftime('%j'))

    # if gs_start_doy < foo_day.doy < gs_end_doy:
    kc_bas = ndvi * ndvi_beta + ndvi_alpha

    # if et_cell.input[foo_day.dt_string][capture_flag] > 0.:
    #     foo.capture = 1
    # else:
    #     foo.capture = 0

    # Save kcb value for use tomorrow in case curve needs to be extended until frost
    # Save kc_bas_prev prior to CO2 adjustment to avoid double correction
    kc_bas_prev = kc_bas

    # Water has only 'kcb'

    kc_act = kc_bas
    kc_pot = kc_bas

    # ETr changed to ETref 12/26/2007
    etc_act = kc_act * refet
    etc_pot = kc_pot * refet
    etc_bas = kc_bas * refet

    # dgketchum mod to 'turn on' root growth
    condition1 = (doy > gs_start_doy) & (0.10 <= kc_bas)
    condition2 = (doy < gs_start_doy) | (doy > gs_end_doy)

    grow_root = np.where(condition1, True, np.where(condition2, False, grow_root))

    height = np.maximum(height, 0.05)

    # what variables are final and need to be output?
    return [kc_bas_prev, etc_act, etc_pot, etc_bas, grow_root, height]
