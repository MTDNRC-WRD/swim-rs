"""crop_cycle.py
Defines DayData class
Defines crop_cycle_mp, crop_cycle, crop_day_loop_mp, crop_day_loop,
    write_crop_output
Called by mod_crop_et.py

"""
import os.path

import numpy as np
import pandas as pd
import calendar
import xarray

from model.etd import calculate_height
from model.etd import compute_field_et
from model.etd import obs_kcb_daily
from model.etd.initialize_tracker import PlotTracker

OUTPUT_FMT = ['et_act',
              'etref',
              'kc_act',
              'kc_bas',
              'ks',
              'ke',
              'melt',
              'rain',
              'depl_root',
              'depl_ze',
              'dperc',
              'runoff',
              'delta_soil_water',
              'wbal',
              'ppt',
              'snow_fall',
              'taw',
              'taw3',
              'daw3',
              'delta_daw3',
              'swe',
              'tavg',
              'tmax',
              'irrigation',
              'fc',
              't',
              'e',
              'few',
              'zr',
              'p_rz',
              'p_eft',
              'soil_water',
              'niwr',
              'irr_day',
              'season',
              'capture',
              ]

# 5 parameter defaults, used when no model parameters are provided to field_day_loop functions.
DEFAULTS = {'ndvi_beta': 1.35,
            'ndvi_alpha': -0.44,
            'mad': 1.0,
            'swe_alpha': 0.073,
            'swe_beta': 1.38}


class DayData:

    def __init__(self):
        self.etref_array = np.zeros(30)


class WaterBalanceError(Exception):
    pass


def year_len(year):
    return 366 if calendar.isleap(year) else 365


# Edited from field_day_loop_nc_5, messing with parameter input.
def field_day_loop_nc_5(config, plots, params=None, debug_flag=False, save_out=None, targets=None):
    """ Run SWIM. Main model code for use with netcdf input files.

    This function loops through the daily time series for all fields at the same time.

    config: obj, loaded config file information
    plots: obj, contains input data
    debug_flag: bool, optional; if False, return only swe and etf timeseries info as numpy array (default);
    if True, output all timeseries info as xarray dataset
    params: dict, optional; if provided, specifies parameters for model - values may be either single numbers
    (for running one field or same values for all fields), or lists/arrays the length of the number of fields being run.
    If not provided, DEFAULT parameter values will be used for ndvi_alpha, ndvi_beta, mad, swe_alpha, and swe_beta,
    and values from plots will be used for aw, rew, and tew.
    save_out: str, optional; if provided, and debug_flag=True, the filepath where all timeseries information
    will be saved; else, results will only be stored in memory.
    targets: list, optional; if provided, list of field IDs to use as targets. model will only run on those fields.

    Returns
    If debug_flag==False: an array of eta and swe time series.
    If debug_flag==True: an xarray dataset of all of the water balance components.
    """
    #
    if not targets:
        targets = plots.input[config.field_index].values  # default indices

    etf, swe = None, None
    size = len(targets)
    # size = len(selected)
    # tracker.load_soils_nc(plots[selected])  # does this work?
    tracker = PlotTracker(size)
    tracker.load_soils_nc(plots[targets])  # supplies aw, rew, tew parameters, and other values.

    # apply calibration parameter updates here

    # Assumes params is a dictionary of values.
    if params:
        if isinstance(params['ndvi_alpha'], (int, float)):  # same values for all fields.
            for k, v in params.items():
                arr = np.ones((1, size)) * v
                tracker.__setattr__(k, arr)
        else:  # unique values for each field. - not sure this works yet.
            for k, v in params.items():
                arr = np.reshape(v, (1, -1))
                tracker.__setattr__(k, arr)
    else:  # same values for all fields, except aw, rew, tew from load_soils_nc() above.
        for k, v in DEFAULTS.items():
            arr = np.ones((1, size)) * v
            tracker.__setattr__(k, arr)

    # Initialize crop data frame
    time_range = pd.date_range(config.start_dt, config.end_dt, freq='D')
    if debug_flag:
        tracker.setup_dataframe(targets)  # creates empty dict of dict with FIDs as keys
    else:
        # creating properly-sized arrays for model results.
        empty = np.zeros((len(time_range), len(targets))) * np.nan
        etf, swe = empty.copy(), empty.copy()

    tracker.set_kc_max()

    foo_day = DayData()
    foo_day.sdays = 0
    foo_day.doy_prev = 0
    foo_day.irr_status = None

    hr_ppt_keys = ['prcp_hr_{}'.format(str(i).rjust(2, '0')) for i in range(0, 24)]
    # cols = ['ndvi_irr', 'etf_irr_ct', '{}_mm_corrected'.format(config.refet_type),
    #         'ndvi_inv_irr', 'etf_inv_irr_ct', '{}_mm'.format(config.refet_type)]

    # looping through days
    for j, step_dt in enumerate(time_range):
        # I think this select statement is ruining everything.
        vals = plots.input.sel(date=step_dt)  # all data for all fields for that date

        # Track variables for each day
        # For now, cast all values to native Python types
        foo_day.sdays += 1
        dt = pd.to_datetime(step_dt)
        foo_day.dt_string = dt.strftime('%Y-%m-%d')

        foo_day.year = dt.year
        foo_day.month = dt.month
        foo_day.day = dt.day
        foo_day.doy = dt.dayofyear

        # Check irrigation status on first date and first day of each year.  # Should be zeros and ones? Is it?
        if foo_day.doy == 1 or foo_day.irr_status is None:
            foo_day.irr_status = np.array([plots.input['irr'].sel(year=dt.year).values])

        # Using yearly irr_status as condition for which variable type to store for each field on this day
        foo_day.ndvi = np.where(foo_day.irr_status, vals['ndvi_irr'], vals['ndvi_inv_irr'])
        foo_day.capture = np.where(foo_day.irr_status, vals['ndvi_irr_ct'], vals['ndvi_inv_irr_ct'])  # why were these etf, and not ndvi?
        foo_day.refet = np.where(foo_day.irr_status, vals['{}_mm_corrected'.format(config.refet_type)],
                                 vals['{}_mm'.format(config.refet_type)])

        # Why would refet be different whether or not it's irrigated?
        foo_day.irr_day = np.array(vals['irr_days']).reshape(1, -1)
        foo_day.min_temp = np.array(vals['tmin_c']).reshape(1, -1)
        foo_day.max_temp = np.array(vals['tmax_c']).reshape(1, -1)
        foo_day.temp_avg = (foo_day.min_temp + foo_day.max_temp) / 2.
        foo_day.srad = np.array(vals['srad_wm2']).reshape(1, -1)
        foo_day.precip = np.array(vals['prcp_mm'])

        if np.any(foo_day.precip > 0.):
            hr_ppt = np.array([vals[k] for k in hr_ppt_keys]).reshape(24, size)
            foo_day.hr_precip = hr_ppt

        foo_day.precip = foo_day.precip.reshape(1, -1)

        # Calculate height of vegetation.
        # Moved up to this point 12/26/07 for use in adj. Kcb and kc_max
        calculate_height.calculate_height(tracker)

        # Interpolate Kcb and make climate adjustment (for ETo basis)
        obs_kcb_daily.kcb_daily(config, plots, tracker, foo_day)  # It doesn't actually use the first two inputs?

        # Calculate Kcb, Ke, ETc
        compute_field_et.compute_field_et(config, plots, tracker, foo_day,
                                          debug_flag)

        # Retrieve values from foo_day and write to output data frame
        # Eventually let compute_crop_et() write directly to output df

        if debug_flag:
            # TODO: should tracker log the tuned parameters? Where do those live?
            for i, fid in enumerate(targets):
                tracker.crop_df[fid][step_dt] = {}
                sample_idx = 0, i
                tracker.crop_df[fid][step_dt]['etref'] = foo_day.refet[sample_idx]

                eta_act = tracker.etc_act[sample_idx]
                tracker.crop_df[fid][step_dt]['capture'] = foo_day.capture[sample_idx]
                tracker.crop_df[fid][step_dt]['t'] = tracker.t[sample_idx]
                tracker.crop_df[fid][step_dt]['e'] = tracker.e[sample_idx]
                tracker.crop_df[fid][step_dt]['kc_act'] = tracker.kc_act[sample_idx]
                tracker.crop_df[fid][step_dt]['ks'] = tracker.ks[sample_idx]
                tracker.crop_df[fid][step_dt]['ke'] = tracker.ke[sample_idx]

                # water balance components
                tracker.crop_df[fid][step_dt]['et_act'] = eta_act

                ppt = foo_day.precip[sample_idx]
                tracker.crop_df[fid][step_dt]['ppt'] = ppt

                melt = tracker.melt[sample_idx]
                tracker.crop_df[fid][step_dt]['melt'] = melt
                rain = tracker.rain[sample_idx]
                tracker.crop_df[fid][step_dt]['rain'] = rain

                runoff = tracker.sro[sample_idx]
                tracker.crop_df[fid][step_dt]['runoff'] = runoff
                dperc = tracker.dperc[sample_idx]
                tracker.crop_df[fid][step_dt]['dperc'] = dperc

                depl_root = tracker.depl_root[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_root'] = depl_root
                depl_root_prev = tracker.depl_root_prev[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_root_prev'] = depl_root_prev

                daw3 = tracker.daw3[sample_idx]
                tracker.crop_df[fid][step_dt]['daw3'] = daw3
                daw3_prev = tracker.daw3_prev[sample_idx]
                tracker.crop_df[fid][step_dt]['daw3_prev'] = daw3_prev
                delta_daw3 = daw3 - daw3_prev
                tracker.crop_df[fid][step_dt]['delta_daw3'] = delta_daw3

                soil_water = tracker.soil_water[sample_idx]
                tracker.crop_df[fid][step_dt]['soil_water'] = soil_water
                soil_water_prev = tracker.soil_water_prev[sample_idx]
                tracker.crop_df[fid][step_dt]['soil_water_prev'] = soil_water_prev
                delta_soil_water = tracker.delta_soil_water[sample_idx]
                tracker.crop_df[fid][step_dt]['delta_soil_water'] = delta_soil_water

                depl_ze = tracker.depl_ze[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_ze'] = depl_ze
                tracker.crop_df[fid][step_dt]['p_rz'] = tracker.p_rz[sample_idx]
                tracker.crop_df[fid][step_dt]['p_eft'] = tracker.p_eft[sample_idx]
                tracker.crop_df[fid][step_dt]['fc'] = tracker.fc[sample_idx]
                tracker.crop_df[fid][step_dt]['few'] = tracker.few[sample_idx]
                tracker.crop_df[fid][step_dt]['aw'] = tracker.aw[sample_idx]
                tracker.crop_df[fid][step_dt]['aw3'] = tracker.aw3[sample_idx]
                tracker.crop_df[fid][step_dt]['taw'] = tracker.taw[sample_idx]
                tracker.crop_df[fid][step_dt]['taw3'] = tracker.taw3[sample_idx]
                tracker.crop_df[fid][step_dt]['irrigation'] = tracker.irr_sim[sample_idx]
                tracker.crop_df[fid][step_dt]['irr_day'] = foo_day.irr_day[sample_idx]
                tracker.crop_df[fid][step_dt]['swe'] = tracker.swe[sample_idx]
                tracker.crop_df[fid][step_dt]['snow_fall'] = tracker.snow_fall[sample_idx]
                tracker.crop_df[fid][step_dt]['tavg'] = foo_day.temp_avg[sample_idx]
                tracker.crop_df[fid][step_dt]['tmax'] = foo_day.max_temp[sample_idx]
                tracker.crop_df[fid][step_dt]['zr'] = tracker.zr[sample_idx]
                tracker.crop_df[fid][step_dt]['kc_bas'] = tracker.kc_bas[sample_idx]
                tracker.crop_df[fid][step_dt]['niwr'] = tracker.niwr[sample_idx]
                tracker.crop_df[fid][step_dt]['et_bas'] = tracker.etc_bas
                tracker.crop_df[fid][step_dt]['season'] = tracker.in_season

                water_out = eta_act + dperc + runoff
                water_stored = soil_water - soil_water_prev
                water_in = melt + rain
                balance = water_in - water_stored - water_out

                tracker.crop_df[fid][step_dt]['wbal'] = balance

                if abs(balance) > 0.1 and foo_day.year > 2000:
                    pass
                    # raise WaterBalanceError('Check November water balance')

        else:
            if np.isnan(tracker.kc_act.any()):
                raise ValueError('NaN in Kc_act')

            if np.isnan(tracker.swe.any()):
                raise ValueError('NaN in SWE')

            etf[j, :] = tracker.etc_act
            swe[j, :] = tracker.swe

    if debug_flag:
        # pass final dataset to calling script
        tracker.crop_df = {fid: pd.DataFrame().from_dict(tracker.crop_df[fid], orient='index')[OUTPUT_FMT]
                           for fid in targets}  # turn nested dict to single dict of pd df
        for fid in tracker.crop_df:
            tracker.crop_df[fid].index = pd.to_datetime(tracker.crop_df[fid].index)  # turn indices into dt
            tracker.crop_df[fid].index = tracker.crop_df[fid].index.rename('date')  # rename index (should be after concat)
        out_ds_list = [tracker.crop_df[fid].to_xarray() for fid in targets]  # convert to xarray
        out_ds = xarray.concat(out_ds_list, pd.Index(tracker.crop_df.keys(), name=config.field_index))  # create out 1 ds w/ 2 dims

        if save_out:
            out_ds.to_netcdf(save_out, engine='netcdf4')
        return out_ds

    else:
        # if not debug, just return the actual ET and SWE results as ndarray
        return np.asarray([etf, swe])


# Edited from field_day_loop_nc_3, changing output from ETf to ETa
# This is the good one. But slow.
def field_day_loop_nc_4(config, plots, params=None, debug_flag=False, save_out=None) -> np.ndarray | xarray.Dataset:
    """Run SWIM. Main model code for use with netcdf input files.

    This function loops through the daily time series for all fields at the same time.

    Args:
        config: obj, loaded config file information
        plots: obj, contains input data
        debug_flag: bool, optional; if False, return only swe and eta timeseries info as numpy array (default);
          if True, output all timeseries info as xarray dataset
        params: dict, optional; if provided, specifies parameters for model - values may be either single numbers
          (for running one field or same values for all fields), or lists/arrays the length of the number of fields
          being run. If not provided, DEFAULT parameter values will be used for ndvi_alpha, ndvi_beta, mad, swe_alpha,
          and swe_beta, and values from plots will be used for aw, rew, and tew.
        save_out: str, optional; if provided, and debug_flag=True, the filepath where all timeseries information
          will be saved; else, results will only be stored in memory.

    Returns:
        If debug_flag==False, return 2xN array of swe and eta timeseries info; If debug_flag==True, return all
        timeseries as xarray dataset.
    """
    #
    etf, swe = None, None
    size = len(plots.input[config.field_index].values)
    # size = len(selected)
    # tracker.load_soils_nc(plots[selected])  # does this work?
    tracker = PlotTracker(size)
    tracker.load_soils_nc(plots)  # supplies aw, rew, tew parameters, and other values.

    # apply calibration parameter updates here

    # Assumes params is a dictionary of values.
    if params:
        if isinstance(params['ndvi_alpha'], (int, float)):  # same values for all fields.
            for k, v in params.items():
                arr = np.ones((1, size)) * v
                tracker.__setattr__(k, arr)
        else:  # unique values for each field. - not sure this works yet.
            for k, v in params.items():
                arr = np.reshape(v, (1, -1))
                tracker.__setattr__(k, arr)
    else:  # same values for all fields, except aw, rew, tew from load_soils_nc() above.
        for k, v in DEFAULTS.items():
            arr = np.ones((1, size)) * v
            tracker.__setattr__(k, arr)

    targets = plots.input[config.field_index].values

    # Initialize crop data frame
    time_range = pd.date_range(config.start_dt, config.end_dt, freq='D')
    if debug_flag:
        tracker.setup_dataframe(targets)  # creates empty dict of dict with FIDs as keys
    else:
        # creating properly-sized arrays for model results.
        empty = np.zeros((len(time_range), len(targets))) * np.nan
        etf, swe = empty.copy(), empty.copy()

    tracker.set_kc_max()

    foo_day = DayData()
    foo_day.sdays = 0
    foo_day.doy_prev = 0
    foo_day.irr_status = None

    hr_ppt_keys = ['prcp_hr_{}'.format(str(i).rjust(2, '0')) for i in range(0, 24)]
    # cols = ['ndvi_irr', 'etf_irr_ct', '{}_mm_corrected'.format(config.refet_type),
    #         'ndvi_inv_irr', 'etf_inv_irr_ct', '{}_mm'.format(config.refet_type)]

    # looping through days
    for j, step_dt in enumerate(time_range):
        # I think this select statement is ruining everything.
        vals = plots.input.sel(date=step_dt)  # all data for all fields for that date

        # Track variables for each day
        # For now, cast all values to native Python types
        foo_day.sdays += 1
        dt = pd.to_datetime(step_dt)
        foo_day.dt_string = dt.strftime('%Y-%m-%d')

        foo_day.year = dt.year
        foo_day.month = dt.month
        foo_day.day = dt.day
        foo_day.doy = dt.dayofyear

        # Check irrigation status on first date and first day of each year.
        if foo_day.doy == 1 or foo_day.irr_status is None:
            foo_day.irr_status = np.array([plots.input['irr'].sel(year=dt.year).values])

        # Using yearly irr_status as condition for which variable type to store for each field on this day
        foo_day.ndvi = np.where(foo_day.irr_status, vals['ndvi_irr'], vals['ndvi_inv_irr'])
        foo_day.capture = np.where(foo_day.irr_status, vals['ndvi_irr_ct'], vals['ndvi_inv_irr_ct'])  # why were these etf, and not ndvi?
        foo_day.refet = np.where(foo_day.irr_status, vals['{}_mm_corrected'.format(config.refet_type)],
                                 vals['{}_mm'.format(config.refet_type)])

        # Why would refet be different whether or not it's irrigated?
        foo_day.irr_day = np.array(vals['irr_days']).reshape(1, -1).astype(bool)
        foo_day.min_temp = np.array(vals['tmin_c']).reshape(1, -1)
        foo_day.max_temp = np.array(vals['tmax_c']).reshape(1, -1)
        foo_day.temp_avg = (foo_day.min_temp + foo_day.max_temp) / 2.
        foo_day.srad = np.array(vals['srad_wm2']).reshape(1, -1)  # where is this used?
        foo_day.precip = np.array(vals['prcp_mm'])

        if np.any(foo_day.precip > 0.):
            hr_ppt = np.array([vals[k] for k in hr_ppt_keys]).reshape(24, size)
            foo_day.hr_precip = hr_ppt  # where is this even used?

        foo_day.precip = foo_day.precip.reshape(1, -1)

        # Calculate height of vegetation.
        # Moved up to this point 12/26/07 for use in adj. Kcb and kc_max
        calculate_height.calculate_height(tracker)

        # Interpolate Kcb and make climate adjustment (for ETo basis)
        obs_kcb_daily.kcb_daily(config, plots, tracker, foo_day)  # It doesn't actually use the first two inputs?

        # Calculate Kcb, Ke, ETc
        # print(foo_day.dt_string)
        # if foo_day.dt_string == '1992-03-30':
        #     print(foo_day.irr_day)
        #     print(tracker.depl_root)
        #     print(tracker.max_irr_rate)
        #     # print(tracker.depl_root)
        #
        # if foo_day.dt_string == '1992-03-31':
        #     print(foo_day.irr_day)  # is this the first irrigated day?
        #     print(tracker.depl_root)
        #     print(tracker.max_irr_rate)
        compute_field_et.compute_field_et(config, plots, tracker, foo_day,
                                          debug_flag)

        # Retrieve values from foo_day and write to output data frame
        # Eventually let compute_crop_et() write directly to output df

        if debug_flag:
            # TODO: should tracker log the tuned parameters? Where do those live?
            for i, fid in enumerate(targets):
                tracker.crop_df[fid][step_dt] = {}
                sample_idx = 0, i
                tracker.crop_df[fid][step_dt]['etref'] = foo_day.refet[sample_idx]

                eta_act = tracker.etc_act[sample_idx]
                tracker.crop_df[fid][step_dt]['capture'] = foo_day.capture[sample_idx]
                tracker.crop_df[fid][step_dt]['t'] = tracker.t[sample_idx]
                tracker.crop_df[fid][step_dt]['e'] = tracker.e[sample_idx]
                tracker.crop_df[fid][step_dt]['kc_act'] = tracker.kc_act[sample_idx]
                tracker.crop_df[fid][step_dt]['ks'] = tracker.ks[sample_idx]
                tracker.crop_df[fid][step_dt]['ke'] = tracker.ke[sample_idx]

                # water balance components
                tracker.crop_df[fid][step_dt]['et_act'] = eta_act

                ppt = foo_day.precip[sample_idx]
                tracker.crop_df[fid][step_dt]['ppt'] = ppt

                melt = tracker.melt[sample_idx]
                tracker.crop_df[fid][step_dt]['melt'] = melt
                rain = tracker.rain[sample_idx]
                tracker.crop_df[fid][step_dt]['rain'] = rain

                runoff = tracker.sro[sample_idx]
                tracker.crop_df[fid][step_dt]['runoff'] = runoff
                dperc = tracker.dperc[sample_idx]
                tracker.crop_df[fid][step_dt]['dperc'] = dperc

                depl_root = tracker.depl_root[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_root'] = depl_root
                depl_root_prev = tracker.depl_root_prev[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_root_prev'] = depl_root_prev

                daw3 = tracker.daw3[sample_idx]
                tracker.crop_df[fid][step_dt]['daw3'] = daw3
                daw3_prev = tracker.daw3_prev[sample_idx]
                tracker.crop_df[fid][step_dt]['daw3_prev'] = daw3_prev
                delta_daw3 = daw3 - daw3_prev
                tracker.crop_df[fid][step_dt]['delta_daw3'] = delta_daw3

                soil_water = tracker.soil_water[sample_idx]
                tracker.crop_df[fid][step_dt]['soil_water'] = soil_water
                soil_water_prev = tracker.soil_water_prev[sample_idx]
                tracker.crop_df[fid][step_dt]['soil_water_prev'] = soil_water_prev
                delta_soil_water = tracker.delta_soil_water[sample_idx]
                tracker.crop_df[fid][step_dt]['delta_soil_water'] = delta_soil_water

                depl_ze = tracker.depl_ze[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_ze'] = depl_ze
                tracker.crop_df[fid][step_dt]['p_rz'] = tracker.p_rz[sample_idx]
                tracker.crop_df[fid][step_dt]['p_eft'] = tracker.p_eft[sample_idx]
                tracker.crop_df[fid][step_dt]['fc'] = tracker.fc[sample_idx]
                tracker.crop_df[fid][step_dt]['few'] = tracker.few[sample_idx]
                tracker.crop_df[fid][step_dt]['aw'] = tracker.aw[sample_idx]
                tracker.crop_df[fid][step_dt]['aw3'] = tracker.aw3[sample_idx]
                tracker.crop_df[fid][step_dt]['taw'] = tracker.taw[sample_idx]
                tracker.crop_df[fid][step_dt]['taw3'] = tracker.taw3[sample_idx]
                tracker.crop_df[fid][step_dt]['irrigation'] = tracker.irr_sim[sample_idx]
                tracker.crop_df[fid][step_dt]['irr_day'] = foo_day.irr_day[sample_idx]
                tracker.crop_df[fid][step_dt]['swe'] = tracker.swe[sample_idx]
                tracker.crop_df[fid][step_dt]['snow_fall'] = tracker.snow_fall[sample_idx]
                tracker.crop_df[fid][step_dt]['tavg'] = foo_day.temp_avg[sample_idx]
                tracker.crop_df[fid][step_dt]['tmax'] = foo_day.max_temp[sample_idx]
                tracker.crop_df[fid][step_dt]['zr'] = tracker.zr[sample_idx]
                tracker.crop_df[fid][step_dt]['kc_bas'] = tracker.kc_bas[sample_idx]
                tracker.crop_df[fid][step_dt]['niwr'] = tracker.niwr[sample_idx]
                tracker.crop_df[fid][step_dt]['et_bas'] = tracker.etc_bas
                tracker.crop_df[fid][step_dt]['season'] = tracker.in_season

                water_out = eta_act + dperc + runoff
                water_stored = soil_water - soil_water_prev
                water_in = melt + rain
                balance = water_in - water_stored - water_out

                tracker.crop_df[fid][step_dt]['wbal'] = balance

                if abs(balance) > 0.1 and foo_day.year > 2000:
                    pass
                    # raise WaterBalanceError('Check November water balance')

        else:
            if np.isnan(tracker.kc_act.any()):
                raise ValueError('NaN in Kc_act')

            if np.isnan(tracker.swe.any()):
                raise ValueError('NaN in SWE')

            etf[j, :] = tracker.etc_act  # ? etc_act vs eta_act? (eta_act is not a thing in tracker.)
            swe[j, :] = tracker.swe

    if debug_flag:
        # pass final dataset to calling script
        tracker.crop_df = {fid: pd.DataFrame().from_dict(tracker.crop_df[fid], orient='index')[OUTPUT_FMT]
                           for fid in targets}  # turn nested dict to single dict of pd df
        for fid in tracker.crop_df:
            tracker.crop_df[fid].index = pd.to_datetime(tracker.crop_df[fid].index)  # turn indices into dt
            tracker.crop_df[fid].index = tracker.crop_df[fid].index.rename('date')  # rename index (should be after concat)
        out_ds_list = [tracker.crop_df[fid].to_xarray() for fid in targets]  # convert to xarray
        out_ds = xarray.concat(out_ds_list, pd.Index(tracker.crop_df.keys(), name=config.field_index))  # create out 1 ds w/ 2 dims

        if save_out:
            out_ds.to_netcdf(save_out, engine='netcdf4')
        return out_ds

    else:
        # if not debug, just return the actual ET and SWE results as ndarray
        return np.asarray([etf, swe])


# Edited from field_day_loop_nc_4, trying to make model run faster by using dicts.
# Use this one now, much faster.
def field_day_loop_nc_4_dict(config, plots, params=None, debug_flag=False, save_out=None) -> np.ndarray | xarray.Dataset:
    """Run SWIM. Main model code for use with netcdf input files.

    This function loops through the daily time series for all fields at the same time.

    Args:
        config: obj, loaded config file information
        plots: obj, contains input data
        debug_flag: bool, optional; if False, return only swe and eta timeseries info as numpy array (default);
          if True, output all timeseries info as xarray dataset
        params: dict, optional; if provided, specifies parameters for model - values may be either single numbers
          (for running one field or same values for all fields), or lists/arrays the length of the number of fields
          being run. If not provided, DEFAULT parameter values will be used for ndvi_alpha, ndvi_beta, mad, swe_alpha,
          and swe_beta, and values from plots will be used for aw, rew, and tew.
        save_out: str, optional; if provided, and debug_flag=True, the filepath where all timeseries information
          will be saved; else, results will only be stored in memory.

    Returns:
        If debug_flag==False, return 2xN array of swe and eta timeseries info; If debug_flag==True, return all
        timeseries as xarray dataset.
    """
    #
    etf, swe = None, None
    size = plots.input['dims'][config.field_index]
    # size = len(selected)
    # tracker.load_soils_nc(plots[selected])  # does this work?
    tracker = PlotTracker(size)
    tracker.load_soils_nc_dict(plots)  # supplies aw, rew, tew parameters, and other values.

    # apply calibration parameter updates here

    # Assumes params is a dictionary of values.
    if params:
        if isinstance(params['ndvi_alpha'], (int, float)):  # same values for all fields.
            for k, v in params.items():
                arr = np.ones((1, size)) * v
                tracker.__setattr__(k, arr)
        else:  # unique values for each field. - not sure this works yet.
            for k, v in params.items():
                arr = np.reshape(v, (1, -1))
                tracker.__setattr__(k, arr)
    else:  # same values for all fields, except aw, rew, tew from load_soils_nc() above.
        for k, v in DEFAULTS.items():
            arr = np.ones((1, size)) * v
            tracker.__setattr__(k, arr)

    targets = plots.input['coords'][config.field_index]['data']

    # Initialize crop data frame
    time_range = pd.date_range(config.start_dt, config.end_dt, freq='D')
    if debug_flag:
        tracker.setup_dataframe(targets)  # creates empty dict of dict with FIDs as keys
    else:
        # creating properly-sized arrays for model results.
        empty = np.zeros((len(time_range), len(targets))) * np.nan
        etf, swe = empty.copy(), empty.copy()

    tracker.set_kc_max()

    foo_day = DayData()
    foo_day.sdays = 0
    foo_day.doy_prev = 0
    foo_day.irr_status = None

    hr_ppt_keys = ['prcp_hr_{}'.format(str(i).rjust(2, '0')) for i in range(0, 24)]
    # cols = ['ndvi_irr', 'etf_irr_ct', '{}_mm_corrected'.format(config.refet_type),
    #         'ndvi_inv_irr', 'etf_inv_irr_ct', '{}_mm'.format(config.refet_type)]

    # looping through days
    for j, step_dt in enumerate(time_range):
        # I think this select statement is ruining everything.
        # vals = plots.input.sel(date=step_dt)  # all data for all fields for that date
        vals = plots.input['data_vars']  # all data for all fields

        # Track variables for each day
        # For now, cast all values to native Python types
        foo_day.sdays += 1
        dt = pd.to_datetime(step_dt)
        foo_day.dt_string = dt.strftime('%Y-%m-%d')

        foo_day.year = dt.year
        foo_day.month = dt.month
        foo_day.day = dt.day
        foo_day.doy = dt.dayofyear

        # Check irrigation status on first date and first day of each year.
        if foo_day.doy == 1 or foo_day.irr_status is None:
            # find index of year in data
            yr_idx = np.argwhere(plots.input['coords']['year']['data'] == dt.year)[0][0]
            # apply index to get irrigated fraction for each field for that year.
            foo_day.irr_status = vals['irr']['data'][:, yr_idx]

        # Using yearly irr_status as condition for which variable type to store for each field on this day
        foo_day.ndvi = np.where(foo_day.irr_status, vals['ndvi_irr']['data'][j], vals['ndvi_inv_irr']['data'][j])
        foo_day.capture = np.where(foo_day.irr_status, vals['ndvi_irr_ct']['data'][j], vals['ndvi_inv_irr_ct']['data'][j])  # why were these etf, and not ndvi?
        foo_day.refet = np.where(foo_day.irr_status, vals['{}_mm_corrected'.format(config.refet_type)]['data'][j],
                                 vals['{}_mm'.format(config.refet_type)]['data'][j])

        foo_day.ndvi = foo_day.ndvi.reshape(1, -1)
        foo_day.capture = foo_day.capture.reshape(1, -1)
        foo_day.refet = foo_day.refet.reshape(1, -1)

        foo_day.irr_day = vals['irr_days']['data'][j].reshape(1, -1).astype(bool)
        foo_day.min_temp = vals['tmin_c']['data'][j].reshape(1, -1)
        foo_day.max_temp = vals['tmax_c']['data'][j].reshape(1, -1)
        foo_day.temp_avg = (foo_day.min_temp + foo_day.max_temp) / 2.
        foo_day.srad = vals['srad_wm2']['data'][j].reshape(1, -1)  # where is this used?
        foo_day.precip = vals['prcp_mm']['data'][j]

        if np.any(foo_day.precip > 0.):
            hr_ppt = np.array([vals[k]['data'][j] for k in hr_ppt_keys]).reshape(24, size)
            foo_day.hr_precip = hr_ppt  # where is this even used?

        foo_day.precip = foo_day.precip.reshape(1, -1)

        # Calculate height of vegetation.
        # Moved up to this point 12/26/07 for use in adj. Kcb and kc_max
        calculate_height.calculate_height(tracker)

        # Interpolate Kcb and make climate adjustment (for ETo basis)
        obs_kcb_daily.kcb_daily(config, plots, tracker, foo_day)  # It doesn't actually use the first two inputs?

        # Calculate Kcb, Ke, ETc
        # print(foo_day.dt_string)
        # if foo_day.dt_string == '1992-03-30':
        #     print(foo_day.irr_day)
        #     print(tracker.depl_root)
        #     print(tracker.max_irr_rate)
        #     # print(tracker.depl_root)
        #
        # if foo_day.dt_string == '1992-03-31':
        #     print(foo_day.irr_day)  # is this the first irrigated day?
        #     print(tracker.depl_root)
        #     print(tracker.max_irr_rate)
        compute_field_et.compute_field_et(config, plots, tracker, foo_day,
                                          debug_flag)

        # Retrieve values from foo_day and write to output data frame
        # Eventually let compute_crop_et() write directly to output df

        if debug_flag:
            # TODO: should tracker log the tuned parameters? Where do those live?
            for i, fid in enumerate(targets):
                tracker.crop_df[fid][step_dt] = {}
                sample_idx = 0, i
                tracker.crop_df[fid][step_dt]['etref'] = foo_day.refet[sample_idx]

                eta_act = tracker.etc_act[sample_idx]
                tracker.crop_df[fid][step_dt]['capture'] = foo_day.capture[sample_idx]
                tracker.crop_df[fid][step_dt]['t'] = tracker.t[sample_idx]
                tracker.crop_df[fid][step_dt]['e'] = tracker.e[sample_idx]
                tracker.crop_df[fid][step_dt]['kc_act'] = tracker.kc_act[sample_idx]
                tracker.crop_df[fid][step_dt]['ks'] = tracker.ks[sample_idx]
                tracker.crop_df[fid][step_dt]['ke'] = tracker.ke[sample_idx]

                # water balance components
                tracker.crop_df[fid][step_dt]['et_act'] = eta_act

                ppt = foo_day.precip[sample_idx]
                tracker.crop_df[fid][step_dt]['ppt'] = ppt

                melt = tracker.melt[sample_idx]
                tracker.crop_df[fid][step_dt]['melt'] = melt
                rain = tracker.rain[sample_idx]
                tracker.crop_df[fid][step_dt]['rain'] = rain

                runoff = tracker.sro[sample_idx]
                tracker.crop_df[fid][step_dt]['runoff'] = runoff
                dperc = tracker.dperc[sample_idx]
                tracker.crop_df[fid][step_dt]['dperc'] = dperc

                depl_root = tracker.depl_root[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_root'] = depl_root
                depl_root_prev = tracker.depl_root_prev[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_root_prev'] = depl_root_prev

                daw3 = tracker.daw3[sample_idx]
                tracker.crop_df[fid][step_dt]['daw3'] = daw3
                daw3_prev = tracker.daw3_prev[sample_idx]
                tracker.crop_df[fid][step_dt]['daw3_prev'] = daw3_prev
                delta_daw3 = daw3 - daw3_prev
                tracker.crop_df[fid][step_dt]['delta_daw3'] = delta_daw3

                soil_water = tracker.soil_water[sample_idx]
                tracker.crop_df[fid][step_dt]['soil_water'] = soil_water
                soil_water_prev = tracker.soil_water_prev[sample_idx]
                tracker.crop_df[fid][step_dt]['soil_water_prev'] = soil_water_prev
                delta_soil_water = tracker.delta_soil_water[sample_idx]
                tracker.crop_df[fid][step_dt]['delta_soil_water'] = delta_soil_water

                depl_ze = tracker.depl_ze[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_ze'] = depl_ze
                tracker.crop_df[fid][step_dt]['p_rz'] = tracker.p_rz[sample_idx]
                tracker.crop_df[fid][step_dt]['p_eft'] = tracker.p_eft[sample_idx]
                tracker.crop_df[fid][step_dt]['fc'] = tracker.fc[sample_idx]
                tracker.crop_df[fid][step_dt]['few'] = tracker.few[sample_idx]
                tracker.crop_df[fid][step_dt]['aw'] = tracker.aw[sample_idx]
                tracker.crop_df[fid][step_dt]['aw3'] = tracker.aw3[sample_idx]
                tracker.crop_df[fid][step_dt]['taw'] = tracker.taw[sample_idx]
                tracker.crop_df[fid][step_dt]['taw3'] = tracker.taw3[sample_idx]
                tracker.crop_df[fid][step_dt]['irrigation'] = tracker.irr_sim[sample_idx]
                tracker.crop_df[fid][step_dt]['irr_day'] = foo_day.irr_day[sample_idx]
                tracker.crop_df[fid][step_dt]['swe'] = tracker.swe[sample_idx]
                tracker.crop_df[fid][step_dt]['snow_fall'] = tracker.snow_fall[sample_idx]
                tracker.crop_df[fid][step_dt]['tavg'] = foo_day.temp_avg[sample_idx]
                tracker.crop_df[fid][step_dt]['tmax'] = foo_day.max_temp[sample_idx]
                tracker.crop_df[fid][step_dt]['zr'] = tracker.zr[sample_idx]
                tracker.crop_df[fid][step_dt]['kc_bas'] = tracker.kc_bas[sample_idx]
                tracker.crop_df[fid][step_dt]['niwr'] = tracker.niwr[sample_idx]
                tracker.crop_df[fid][step_dt]['et_bas'] = tracker.etc_bas
                tracker.crop_df[fid][step_dt]['season'] = tracker.in_season

                water_out = eta_act + dperc + runoff
                water_stored = soil_water - soil_water_prev
                water_in = melt + rain
                balance = water_in - water_stored - water_out

                tracker.crop_df[fid][step_dt]['wbal'] = balance

                if abs(balance) > 0.1 and foo_day.year > 2000:
                    pass
                    # raise WaterBalanceError('Check November water balance')

        else:
            if np.isnan(tracker.kc_act.any()):
                raise ValueError('NaN in Kc_act')

            if np.isnan(tracker.swe.any()):
                raise ValueError('NaN in SWE')

            etf[j, :] = tracker.etc_act  # ? etc_act vs eta_act? (eta_act is not a thing in tracker.)
            swe[j, :] = tracker.swe

    if debug_flag:
        # pass final dataset to calling script
        tracker.crop_df = {fid: pd.DataFrame().from_dict(tracker.crop_df[fid], orient='index')[OUTPUT_FMT]
                           for fid in targets}  # turn nested dict to single dict of pd df
        for fid in tracker.crop_df:
            tracker.crop_df[fid].index = pd.to_datetime(tracker.crop_df[fid].index)  # turn indices into dt
            tracker.crop_df[fid].index = tracker.crop_df[fid].index.rename('date')  # rename index (should be after concat)
        out_ds_list = [tracker.crop_df[fid].to_xarray() for fid in targets]  # convert to xarray
        out_ds = xarray.concat(out_ds_list, pd.Index(tracker.crop_df.keys(), name=config.field_index))  # create out 1 ds w/ 2 dims

        if save_out:
            out_ds.to_netcdf(save_out, engine='netcdf4')
        return out_ds

    else:
        # if not debug, just return the actual ET and SWE results as ndarray
        return np.asarray([etf, swe])


# Edited from field_day_loop_nc_1, altering parameter input. If selecting fields, do that outside the model function.
def field_day_loop_nc_3(config, plots, debug_flag=False, params=None, save_out=None):
    """ Run SWIM. Main model code for use with netcdf input files.

    This function loops through the daily time series for all fields at the same time.

    config: obj, loaded config file information
    plots: obj, contains input data
    debug_flag: bool, optional; if False, return only swe and etf timeseries info as numpy array (default);
    if True, output all timeseries info as xarray dataset
    params: dict, optional; if provided, specifies parameters for model - values may be either single numbers
    (for running one field or same values for all fields), or lists/arrays the length of the number of fields being run.
    If not provided, DEFAULT parameter values will be used for ndvi_alpha, ndvi_beta, mad, swe_alpha, and swe_beta,
    and values from plots will be used for aw, rew, and tew.
    save_out: str, optional; if provided, and debug_flag=True, the filepath where all timeseries information
    will be saved; else, results will only be stored in memory.
    """
    #
    etf, swe = None, None
    size = len(plots.input[config.field_index].values)
    # size = len(selected)
    # tracker.load_soils_nc(plots[selected])  # does this work?
    tracker = PlotTracker(size)
    tracker.load_soils_nc(plots)  # supplies aw, rew, tew parameters, and other values.

    # apply calibration parameter updates here

    # Assumes params is a dictionary of values.
    if params:
        if isinstance(params['ndvi_alpha'], (int, float)):  # same values for all fields.
            for k, v in params.items():
                arr = np.ones((1, size)) * v
                tracker.__setattr__(k, arr)
        else:  # unique values for each field. - not sure this works yet.
            for k, v in params.items():
                arr = np.reshape(v, (1, -1))
                tracker.__setattr__(k, arr)
    else:  # same values for all fields, except aw, rew, tew from load_soils_nc() above.
        for k, v in DEFAULTS.items():
            arr = np.ones((1, size)) * v
            tracker.__setattr__(k, arr)

    targets = plots.input[config.field_index].values

    # Initialize crop data frame
    time_range = pd.date_range(config.start_dt, config.end_dt, freq='D')
    if debug_flag:
        tracker.setup_dataframe(targets)  # creates empty dict of dict with FIDs as keys
    else:
        # creating properly-sized arrays for model results.
        empty = np.zeros((len(time_range), len(targets))) * np.nan
        etf, swe = empty.copy(), empty.copy()

    tracker.set_kc_max()

    foo_day = DayData()
    foo_day.sdays = 0
    foo_day.doy_prev = 0
    foo_day.irr_status = None

    hr_ppt_keys = ['prcp_hr_{}'.format(str(i).rjust(2, '0')) for i in range(0, 24)]
    # cols = ['ndvi_irr', 'etf_irr_ct', '{}_mm_corrected'.format(config.refet_type),
    #         'ndvi_inv_irr', 'etf_inv_irr_ct', '{}_mm'.format(config.refet_type)]

    # looping through days
    for j, step_dt in enumerate(time_range):
        # I think this select statement is ruining everything.
        vals = plots.input.sel(date=step_dt)  # all data for all fields for that date

        # Track variables for each day
        # For now, cast all values to native Python types
        foo_day.sdays += 1
        dt = pd.to_datetime(step_dt)
        foo_day.dt_string = dt.strftime('%Y-%m-%d')

        foo_day.year = dt.year
        foo_day.month = dt.month
        foo_day.day = dt.day
        foo_day.doy = dt.dayofyear

        # Check irrigation status on first date and first day of each year.
        if foo_day.doy == 1 or foo_day.irr_status is None:
            foo_day.irr_status = np.array([plots.input['irr'].sel(year=dt.year).values])

        # Using yearly irr_status as condition for which variable type to store for each field on this day
        foo_day.ndvi = np.where(foo_day.irr_status, vals['ndvi_irr'], vals['ndvi_inv_irr'])
        foo_day.capture = np.where(foo_day.irr_status, vals['ndvi_irr_ct'], vals['ndvi_inv_irr_ct'])  # why were these etf, and not ndvi?
        foo_day.refet = np.where(foo_day.irr_status, vals['{}_mm_corrected'.format(config.refet_type)],
                                 vals['{}_mm'.format(config.refet_type)])

        # Why would refet be different whether or not it's irrigated?
        foo_day.irr_day = np.array(vals['irr_days']).reshape(1, -1)
        foo_day.min_temp = np.array(vals['tmin_c']).reshape(1, -1)
        foo_day.max_temp = np.array(vals['tmax_c']).reshape(1, -1)
        foo_day.temp_avg = (foo_day.min_temp + foo_day.max_temp) / 2.
        foo_day.srad = np.array(vals['srad_wm2']).reshape(1, -1)
        foo_day.precip = np.array(vals['prcp_mm'])

        if np.any(foo_day.precip > 0.):
            hr_ppt = np.array([vals[k] for k in hr_ppt_keys]).reshape(24, size)
            foo_day.hr_precip = hr_ppt

        foo_day.precip = foo_day.precip.reshape(1, -1)

        # Calculate height of vegetation.
        # Moved up to this point 12/26/07 for use in adj. Kcb and kc_max
        calculate_height.calculate_height(tracker)

        # Interpolate Kcb and make climate adjustment (for ETo basis)
        obs_kcb_daily.kcb_daily(config, plots, tracker, foo_day)  # It doesn't actually use the first two inputs?

        # Calculate Kcb, Ke, ETc
        compute_field_et.compute_field_et(config, plots, tracker, foo_day,
                                          debug_flag)

        # Retrieve values from foo_day and write to output data frame
        # Eventually let compute_crop_et() write directly to output df

        if debug_flag:
            # TODO: should tracker log the tuned parameters? Where do those live?
            for i, fid in enumerate(targets):
                tracker.crop_df[fid][step_dt] = {}
                sample_idx = 0, i
                tracker.crop_df[fid][step_dt]['etref'] = foo_day.refet[sample_idx]

                eta_act = tracker.etc_act[sample_idx]
                tracker.crop_df[fid][step_dt]['capture'] = foo_day.capture[sample_idx]
                tracker.crop_df[fid][step_dt]['t'] = tracker.t[sample_idx]
                tracker.crop_df[fid][step_dt]['e'] = tracker.e[sample_idx]
                tracker.crop_df[fid][step_dt]['kc_act'] = tracker.kc_act[sample_idx]
                tracker.crop_df[fid][step_dt]['ks'] = tracker.ks[sample_idx]
                tracker.crop_df[fid][step_dt]['ke'] = tracker.ke[sample_idx]

                # water balance components
                tracker.crop_df[fid][step_dt]['et_act'] = eta_act

                ppt = foo_day.precip[sample_idx]
                tracker.crop_df[fid][step_dt]['ppt'] = ppt

                melt = tracker.melt[sample_idx]
                tracker.crop_df[fid][step_dt]['melt'] = melt
                rain = tracker.rain[sample_idx]
                tracker.crop_df[fid][step_dt]['rain'] = rain

                runoff = tracker.sro[sample_idx]
                tracker.crop_df[fid][step_dt]['runoff'] = runoff
                dperc = tracker.dperc[sample_idx]
                tracker.crop_df[fid][step_dt]['dperc'] = dperc

                depl_root = tracker.depl_root[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_root'] = depl_root
                depl_root_prev = tracker.depl_root_prev[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_root_prev'] = depl_root_prev

                daw3 = tracker.daw3[sample_idx]
                tracker.crop_df[fid][step_dt]['daw3'] = daw3
                daw3_prev = tracker.daw3_prev[sample_idx]
                tracker.crop_df[fid][step_dt]['daw3_prev'] = daw3_prev
                delta_daw3 = daw3 - daw3_prev
                tracker.crop_df[fid][step_dt]['delta_daw3'] = delta_daw3

                soil_water = tracker.soil_water[sample_idx]
                tracker.crop_df[fid][step_dt]['soil_water'] = soil_water
                soil_water_prev = tracker.soil_water_prev[sample_idx]
                tracker.crop_df[fid][step_dt]['soil_water_prev'] = soil_water_prev
                delta_soil_water = tracker.delta_soil_water[sample_idx]
                tracker.crop_df[fid][step_dt]['delta_soil_water'] = delta_soil_water

                depl_ze = tracker.depl_ze[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_ze'] = depl_ze
                tracker.crop_df[fid][step_dt]['p_rz'] = tracker.p_rz[sample_idx]
                tracker.crop_df[fid][step_dt]['p_eft'] = tracker.p_eft[sample_idx]
                tracker.crop_df[fid][step_dt]['fc'] = tracker.fc[sample_idx]
                tracker.crop_df[fid][step_dt]['few'] = tracker.few[sample_idx]
                tracker.crop_df[fid][step_dt]['aw'] = tracker.aw[sample_idx]
                tracker.crop_df[fid][step_dt]['aw3'] = tracker.aw3[sample_idx]
                tracker.crop_df[fid][step_dt]['taw'] = tracker.taw[sample_idx]
                tracker.crop_df[fid][step_dt]['taw3'] = tracker.taw3[sample_idx]
                tracker.crop_df[fid][step_dt]['irrigation'] = tracker.irr_sim[sample_idx]
                tracker.crop_df[fid][step_dt]['irr_day'] = foo_day.irr_day[sample_idx]
                tracker.crop_df[fid][step_dt]['swe'] = tracker.swe[sample_idx]
                tracker.crop_df[fid][step_dt]['snow_fall'] = tracker.snow_fall[sample_idx]
                tracker.crop_df[fid][step_dt]['tavg'] = foo_day.temp_avg[sample_idx]
                tracker.crop_df[fid][step_dt]['tmax'] = foo_day.max_temp[sample_idx]
                tracker.crop_df[fid][step_dt]['zr'] = tracker.zr[sample_idx]
                tracker.crop_df[fid][step_dt]['kc_bas'] = tracker.kc_bas[sample_idx]
                tracker.crop_df[fid][step_dt]['niwr'] = tracker.niwr[sample_idx]
                tracker.crop_df[fid][step_dt]['et_bas'] = tracker.etc_bas
                tracker.crop_df[fid][step_dt]['season'] = tracker.in_season

                water_out = eta_act + dperc + runoff
                water_stored = soil_water - soil_water_prev
                water_in = melt + rain
                balance = water_in - water_stored - water_out

                tracker.crop_df[fid][step_dt]['wbal'] = balance

                if abs(balance) > 0.1 and foo_day.year > 2000:
                    pass
                    # raise WaterBalanceError('Check November water balance')

        else:
            if np.isnan(tracker.kc_act.any()):
                raise ValueError('NaN in Kc_act')

            if np.isnan(tracker.swe.any()):
                raise ValueError('NaN in SWE')

            etf[j, :] = tracker.kc_act
            swe[j, :] = tracker.swe

    if debug_flag:
        # pass final dataset to calling script
        tracker.crop_df = {fid: pd.DataFrame().from_dict(tracker.crop_df[fid], orient='index')[OUTPUT_FMT]
                           for fid in targets}  # turn nested dict to single dict of pd df
        for fid in tracker.crop_df:
            tracker.crop_df[fid].index = pd.to_datetime(tracker.crop_df[fid].index)  # turn indices into dt
            tracker.crop_df[fid].index = tracker.crop_df[fid].index.rename('date')  # rename index (should be after concat)
        out_ds_list = [tracker.crop_df[fid].to_xarray() for fid in targets]  # convert to xarray
        out_ds = xarray.concat(out_ds_list, pd.Index(tracker.crop_df.keys(), name=config.field_index))  # create out 1 ds w/ 2 dims

        if save_out:
            out_ds.to_netcdf(save_out, engine='netcdf4')
        return out_ds

    else:
        # if not debug, just return the actual ET and SWE results as ndarray
        return np.asarray([etf, swe])


# This is what I'm starting with.
def field_day_loop_nc_1(config, plots, debug_flag=False, params=None, save_out=None):
    etf, swe = None, None
    size = len(plots.input[config.field_index].values)
    tracker = PlotTracker(size)
    tracker.load_soils_nc(plots)

    # apply calibration parameter updates here
    if config.calibration:  # this has not been updated
        # load PEST++ parameter proposition
        cal_arr = {k: np.zeros((1, size)) for k in config.calibration_groups}

        for k, f in config.calibration_files.items():

            group, fid = '_'.join(k.split('_')[:-1]), k.split('_')[-1]
            idx = plots.input['order'].index(fid)

            if params:
                value = params[k]
            else:
                v = pd.read_csv(f, index_col=None, header=0)
                value = v.loc[0, '1']

            cal_arr[group][0, idx] = value

        for k, v in cal_arr.items():

            tracker.__setattr__(k, v)

            if debug_flag:
                print('{}: {}'.format(k, ['{:.2f}'.format(p) for p in v.flatten()]))

    elif config.forecast:  # this has not been updated

        param_arr = {k: np.zeros((1, size)) for k in config.forecast_parameter_groups}

        for k, v in config.forecast_parameters.items():

            group, fid = '_'.join(k.split('_')[:-1]), k.split('_')[-1]

            # PEST++ has lower-cased the FIDs
            l = [x.lower() for x in plots.input['order']]
            idx = l.index(fid)

            if fid not in l:
                continue

            if params:
                value = params[k]
            else:
                value = v

            param_arr[group][0, idx] = value

        for k, v in param_arr.items():

            tracker.__setattr__(k, v)

            if debug_flag:
                print('{}: {}'.format(k, ['{:.2f}'.format(p) for p in v.flatten()]))

    else:
        for k, v in DEFAULTS.items():
            arr = np.ones((1, size)) * v
            tracker.__setattr__(k, arr)

    targets = plots.input[config.field_index].values

    # valid_data = {dt: val for dt, val in plots.input['time_series'].items() if
    #               (config.start_dt <= pd.to_datetime(dt) <= config.end_dt)}

    # Initialize crop data frame
    time_range = pd.date_range(config.start_dt, config.end_dt, freq='D')
    if debug_flag:
        tracker.setup_dataframe(targets)  # creates empty dict of dict with FIDs as keys
    else:
        # Isn't this stuff essentially already done? It's in an xarray by FID and date...
        empty = np.zeros((len(time_range), len(targets))) * np.nan
        etf, swe = empty.copy(), empty.copy()

    tracker.set_kc_max()

    foo_day = DayData()
    foo_day.sdays = 0
    foo_day.doy_prev = 0
    foo_day.irr_status = None

    hr_ppt_keys = ['prcp_hr_{}'.format(str(i).rjust(2, '0')) for i in range(0, 24)]
    # cols = ['ndvi_irr', 'etf_irr_ct', '{}_mm_corrected'.format(config.refet_type),
    #         'ndvi_inv_irr', 'etf_inv_irr_ct', '{}_mm'.format(config.refet_type)]

    # looping through days
    for j, step_dt in enumerate(time_range):
        # I think this select statement is ruining everything.
        vals = plots.input.sel(date=step_dt)  # all data for all fields for that date

        # Track variables for each day
        # For now, cast all values to native Python types
        foo_day.sdays += 1
        dt = pd.to_datetime(step_dt)
        foo_day.dt_string = dt.strftime('%Y-%m-%d')

        foo_day.year = dt.year
        foo_day.month = dt.month
        foo_day.day = dt.day
        foo_day.doy = dt.dayofyear

        # Check irrigation status on first date and first day of each year.
        if foo_day.doy == 1 or foo_day.irr_status is None:
            foo_day.irr_status = np.array([plots.input['irr'].sel(year=dt.year).values])

        # Using yearly irr_status as condition for which variable type to store for each field on this day
        foo_day.ndvi = np.where(foo_day.irr_status, vals['ndvi_irr'], vals['ndvi_inv_irr'])
        foo_day.capture = np.where(foo_day.irr_status, vals['ndvi_irr_ct'], vals['ndvi_inv_irr_ct'])  # why were these etf, and not ndvi?
        foo_day.refet = np.where(foo_day.irr_status, vals['{}_mm_corrected'.format(config.refet_type)],
                                 vals['{}_mm'.format(config.refet_type)])

        # Why would refet be different whether or not it's irrigated?
        foo_day.irr_day = np.array(vals['irr_days']).reshape(1, -1)
        foo_day.min_temp = np.array(vals['tmin_c']).reshape(1, -1)
        foo_day.max_temp = np.array(vals['tmax_c']).reshape(1, -1)
        foo_day.temp_avg = (foo_day.min_temp + foo_day.max_temp) / 2.
        foo_day.srad = np.array(vals['srad_wm2']).reshape(1, -1)
        foo_day.precip = np.array(vals['prcp_mm'])

        if np.any(foo_day.precip > 0.):
            hr_ppt = np.array([vals[k] for k in hr_ppt_keys]).reshape(24, size)
            foo_day.hr_precip = hr_ppt

        foo_day.precip = foo_day.precip.reshape(1, -1)

        # Calculate height of vegetation.
        # Moved up to this point 12/26/07 for use in adj. Kcb and kc_max
        calculate_height.calculate_height(tracker)

        # Interpolate Kcb and make climate adjustment (for ETo basis)
        obs_kcb_daily.kcb_daily(config, plots, tracker, foo_day)  # It doesn't actually use the first two inputs?

        # Calculate Kcb, Ke, ETc
        compute_field_et.compute_field_et(config, plots, tracker, foo_day,
                                          debug_flag)

        # Retrieve values from foo_day and write to output data frame
        # Eventually let compute_crop_et() write directly to output df

        if debug_flag:
            # TODO: should tracker log the tuned parameters? Where do those live?
            for i, fid in enumerate(targets):
                tracker.crop_df[fid][step_dt] = {}
                sample_idx = 0, i
                tracker.crop_df[fid][step_dt]['etref'] = foo_day.refet[sample_idx]

                eta_act = tracker.etc_act[sample_idx]
                tracker.crop_df[fid][step_dt]['capture'] = foo_day.capture[sample_idx]
                tracker.crop_df[fid][step_dt]['t'] = tracker.t[sample_idx]
                tracker.crop_df[fid][step_dt]['e'] = tracker.e[sample_idx]
                tracker.crop_df[fid][step_dt]['kc_act'] = tracker.kc_act[sample_idx]
                tracker.crop_df[fid][step_dt]['ks'] = tracker.ks[sample_idx]
                tracker.crop_df[fid][step_dt]['ke'] = tracker.ke[sample_idx]

                # water balance components
                tracker.crop_df[fid][step_dt]['et_act'] = eta_act

                ppt = foo_day.precip[sample_idx]
                tracker.crop_df[fid][step_dt]['ppt'] = ppt

                melt = tracker.melt[sample_idx]
                tracker.crop_df[fid][step_dt]['melt'] = melt
                rain = tracker.rain[sample_idx]
                tracker.crop_df[fid][step_dt]['rain'] = rain

                runoff = tracker.sro[sample_idx]
                tracker.crop_df[fid][step_dt]['runoff'] = runoff
                dperc = tracker.dperc[sample_idx]
                tracker.crop_df[fid][step_dt]['dperc'] = dperc

                depl_root = tracker.depl_root[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_root'] = depl_root
                depl_root_prev = tracker.depl_root_prev[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_root_prev'] = depl_root_prev

                daw3 = tracker.daw3[sample_idx]
                tracker.crop_df[fid][step_dt]['daw3'] = daw3
                daw3_prev = tracker.daw3_prev[sample_idx]
                tracker.crop_df[fid][step_dt]['daw3_prev'] = daw3_prev
                delta_daw3 = daw3 - daw3_prev
                tracker.crop_df[fid][step_dt]['delta_daw3'] = delta_daw3

                soil_water = tracker.soil_water[sample_idx]
                tracker.crop_df[fid][step_dt]['soil_water'] = soil_water
                soil_water_prev = tracker.soil_water_prev[sample_idx]
                tracker.crop_df[fid][step_dt]['soil_water_prev'] = soil_water_prev
                delta_soil_water = tracker.delta_soil_water[sample_idx]
                tracker.crop_df[fid][step_dt]['delta_soil_water'] = delta_soil_water

                depl_ze = tracker.depl_ze[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_ze'] = depl_ze
                tracker.crop_df[fid][step_dt]['p_rz'] = tracker.p_rz[sample_idx]
                tracker.crop_df[fid][step_dt]['p_eft'] = tracker.p_eft[sample_idx]
                tracker.crop_df[fid][step_dt]['fc'] = tracker.fc[sample_idx]
                tracker.crop_df[fid][step_dt]['few'] = tracker.few[sample_idx]
                tracker.crop_df[fid][step_dt]['aw'] = tracker.aw[sample_idx]
                tracker.crop_df[fid][step_dt]['aw3'] = tracker.aw3[sample_idx]
                tracker.crop_df[fid][step_dt]['taw'] = tracker.taw[sample_idx]
                tracker.crop_df[fid][step_dt]['taw3'] = tracker.taw3[sample_idx]
                tracker.crop_df[fid][step_dt]['irrigation'] = tracker.irr_sim[sample_idx]
                tracker.crop_df[fid][step_dt]['irr_day'] = foo_day.irr_day[sample_idx]
                tracker.crop_df[fid][step_dt]['swe'] = tracker.swe[sample_idx]
                tracker.crop_df[fid][step_dt]['snow_fall'] = tracker.snow_fall[sample_idx]
                tracker.crop_df[fid][step_dt]['tavg'] = foo_day.temp_avg[sample_idx]
                tracker.crop_df[fid][step_dt]['tmax'] = foo_day.max_temp[sample_idx]
                tracker.crop_df[fid][step_dt]['zr'] = tracker.zr[sample_idx]
                tracker.crop_df[fid][step_dt]['kc_bas'] = tracker.kc_bas[sample_idx]
                tracker.crop_df[fid][step_dt]['niwr'] = tracker.niwr[sample_idx]
                tracker.crop_df[fid][step_dt]['et_bas'] = tracker.etc_bas
                tracker.crop_df[fid][step_dt]['season'] = tracker.in_season

                water_out = eta_act + dperc + runoff
                water_stored = soil_water - soil_water_prev
                water_in = melt + rain
                balance = water_in - water_stored - water_out

                tracker.crop_df[fid][step_dt]['wbal'] = balance

                if abs(balance) > 0.1 and foo_day.year > 2000:
                    pass
                    # raise WaterBalanceError('Check November water balance')

        else:
            if np.isnan(tracker.kc_act.any()):
                raise ValueError('NaN in Kc_act')

            if np.isnan(tracker.swe.any()):
                raise ValueError('NaN in SWE')

            etf[j, :] = tracker.kc_act
            swe[j, :] = tracker.swe

    if debug_flag:
        # pass final dataset to calling script
        tracker.crop_df = {fid: pd.DataFrame().from_dict(tracker.crop_df[fid], orient='index')[OUTPUT_FMT]
                           for fid in targets}  # turn nested dict to single dict of pd df
        for fid in tracker.crop_df:
            tracker.crop_df[fid].index = pd.to_datetime(tracker.crop_df[fid].index)  # turn indices into dt
            tracker.crop_df[fid].index = tracker.crop_df[fid].index.rename('date')  # rename index (should be after concat)
        out_ds_list = [tracker.crop_df[fid].to_xarray() for fid in targets]  # convert to xarray
        out_ds = xarray.concat(out_ds_list, pd.Index(tracker.crop_df.keys(), name=config.field_index))  # create out 1 ds w/ 2 dims

        if save_out:
            out_ds.to_netcdf(save_out, engine='netcdf4')
        return out_ds

    else:
        # if not debug, just return the actual ET and SWE results as ndarray
        return np.asarray([etf, swe])


# this one uses numba decorators?
# Bad, it is not any faster.
def field_day_loop_nc_2(config, plots, debug_flag=False, params=None, save_out=None):
    etf, swe = None, None
    size = len(plots.input[config.field_index].values)
    tracker = PlotTracker(size)
    tracker.load_soils_nc(plots)

    # apply calibration parameter updates here
    if config.calibration:  # this has not been updated
        # load PEST++ parameter proposition
        cal_arr = {k: np.zeros((1, size)) for k in config.calibration_groups}

        for k, f in config.calibration_files.items():

            group, fid = '_'.join(k.split('_')[:-1]), k.split('_')[-1]
            idx = plots.input['order'].index(fid)

            if params:
                value = params[k]
            else:
                v = pd.read_csv(f, index_col=None, header=0)
                value = v.loc[0, '1']

            cal_arr[group][0, idx] = value

        for k, v in cal_arr.items():

            tracker.__setattr__(k, v)

            if debug_flag:
                print('{}: {}'.format(k, ['{:.2f}'.format(p) for p in v.flatten()]))

    elif config.forecast:  # this has not been updated

        param_arr = {k: np.zeros((1, size)) for k in config.forecast_parameter_groups}

        for k, v in config.forecast_parameters.items():

            group, fid = '_'.join(k.split('_')[:-1]), k.split('_')[-1]

            # PEST++ has lower-cased the FIDs
            l = [x.lower() for x in plots.input['order']]
            idx = l.index(fid)

            if fid not in l:
                continue

            if params:
                value = params[k]
            else:
                value = v

            param_arr[group][0, idx] = value

        for k, v in param_arr.items():

            tracker.__setattr__(k, v)

            if debug_flag:
                print('{}: {}'.format(k, ['{:.2f}'.format(p) for p in v.flatten()]))

    else:
        for k, v in DEFAULTS.items():
            arr = np.ones((1, size)) * v
            tracker.__setattr__(k, arr)

    targets = plots.input[config.field_index].values

    # valid_data = {dt: val for dt, val in plots.input['time_series'].items() if
    #               (config.start_dt <= pd.to_datetime(dt) <= config.end_dt)}

    # Initialize crop data frame
    time_range = pd.date_range(config.start_dt, config.end_dt, freq='D')
    if debug_flag:
        tracker.setup_dataframe(targets)  # creates empty dict of dict with FIDs as keys
    else:
        # Isn't this stuff essentially already done? It's in an xarray by FID and date...
        empty = np.zeros((len(time_range), len(targets))) * np.nan
        etf, swe = empty.copy(), empty.copy()

    tracker.set_kc_max()

    foo_day = DayData()
    foo_day.sdays = 0
    foo_day.doy_prev = 0
    foo_day.irr_status = None

    hr_ppt_keys = ['prcp_hr_{}'.format(str(i).rjust(2, '0')) for i in range(0, 24)]
    # cols = ['ndvi_irr', 'etf_irr_ct', '{}_mm_corrected'.format(config.refet_type),
    #         'ndvi_inv_irr', 'etf_inv_irr_ct', '{}_mm'.format(config.refet_type)]

    # looping through days
    for j, step_dt in enumerate(time_range):
        # I think this select statement is ruining everything.
        vals = plots.input.sel(date=step_dt).load()  # all data for all fields for that date

        # Track variables for each day
        # For now, cast all values to native Python types
        foo_day.sdays += 1
        dt = pd.to_datetime(step_dt)
        foo_day.dt_string = dt.strftime('%Y-%m-%d')

        foo_day.year = dt.year
        foo_day.month = dt.month
        foo_day.day = dt.day
        foo_day.doy = dt.dayofyear

        # Check irrigation status on first date and first day of each year.
        if foo_day.doy == 1 or foo_day.irr_status is None:
            foo_day.irr_status = np.array([plots.input['irr'].sel(year=dt.year).values])

        # Using yearly irr_status as condition for which variable type to store for each field on this day
        foo_day.ndvi = np.where(foo_day.irr_status, vals['ndvi_irr'], vals['ndvi_inv_irr'])
        foo_day.capture = np.where(foo_day.irr_status, vals['ndvi_irr_ct'], vals['ndvi_inv_irr_ct'])  # why were these etf, and not ndvi?
        foo_day.refet = np.where(foo_day.irr_status, vals['{}_mm_corrected'.format(config.refet_type)],
                                 vals['{}_mm'.format(config.refet_type)])

        # Why would refet be different whether or not it's irrigated?
        foo_day.irr_day = np.array(vals['irr_days']).reshape(1, -1)
        foo_day.min_temp = np.array(vals['tmin_c']).reshape(1, -1)
        foo_day.max_temp = np.array(vals['tmax_c']).reshape(1, -1)
        foo_day.temp_avg = (foo_day.min_temp + foo_day.max_temp) / 2.
        foo_day.srad = np.array(vals['srad_wm2']).reshape(1, -1)
        foo_day.precip = np.array(vals['prcp_mm'])

        if np.any(foo_day.precip > 0.):
            hr_ppt = np.array([vals[k] for k in hr_ppt_keys]).reshape(24, size)
            foo_day.hr_precip = hr_ppt

        foo_day.precip = foo_day.precip.reshape(1, -1)

        # What are the dataypes of the inputs to these functions? Are they things that numba will like?

        # Calculate height of vegetation.
        # Moved up to this point 12/26/07 for use in adj. Kcb and kc_max
        # tracker.height = calculate_height.calculate_height_1(tracker.height, tracker.kc_bas, tracker.kc_min,
        #                                                      tracker.height_initial, tracker.kc_bas_mid,
        #                                                      tracker.height_max)
        calculate_height.calculate_height(tracker)
        # This made things twice as slow.
        # This does not seem numba-able. Just too simple already? I dunno.
        # It could be a good test to see if I can get things to work with numba.

        # Interpolate Kcb and make climate adjustment (for ETo basis)
        # obs_kcb_daily.kcb_daily(config, plots, tracker, foo_day)  # It doesn't actually use the first two inputs?
        result = obs_kcb_daily.kcb_daily_1(foo_day.year, foo_day.ndvi, foo_day.refet, foo_day.doy, config.ndvi_alpha,
                                           config.ndvi_beta, tracker.grow_root, tracker.height)

        tracker.kc_bas_prev = result[0]
        tracker.etc_act = result[1]
        tracker.etc_pot = result[2]
        tracker.etc_bas = result[3]
        tracker.grow_root = result[4]
        tracker.height = result[5]

        # Calculate Kcb, Ke, ETc
        compute_field_et.compute_field_et(config, plots, tracker, foo_day,
                                          debug_flag)  # This is potentially a good place to try numba?

        # Retrieve values from foo_day and write to output data frame
        # Eventually let compute_crop_et() write directly to output df

        if debug_flag:
            # TODO: should tracker log the tuned parameters? Where do those live?
            for i, fid in enumerate(targets):
                tracker.crop_df[fid][step_dt] = {}
                sample_idx = 0, i
                tracker.crop_df[fid][step_dt]['etref'] = foo_day.refet[sample_idx]

                eta_act = tracker.etc_act[sample_idx]
                tracker.crop_df[fid][step_dt]['capture'] = foo_day.capture[sample_idx]
                tracker.crop_df[fid][step_dt]['t'] = tracker.t[sample_idx]
                tracker.crop_df[fid][step_dt]['e'] = tracker.e[sample_idx]
                tracker.crop_df[fid][step_dt]['kc_act'] = tracker.kc_act[sample_idx]
                tracker.crop_df[fid][step_dt]['ks'] = tracker.ks[sample_idx]
                tracker.crop_df[fid][step_dt]['ke'] = tracker.ke[sample_idx]

                # water balance components
                tracker.crop_df[fid][step_dt]['et_act'] = eta_act

                ppt = foo_day.precip[sample_idx]
                tracker.crop_df[fid][step_dt]['ppt'] = ppt

                melt = tracker.melt[sample_idx]
                tracker.crop_df[fid][step_dt]['melt'] = melt
                rain = tracker.rain[sample_idx]
                tracker.crop_df[fid][step_dt]['rain'] = rain

                runoff = tracker.sro[sample_idx]
                tracker.crop_df[fid][step_dt]['runoff'] = runoff
                dperc = tracker.dperc[sample_idx]
                tracker.crop_df[fid][step_dt]['dperc'] = dperc

                depl_root = tracker.depl_root[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_root'] = depl_root
                depl_root_prev = tracker.depl_root_prev[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_root_prev'] = depl_root_prev

                daw3 = tracker.daw3[sample_idx]
                tracker.crop_df[fid][step_dt]['daw3'] = daw3
                daw3_prev = tracker.daw3_prev[sample_idx]
                tracker.crop_df[fid][step_dt]['daw3_prev'] = daw3_prev
                delta_daw3 = daw3 - daw3_prev
                tracker.crop_df[fid][step_dt]['delta_daw3'] = delta_daw3

                soil_water = tracker.soil_water[sample_idx]
                tracker.crop_df[fid][step_dt]['soil_water'] = soil_water
                soil_water_prev = tracker.soil_water_prev[sample_idx]
                tracker.crop_df[fid][step_dt]['soil_water_prev'] = soil_water_prev
                delta_soil_water = tracker.delta_soil_water[sample_idx]
                tracker.crop_df[fid][step_dt]['delta_soil_water'] = delta_soil_water

                depl_ze = tracker.depl_ze[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_ze'] = depl_ze
                tracker.crop_df[fid][step_dt]['p_rz'] = tracker.p_rz[sample_idx]
                tracker.crop_df[fid][step_dt]['p_eft'] = tracker.p_eft[sample_idx]
                tracker.crop_df[fid][step_dt]['fc'] = tracker.fc[sample_idx]
                tracker.crop_df[fid][step_dt]['few'] = tracker.few[sample_idx]
                tracker.crop_df[fid][step_dt]['aw'] = tracker.aw[sample_idx]
                tracker.crop_df[fid][step_dt]['aw3'] = tracker.aw3[sample_idx]
                tracker.crop_df[fid][step_dt]['taw'] = tracker.taw[sample_idx]
                tracker.crop_df[fid][step_dt]['taw3'] = tracker.taw3[sample_idx]
                tracker.crop_df[fid][step_dt]['irrigation'] = tracker.irr_sim[sample_idx]
                tracker.crop_df[fid][step_dt]['irr_day'] = foo_day.irr_day[sample_idx]
                tracker.crop_df[fid][step_dt]['swe'] = tracker.swe[sample_idx]
                tracker.crop_df[fid][step_dt]['snow_fall'] = tracker.snow_fall[sample_idx]
                tracker.crop_df[fid][step_dt]['tavg'] = foo_day.temp_avg[sample_idx]
                tracker.crop_df[fid][step_dt]['tmax'] = foo_day.max_temp[sample_idx]
                tracker.crop_df[fid][step_dt]['zr'] = tracker.zr[sample_idx]
                tracker.crop_df[fid][step_dt]['kc_bas'] = tracker.kc_bas[sample_idx]
                tracker.crop_df[fid][step_dt]['niwr'] = tracker.niwr[sample_idx]
                tracker.crop_df[fid][step_dt]['et_bas'] = tracker.etc_bas
                tracker.crop_df[fid][step_dt]['season'] = tracker.in_season

                water_out = eta_act + dperc + runoff
                water_stored = soil_water - soil_water_prev
                water_in = melt + rain
                balance = water_in - water_stored - water_out

                tracker.crop_df[fid][step_dt]['wbal'] = balance

                if abs(balance) > 0.1 and foo_day.year > 2000:
                    pass
                    # raise WaterBalanceError('Check November water balance')

        else:
            if np.isnan(tracker.kc_act.any()):
                raise ValueError('NaN in Kc_act')

            if np.isnan(tracker.swe.any()):
                raise ValueError('NaN in SWE')

            etf[j, :] = tracker.kc_act
            swe[j, :] = tracker.swe

    if debug_flag:
        # pass final dataset to calling script
        tracker.crop_df = {fid: pd.DataFrame().from_dict(tracker.crop_df[fid], orient='index')[OUTPUT_FMT]
                           for fid in targets}  # turn nested dict to single dict of pd df
        for fid in tracker.crop_df:
            tracker.crop_df[fid].index = pd.to_datetime(tracker.crop_df[fid].index)  # turn indices into dt
            tracker.crop_df[fid].index = tracker.crop_df[fid].index.rename('date')  # rename index (should be after concat)
        out_ds_list = [tracker.crop_df[fid].to_xarray() for fid in targets]  # convert to xarray
        out_ds = xarray.concat(out_ds_list, pd.Index(tracker.crop_df.keys(), name=config.field_index))  # create out 1 ds w/ 2 dims

        if save_out:
            out_ds.to_netcdf(save_out, engine='netcdf4')
        return out_ds

    else:
        # if not debug, just return the actual ET and SWE results as ndarray
        return etf, swe


# This one does not save results to nc file
def field_day_loop_nc(config, plots, debug_flag=False, params=None):
    # This will need to be rewritten from the bottom up to mesh with the netcdf input.
    etf, swe = None, None
    size = len(plots.input[config.field_index].values)
    tracker = PlotTracker(size)
    tracker.load_soils_nc(plots)

    # apply calibration parameter updates here
    if config.calibration:
        # load PEST++ parameter proposition
        cal_arr = {k: np.zeros((1, size)) for k in config.calibration_groups}

        for k, f in config.calibration_files.items():

            group, fid = '_'.join(k.split('_')[:-1]), k.split('_')[-1]
            idx = plots.input['order'].index(fid)

            if params:
                value = params[k]
            else:
                v = pd.read_csv(f, index_col=None, header=0)
                value = v.loc[0, '1']

            cal_arr[group][0, idx] = value

        for k, v in cal_arr.items():

            tracker.__setattr__(k, v)

            if debug_flag:
                print('{}: {}'.format(k, ['{:.2f}'.format(p) for p in v.flatten()]))

    elif config.forecast:

        param_arr = {k: np.zeros((1, size)) for k in config.forecast_parameter_groups}

        for k, v in config.forecast_parameters.items():

            group, fid = '_'.join(k.split('_')[:-1]), k.split('_')[-1]

            # PEST++ has lower-cased the FIDs
            l = [x.lower() for x in plots.input['order']]
            idx = l.index(fid)

            if fid not in l:
                continue

            if params:
                value = params[k]
            else:
                value = v

            param_arr[group][0, idx] = value

        for k, v in param_arr.items():

            tracker.__setattr__(k, v)

            if debug_flag:
                print('{}: {}'.format(k, ['{:.2f}'.format(p) for p in v.flatten()]))

    else:
        for k, v in DEFAULTS.items():
            arr = np.ones((1, size)) * v
            tracker.__setattr__(k, arr)

    targets = plots.input[config.field_index].values

    # valid_data = {dt: val for dt, val in plots.input['time_series'].items() if
    #               (config.start_dt <= pd.to_datetime(dt) <= config.end_dt)}

    # Select appropriate timeseries columns and dates from the xarray dataset. - I don't think I need this.
    # print(valid_data)  # does not remove variables that don't have that coordinate... I guess that's fine for now?
    # Do I need a doy data variable in the input netcdf? - looks like it...
    # For now, it's been added in input.py SamplePlots.initialize_plot_data_nc

    # Initialize crop data frame
    time_range = pd.date_range(config.start_dt, config.end_dt, freq='D')
    if debug_flag:
        tracker.setup_dataframe(targets)  # creates empty dict of dict with FIDs as keys
    else:
        # Isn't this stuff essentially already done? It's in an xarray by FID and date...
        empty = np.zeros((len(time_range), len(targets))) * np.nan
        etf, swe = empty.copy(), empty.copy()

    tracker.set_kc_max()

    foo_day = DayData()
    foo_day.sdays = 0
    foo_day.doy_prev = 0
    foo_day.irr_status = None

    hr_ppt_keys = ['prcp_hr_{}'.format(str(i).rjust(2, '0')) for i in range(0, 24)]

    # looping through days
    # for j, (step_dt, vals) in enumerate(valid_data.items()):
    # other ways to do this?
    for j, step_dt in enumerate(time_range):  # slicing arrays instead of xarray
        # Is the datatype wrong now?
        vals = plots.input.sel(date=step_dt)  # all data for all fields for that date
        # Is it better to lump this with later sel statements? That seems unlikely?

        # Track variables for each day
        # For now, cast all values to native Python types
        foo_day.sdays += 1
        dt = pd.to_datetime(step_dt)
        foo_day.dt_string = dt.strftime('%Y-%m-%d')

        foo_day.year = dt.year
        foo_day.month = dt.month
        foo_day.day = dt.day
        foo_day.doy = dt.dayofyear

        foo_day.ndvi = np.zeros((1, size))
        foo_day.capture = np.zeros((1, size))
        foo_day.refet = np.zeros((1, size))
        foo_day.irr_day = np.zeros((1, size), dtype=int)

        # Check irrigation status on first date and first day of each year.
        if foo_day.doy == 1 or foo_day.irr_status is None:
            # establishing irrigated status
            foo_day.irr_status = np.array([plots.input['irr'].sel(year=dt.year).values])

            foo_day.irr_doys = []  # array of arrays listing all irrigated doys for each field
            # fetching doys for this year
            this_yr = slice('{}-01-01'.format(dt.year), '{}-12-31'.format(dt.year))
            # doys = plots.input['doy'].sel(date=this_yr).values
            doys = np.arange(1, year_len(dt.year)+1)
            # filling in

            for i, fid in enumerate(plots.input['FID'].values):
                # use irr_days as mask to select doys to add to irr_doys for each field
                irr_mask = plots.input['irr_days'].sel(FID=fid, date=this_yr).values
                foo_day.irr_doys.append(doys[irr_mask])

        # I think these are the select statements that are killing me. Can I fix them?
        for i, fid in enumerate(plots.input['FID'].values):  # for each field
            irrigated = foo_day.irr_status[0, i]
            if irrigated >= config.irr_threshold:  # use irr values
                foo_day.ndvi[0, i] = vals['ndvi_irr'].sel(FID=fid)
                foo_day.capture[0, i] = vals['etf_irr_ct'].sel(FID=fid)
                foo_day.refet[0, i] = vals['{}_mm_corrected'.format(config.refet_type)].sel(FID=fid)
                foo_day.irr_day[0, i] = int(foo_day.doy in foo_day.irr_doys[i])  # what is this doing?
            else:  # use inv_irr values
                foo_day.ndvi[0, i] = vals['ndvi_inv_irr'].sel(FID=fid)  # wait, how does this make a difference?
                foo_day.capture[0, i] = vals['etf_inv_irr_ct'].sel(FID=fid)
                foo_day.refet[0, i] = vals['{}_mm'.format(config.refet_type)].sel(FID=fid)
                foo_day.irr_day[0, i] = 0

        # These don't appear to be doing anything?
        foo_day.ndvi = foo_day.ndvi.reshape(1, -1)
        foo_day.capture = foo_day.capture.reshape(1, -1)
        foo_day.refet = foo_day.refet.reshape(1, -1)

        foo_day.min_temp = np.array(vals['tmin_c']).reshape(1, -1)
        foo_day.max_temp = np.array(vals['tmax_c']).reshape(1, -1)
        foo_day.temp_avg = (foo_day.min_temp + foo_day.max_temp) / 2.
        foo_day.srad = np.array(vals['srad_wm2']).reshape(1, -1)
        foo_day.precip = np.array(vals['prcp_mm'])

        if np.any(foo_day.precip > 0.):
            hr_ppt = np.array([vals[k] for k in hr_ppt_keys]).reshape(24, size)
            foo_day.hr_precip = hr_ppt

        foo_day.precip = foo_day.precip.reshape(1, -1)

        # Calculate height of vegetation.
        # Moved up to this point 12/26/07 for use in adj. Kcb and kc_max
        calculate_height.calculate_height(tracker)

        # Interpolate Kcb and make climate adjustment (for ETo basis)
        obs_kcb_daily.kcb_daily(config, plots, tracker, foo_day)  # It doesn't actually use the first two inputs?

        # Calculate Kcb, Ke, ETc
        compute_field_et.compute_field_et(config, plots, tracker, foo_day,
                                          debug_flag)

        # Retrieve values from foo_day and write to output data frame
        # Eventually let compute_crop_et() write directly to output df

        if debug_flag:
            for i, fid in enumerate(targets):
                tracker.crop_df[fid][step_dt] = {}
                sample_idx = 0, i
                tracker.crop_df[fid][step_dt]['etref'] = foo_day.refet[sample_idx]

                eta_act = tracker.etc_act[sample_idx]
                tracker.crop_df[fid][step_dt]['capture'] = foo_day.capture[sample_idx]
                tracker.crop_df[fid][step_dt]['t'] = tracker.t[sample_idx]
                tracker.crop_df[fid][step_dt]['e'] = tracker.e[sample_idx]
                tracker.crop_df[fid][step_dt]['kc_act'] = tracker.kc_act[sample_idx]
                tracker.crop_df[fid][step_dt]['ks'] = tracker.ks[sample_idx]
                tracker.crop_df[fid][step_dt]['ke'] = tracker.ke[sample_idx]

                # water balance components
                tracker.crop_df[fid][step_dt]['et_act'] = eta_act

                ppt = foo_day.precip[sample_idx]
                tracker.crop_df[fid][step_dt]['ppt'] = ppt

                melt = tracker.melt[sample_idx]
                tracker.crop_df[fid][step_dt]['melt'] = melt
                rain = tracker.rain[sample_idx]
                tracker.crop_df[fid][step_dt]['rain'] = rain

                runoff = tracker.sro[sample_idx]
                tracker.crop_df[fid][step_dt]['runoff'] = runoff
                dperc = tracker.dperc[sample_idx]
                tracker.crop_df[fid][step_dt]['dperc'] = dperc

                depl_root = tracker.depl_root[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_root'] = depl_root
                depl_root_prev = tracker.depl_root_prev[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_root_prev'] = depl_root_prev

                daw3 = tracker.daw3[sample_idx]
                tracker.crop_df[fid][step_dt]['daw3'] = daw3
                daw3_prev = tracker.daw3_prev[sample_idx]
                tracker.crop_df[fid][step_dt]['daw3_prev'] = daw3_prev
                delta_daw3 = daw3 - daw3_prev
                tracker.crop_df[fid][step_dt]['delta_daw3'] = delta_daw3

                soil_water = tracker.soil_water[sample_idx]
                tracker.crop_df[fid][step_dt]['soil_water'] = soil_water
                soil_water_prev = tracker.soil_water_prev[sample_idx]
                tracker.crop_df[fid][step_dt]['soil_water_prev'] = soil_water_prev
                delta_soil_water = tracker.delta_soil_water[sample_idx]
                tracker.crop_df[fid][step_dt]['delta_soil_water'] = delta_soil_water

                depl_ze = tracker.depl_ze[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_ze'] = depl_ze
                tracker.crop_df[fid][step_dt]['p_rz'] = tracker.p_rz[sample_idx]
                tracker.crop_df[fid][step_dt]['p_eft'] = tracker.p_eft[sample_idx]
                tracker.crop_df[fid][step_dt]['fc'] = tracker.fc[sample_idx]
                tracker.crop_df[fid][step_dt]['few'] = tracker.few[sample_idx]
                tracker.crop_df[fid][step_dt]['aw'] = tracker.aw[sample_idx]
                tracker.crop_df[fid][step_dt]['aw3'] = tracker.aw3[sample_idx]
                tracker.crop_df[fid][step_dt]['taw'] = tracker.taw[sample_idx]
                tracker.crop_df[fid][step_dt]['taw3'] = tracker.taw3[sample_idx]
                tracker.crop_df[fid][step_dt]['irrigation'] = tracker.irr_sim[sample_idx]
                tracker.crop_df[fid][step_dt]['irr_day'] = foo_day.irr_day[sample_idx]
                tracker.crop_df[fid][step_dt]['swe'] = tracker.swe[sample_idx]
                tracker.crop_df[fid][step_dt]['snow_fall'] = tracker.snow_fall[sample_idx]
                tracker.crop_df[fid][step_dt]['tavg'] = foo_day.temp_avg[sample_idx]
                tracker.crop_df[fid][step_dt]['tmax'] = foo_day.max_temp[sample_idx]
                tracker.crop_df[fid][step_dt]['zr'] = tracker.zr[sample_idx]
                tracker.crop_df[fid][step_dt]['kc_bas'] = tracker.kc_bas[sample_idx]
                tracker.crop_df[fid][step_dt]['niwr'] = tracker.niwr[sample_idx]
                tracker.crop_df[fid][step_dt]['et_bas'] = tracker.etc_bas
                tracker.crop_df[fid][step_dt]['season'] = tracker.in_season

                water_out = eta_act + dperc + runoff
                water_stored = soil_water - soil_water_prev
                water_in = melt + rain
                balance = water_in - water_stored - water_out

                tracker.crop_df[fid][step_dt]['wbal'] = balance

                if abs(balance) > 0.1 and foo_day.year > 2000:
                    pass
                    # raise WaterBalanceError('Check November water balance')

        else:
            if np.isnan(tracker.kc_act.any()):
                raise ValueError('NaN in Kc_act')

            if np.isnan(tracker.swe.any()):
                raise ValueError('NaN in SWE')

            etf[j, :] = tracker.kc_act
            swe[j, :] = tracker.swe

    if debug_flag:
        # pass final dataframe to calling script

        tracker.crop_df = {fid: pd.DataFrame().from_dict(tracker.crop_df[fid], orient='index')[OUTPUT_FMT]
                           for fid in targets}
        for fid in tracker.crop_df:
            tracker.crop_df[fid].index = pd.to_datetime(tracker.crop_df[fid].index)
        return tracker.crop_df

    else:
        # if not debug, just return the actual ET and SWE results as ndarray
        return etf, swe


def field_day_loop(config, plots, debug_flag=False, params=None):
    etf, swe = None, None
    size = len(plots.input['order'])
    tracker = PlotTracker(size)
    tracker.load_soils(plots)

    # apply calibration parameter updates here
    if config.calibration:
        # load PEST++ parameter proposition
        cal_arr = {k: np.zeros((1, size)) for k in config.calibration_groups}

        for k, f in config.calibration_files.items():

            group, fid = '_'.join(k.split('_')[:-1]), k.split('_')[-1]
            idx = plots.input['order'].index(fid)

            if params:
                value = params[k]
            else:
                v = pd.read_csv(f, index_col=None, header=0)
                value = v.loc[0, '1']

            cal_arr[group][0, idx] = value

        for k, v in cal_arr.items():

            tracker.__setattr__(k, v)

            if debug_flag:
                print('{}: {}'.format(k, ['{:.2f}'.format(p) for p in v.flatten()]))

    elif config.forecast:

        param_arr = {k: np.zeros((1, size)) for k in config.forecast_parameter_groups}

        for k, v in config.forecast_parameters.items():

            group, fid = '_'.join(k.split('_')[:-1]), k.split('_')[-1]

            # PEST++ has lower-cased the FIDs
            l = [x.lower() for x in plots.input['order']]
            idx = l.index(fid)

            if fid not in l:
                continue

            if params:
                value = params[k]
            else:
                value = v

            param_arr[group][0, idx] = value

        for k, v in param_arr.items():

            tracker.__setattr__(k, v)

            if debug_flag:
                print('{}: {}'.format(k, ['{:.2f}'.format(p) for p in v.flatten()]))

    else:
        for k, v in DEFAULTS.items():
            arr = np.ones((1, size)) * v
            tracker.__setattr__(k, arr)

    targets = plots.input['order']

    valid_data = {dt: val for dt, val in plots.input['time_series'].items() if
                  (config.start_dt <= pd.to_datetime(dt) <= config.end_dt)}

    # Initialize crop data frame
    if debug_flag:
        tracker.setup_dataframe(targets)
    else:
        time_range = pd.date_range(config.start_dt, config.end_dt, freq='D')
        empty = np.zeros((len(time_range), len(targets))) * np.nan
        etf, swe = empty.copy(), empty.copy()

    tracker.set_kc_max()

    foo_day = DayData()
    foo_day.sdays = 0
    foo_day.doy_prev = 0
    foo_day.irr_status = None

    hr_ppt_keys = ['prcp_hr_{}'.format(str(i).rjust(2, '0')) for i in range(0, 24)]

    for j, (step_dt, vals) in enumerate(valid_data.items()):

        # Track variables for each day
        # For now, cast all values to native Python types
        foo_day.sdays += 1
        foo_day.doy = vals['doy']
        foo_day.dt_string = step_dt
        dt = pd.to_datetime(step_dt)

        foo_day.year = dt.year
        foo_day.month = dt.month
        foo_day.day = dt.day

        foo_day.ndvi = np.zeros((1, size))
        foo_day.capture = np.zeros((1, size))
        foo_day.refet = np.zeros((1, size))
        foo_day.irr_day = np.zeros((1, size), dtype=int)

        if foo_day.doy == 1 or foo_day.irr_status is None:

            foo_day.irr_status = np.zeros((1, len(plots.input['order'])))
            foo_day.irr_doys = []

            for i, fid in enumerate(plots.input['order']):
                try:
                    irrigated = plots.input['irr_data'][fid][str(dt.year)]['irrigated']
                    foo_day.irr_doys.append(plots.input['irr_data'][fid][str(foo_day.year)]['irr_doys'])
                    foo_day.irr_status[0, i] = irrigated
                except KeyError:
                    foo_day.irr_status[0, i] = 0
                    foo_day.irr_doys.append([])

        for i, fid in enumerate(plots.input['order']):
            irrigated = foo_day.irr_status[0, i]
            if irrigated:
                foo_day.ndvi[0, i] = vals['ndvi_irr'][i]
                foo_day.capture[0, i] = vals['etf_irr_ct'][i]
                foo_day.refet[0, i] = vals['{}_mm'.format(config.refet_type)][i]
                foo_day.irr_day[0, i] = int(foo_day.doy in foo_day.irr_doys[i])

            else:
                foo_day.ndvi[0, i] = vals['ndvi_inv_irr'][i]
                foo_day.capture[0, i] = vals['etf_inv_irr_ct'][i]
                foo_day.refet[0, i] = vals['{}_mm_uncorr'.format(config.refet_type)][i]
                foo_day.irr_day[0, i] = 0

        foo_day.ndvi = foo_day.ndvi.reshape(1, -1)
        foo_day.capture = foo_day.capture.reshape(1, -1)
        foo_day.refet = foo_day.refet.reshape(1, -1)

        foo_day.min_temp = np.array(vals['tmin_c']).reshape(1, -1)
        foo_day.max_temp = np.array(vals['tmax_c']).reshape(1, -1)
        foo_day.temp_avg = (foo_day.min_temp + foo_day.max_temp) / 2.
        foo_day.srad = np.array(vals['srad_wm2']).reshape(1, -1)
        foo_day.precip = np.array(vals['prcp_mm'])

        if np.any(foo_day.precip > 0.):
            hr_ppt = np.array([vals[k] for k in hr_ppt_keys]).reshape(24, size)
            foo_day.hr_precip = hr_ppt

        foo_day.precip = foo_day.precip.reshape(1, -1)

        # Calculate height of vegetation.
        # Moved up to this point 12/26/07 for use in adj. Kcb and kc_max
        calculate_height.calculate_height(tracker)

        # Interpolate Kcb and make climate adjustment (for ETo basis)
        obs_kcb_daily.kcb_daily(config, plots, tracker, foo_day)

        # Calculate Kcb, Ke, ETc
        compute_field_et.compute_field_et(config, plots, tracker, foo_day,
                                          debug_flag)

        # Retrieve values from foo_day and write to output data frame
        # Eventually let compute_crop_et() write directly to output df

        if debug_flag:
            for i, fid in enumerate(targets):
                tracker.crop_df[fid][step_dt] = {}
                sample_idx = 0, i
                tracker.crop_df[fid][step_dt]['etref'] = foo_day.refet[sample_idx]

                eta_act = tracker.etc_act[sample_idx]
                tracker.crop_df[fid][step_dt]['capture'] = foo_day.capture[sample_idx]
                tracker.crop_df[fid][step_dt]['t'] = tracker.t[sample_idx]
                tracker.crop_df[fid][step_dt]['e'] = tracker.e[sample_idx]
                tracker.crop_df[fid][step_dt]['kc_act'] = tracker.kc_act[sample_idx]
                tracker.crop_df[fid][step_dt]['ks'] = tracker.ks[sample_idx]
                tracker.crop_df[fid][step_dt]['ke'] = tracker.ke[sample_idx]

                # water balance components
                tracker.crop_df[fid][step_dt]['et_act'] = eta_act

                ppt = foo_day.precip[sample_idx]
                tracker.crop_df[fid][step_dt]['ppt'] = ppt

                melt = tracker.melt[sample_idx]
                tracker.crop_df[fid][step_dt]['melt'] = melt
                rain = tracker.rain[sample_idx]
                tracker.crop_df[fid][step_dt]['rain'] = rain

                runoff = tracker.sro[sample_idx]
                tracker.crop_df[fid][step_dt]['runoff'] = runoff
                dperc = tracker.dperc[sample_idx]
                tracker.crop_df[fid][step_dt]['dperc'] = dperc

                depl_root = tracker.depl_root[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_root'] = depl_root
                depl_root_prev = tracker.depl_root_prev[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_root_prev'] = depl_root_prev

                daw3 = tracker.daw3[sample_idx]
                tracker.crop_df[fid][step_dt]['daw3'] = daw3
                daw3_prev = tracker.daw3_prev[sample_idx]
                tracker.crop_df[fid][step_dt]['daw3_prev'] = daw3_prev
                delta_daw3 = daw3 - daw3_prev
                tracker.crop_df[fid][step_dt]['delta_daw3'] = delta_daw3

                soil_water = tracker.soil_water[sample_idx]
                tracker.crop_df[fid][step_dt]['soil_water'] = soil_water
                soil_water_prev = tracker.soil_water_prev[sample_idx]
                tracker.crop_df[fid][step_dt]['soil_water_prev'] = soil_water_prev
                delta_soil_water = tracker.delta_soil_water[sample_idx]
                tracker.crop_df[fid][step_dt]['delta_soil_water'] = delta_soil_water

                depl_ze = tracker.depl_ze[sample_idx]
                tracker.crop_df[fid][step_dt]['depl_ze'] = depl_ze
                tracker.crop_df[fid][step_dt]['p_rz'] = tracker.p_rz[sample_idx]
                tracker.crop_df[fid][step_dt]['p_eft'] = tracker.p_eft[sample_idx]
                tracker.crop_df[fid][step_dt]['fc'] = tracker.fc[sample_idx]
                tracker.crop_df[fid][step_dt]['few'] = tracker.few[sample_idx]
                tracker.crop_df[fid][step_dt]['aw'] = tracker.aw[sample_idx]
                tracker.crop_df[fid][step_dt]['aw3'] = tracker.aw3[sample_idx]
                tracker.crop_df[fid][step_dt]['taw'] = tracker.taw[sample_idx]
                tracker.crop_df[fid][step_dt]['taw3'] = tracker.taw3[sample_idx]
                tracker.crop_df[fid][step_dt]['irrigation'] = tracker.irr_sim[sample_idx]
                tracker.crop_df[fid][step_dt]['irr_day'] = foo_day.irr_day[sample_idx]
                tracker.crop_df[fid][step_dt]['swe'] = tracker.swe[sample_idx]
                tracker.crop_df[fid][step_dt]['snow_fall'] = tracker.snow_fall[sample_idx]
                tracker.crop_df[fid][step_dt]['tavg'] = foo_day.temp_avg[sample_idx]
                tracker.crop_df[fid][step_dt]['tmax'] = foo_day.max_temp[sample_idx]
                tracker.crop_df[fid][step_dt]['zr'] = tracker.zr[sample_idx]
                tracker.crop_df[fid][step_dt]['kc_bas'] = tracker.kc_bas[sample_idx]
                tracker.crop_df[fid][step_dt]['niwr'] = tracker.niwr[sample_idx]
                tracker.crop_df[fid][step_dt]['et_bas'] = tracker.etc_bas
                tracker.crop_df[fid][step_dt]['season'] = tracker.in_season

                water_out = eta_act + dperc + runoff
                water_stored = soil_water - soil_water_prev
                water_in = melt + rain
                balance = water_in - water_stored - water_out

                tracker.crop_df[fid][step_dt]['wbal'] = balance

                if abs(balance) > 0.1 and foo_day.year > 2000:
                    pass
                    # raise WaterBalanceError('Check November water balance')

        else:
            if np.isnan(tracker.kc_act):
                raise ValueError('NaN in Kc_act')

            if np.isnan(tracker.swe):
                raise ValueError('NaN in SWE')

            etf[j, :] = tracker.kc_act
            swe[j, :] = tracker.swe

    if debug_flag:
        # pass final dataframe to calling script

        tracker.crop_df = {fid: pd.DataFrame().from_dict(tracker.crop_df[fid], orient='index')[OUTPUT_FMT]
                           for fid in targets}
        for fid in tracker.crop_df:
            tracker.crop_df[fid].index = pd.to_datetime(tracker.crop_df[fid].index)
        return tracker.crop_df

    else:
        # if not debug, just return the actual ET and SWE results as ndarray
        return etf, swe


if __name__ == '__main__':
    pass
