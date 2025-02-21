import json

import numpy as np
import pandas as pd
import xarray


class SamplePlots:
    """A Container for input and output time series, historical, and static field information

    This should include some initial estimate of soil properties and historical
    estimate of irrigated status and crop type.

    """

    def __init__(self):
        super().__init__()
        self.input = None
        self.output = None

    def initialize_plot_data(self, config):
        f = config.input_data
        with open(f, 'r') as fp:
            self.input = json.load(fp)

    def initialize_plot_data_nc(self, config):
        f = config.input_data
        ds = xarray.open_dataset(f)

        # TODO: Add doy variable to input netcdf? for now, I'll add it here
        doys = [pd.to_datetime(date).dayofyear for date in ds['date'].values]
        ds['doy'] = xarray.Variable('date', doys, {'long_name': 'integer day of year'})

        # TODO: rename long gridmet variables in netcdf, retain long name in attributes.
        renaming = {'daily_mean_reference_evapotranspiration_alfalfa': 'etr_mm',
                    'daily_mean_reference_evapotranspiration_grass': 'eto_mm',
                    'precipitation_amount': 'prcp_mm',
                    'daily_mean_shortwave_radiation_at_surface': 'srad_wm2',
                    'daily_maximum_temperature': 'tmax_c',
                    'daily_minimum_temperature': 'tmin_c',
                    'daily_mean_wind_speed': 'u2_ms',
                    'daily_mean_specific_humidity': 'q'}
        ds = ds.rename(renaming)

        self.input = ds

    def input_to_dataframe(self, feature_id):

        idx = self.input['order'].index(feature_id)

        ts = self.input['time_series']
        dct = {k: [] for k in ts[list(ts.keys())[0]]}
        dates = []

        for dt in ts:
            doy_data = ts[dt]
            dates.append(dt)
            for k, v in doy_data.items():
                if k == 'doy':
                    dct['doy'].append(v)
                else:
                    # all other values are lists
                    dct[k].append(v[idx])

        df_ = pd.DataFrame().from_dict(dct)
        df_.index = pd.DatetimeIndex(dates)
        return df_


if __name__ == '__main__':
    pass
