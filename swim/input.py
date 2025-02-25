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
        self.input = xarray.open_dataset(f)

    def input_to_dataframe(self, feature_id):

        # select statement?
        one_field = self.input.sel(FID=feature_id)
        df_ = one_field.to_dataframe()
        print(df_)
        df_.index = df_.index.droplevel(1)
        df_ = df_.groupby(df_.index).first()  # remove duplicate indices
        print(df_)
        print(df_.columns)
        df_ = df_.drop(columns=['awc', 'ksat', 'clay', 'sand', 'area_sq_m', 'irr_days', 'ea_kpa'])
        # problem with maintaining the years dimension on variables that are not using it.

        # idx = self.input['order'].index(feature_id)
        #
        # ts = self.input['time_series']
        # dct = {k: [] for k in ts[list(ts.keys())[0]]}
        # dates = []
        #
        # for dt in ts:
        #     doy_data = ts[dt]
        #     dates.append(dt)
        #     for k, v in doy_data.items():
        #         if k == 'doy':
        #             dct['doy'].append(v)
        #         else:
        #             # all other values are lists
        #             dct[k].append(v[idx])
        #
        # df_ = pd.DataFrame().from_dict(dct)
        # df_.index = pd.DatetimeIndex(dates)
        return df_


if __name__ == '__main__':
    pass
