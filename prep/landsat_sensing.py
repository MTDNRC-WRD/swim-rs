import os
import json

import numpy as np
import pandas as pd
import xarray
from tqdm import tqdm
import geopandas as gpd

from rasterstats import zonal_stats


def landsat_time_series_image(in_shp, tif_dir, years, out_csv, out_csv_ct, min_ct=100, feature_id='FID'):
    """
    Intended to process raw tif to tabular data using zonal statistics on polygons.
    See e.g., 'ndvi_export.export_ndvi() to export such images from Earth Engine. The output of this function
    should be the same format and structure as that from landsat_time_series_station() and
    landsat_time_series_multipolygon(). Ensure the .tif and
    .shp are both in the same coordinate reference system.
    :param feature_id:
    :param in_shp:
    :param tif_dir:
    :param years:
    :param out_csv:
    :param out_csv_ct:
    :param min_ct:
    :return:
    """
    gdf = gpd.read_file(in_shp)
    gdf.index = gdf[feature_id]

    adf, ctdf, first = None, None, True

    for yr in years:

        file_list, dts = get_tif_list(tif_dir, yr)

        dt_index = pd.date_range('{}-01-01'.format(yr), '{}-12-31'.format(yr), freq='D')
        df = pd.DataFrame(index=dt_index, columns=gdf.index)
        ct = pd.DataFrame(index=dt_index, columns=gdf.index)

        print('\n', yr, len(file_list))
        for dt, f in tqdm(zip(dts, file_list), total=len(file_list)):
            stats = zonal_stats(in_shp, f, stats=['mean', 'count'], nodata=0.0, categorical=False, all_touched=False)
            stats = [x['mean'] if isinstance(x['mean'], float) and x['count'] > min_ct else np.nan for x in stats]
            df.loc[dt, :] = stats
            ct.loc[dt, :] = ~pd.isna(stats)
            df.loc[dt, :] /= 1000

        df = df.astype(float).interpolate()
        df = df.interpolate(method='bfill')

        ct = ct.fillna(0)
        ct = ct.astype(int)

        if first:
            adf = df.copy()
            ctdf = ct.copy()
            first = False
        else:
            adf = pd.concat([adf, df], axis=0, ignore_index=False, sort=True)
            ctdf = pd.concat([ctdf, ct], axis=0, ignore_index=False, sort=True)

    adf.to_csv(out_csv)
    ctdf.to_csv(out_csv_ct)


def sparse_landsat_time_series(in_shp, csv_dir, years, out_csv, out_csv_ct, feature_id='FID',
                               select=None):
    """
    Intended to process Earth Engine extracts of buffered point data, e.g., the area around flux tower
    stations. See e.g., ndvi_export.flux_tower_ndvi() to generate such data. The output of this function
    should be the same format and structure as that from landsat_time_series_image()
    and landsat_time_series_multipolygon().
    :param feature_id:
    :param in_shp:
    :param csv_dir:
    :param years:
    :param out_csv:
    :param out_csv_ct:
    :return:
    """
    gdf = gpd.read_file(in_shp)
    gdf.index = gdf[feature_id]

    if select:
        gdf = gdf.loc[select]

    print(csv_dir)

    adf, ctdf, first = None, None, True

    for yr in years:

        dt_index = pd.date_range('{}-01-01'.format(yr), '{}-12-31'.format(yr), freq='D')

        df = pd.DataFrame(index=dt_index, columns=gdf.index)
        ct = pd.DataFrame(index=dt_index, columns=gdf.index)

        file_list = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir) if
                     x.endswith('.csv') and '_{}'.format(yr) in x]

        for f in file_list:
            field = pd.read_csv(f)
            sid = field.columns[0]

            if select and sid not in select:
                continue

            cols = [c for c in field.columns if len(c.split('_')) == 3]
            f_idx = [c.split('_')[-1] for c in cols]
            f_idx = [pd.to_datetime(i) for i in f_idx]
            field = pd.DataFrame(columns=[sid], data=field[cols].values.T, index=f_idx)
            duplicates = field[field.index.duplicated(keep=False)]
            if not duplicates.empty:
                field = field.resample('D').mean()
            field = field.sort_index()

            if 'etf' in csv_dir:
                field[field[sid] < 0.01] = np.nan

            df.loc[field.index, sid] = field[sid]

            ct.loc[f_idx, sid] = ~pd.isna(field[sid])

        df = df.astype(float).interpolate()
        df = df.bfill()

        ct = ct.fillna(0)
        ct = ct.astype(int)

        if first:
            adf = df.copy()
            ctdf = ct.copy()
            first = False
        else:
            adf = pd.concat([adf, df], axis=0, ignore_index=False, sort=True)
            ctdf = pd.concat([ctdf, ct], axis=0, ignore_index=False, sort=True)
        print(yr)

    adf.to_csv(out_csv)
    ctdf.to_csv(out_csv_ct)


def clustered_landsat_time_series(in_shp, csv_dir, years, out_csv, out_csv_ct, feature_id='FID'):
    """
    Intended to process Earth Engine extracts of buffered point data, e.g., the area around flux tower
    stations. See e.g., ndvi_export.clustered_field_ndvi() to generate such data. The output of this function
    should be the same format and structure as that from landsat_time_series_image() and
    landsat_time_series_station().
    :param feature_id:
    :param in_shp:
    :param csv_dir:
    :param years:
    :param out_csv:
    :param out_csv_ct:
    :return:
    """
    gdf = gpd.read_file(in_shp)
    gdf.index = gdf[feature_id]

    print(csv_dir)

    adf, ctdf, first = None, None, True

    for yr in tqdm(years, desc='Processing Landsat Time Series', total=len(years)):

        # if yr != 2007:
        #     continue

        dt_index = pd.date_range('{}-01-01'.format(yr), '{}-12-31'.format(yr), freq='D')

        try:
            f = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir) if
                 x.endswith('.csv') and '_{}'.format(yr) in x][0]
        except IndexError as e:
            print(e, yr)
            continue

        field = pd.read_csv(f)
        field.index = field[feature_id]
        cols = [c for c in field.columns if len(c.split('_')) == 3]
        f_idx = [c.split('_')[-1] for c in cols]
        f_idx = [pd.to_datetime(i) for i in f_idx]
        field = pd.DataFrame(columns=field.index, data=field[cols].values.T, index=f_idx)
        duplicates = field[field.index.duplicated(keep=False)]
        if not duplicates.empty:
            field = field.resample('D').max()  # Take the max value if two images show field.
        field = field.sort_index()

        # field = field[['043_000128']]

        field[field.values == 0.00] = np.nan

        # for both NDVI and ETf, values in agriculture and the vegetated land surface generally,
        # should not go below about 0.01
        # captures of these low values are likely small pixel samples on SLC OFF Landsat 7 or
        # on bad polygons that include water or some other land cover we don't want to use
        # see e.g., https://code.earthengine.google.com/5ea8bc8c6134845a8c0c81a4cdb99fc0
        # TODO: examine these thresholds, prob better to extract pixel count to filter data

        diff_back = field.diff().values
        field = pd.DataFrame(index=field.index, columns=field.columns,
                             data=np.where(diff_back < -0.1, np.nan, field.values))

        diff_for = field.shift(periods=2).diff()
        diff_for = diff_for.shift(periods=-3).values
        field = pd.DataFrame(index=field.index, columns=field.columns,
                             data=np.where(diff_for > 0.1, np.nan, field.values))

        if 'etf' in csv_dir:
            field[field.values < 0.2] = np.nan

        if 'ndvi' in csv_dir:
            field[field.values < 0.2] = np.nan

        df = field.copy()
        df = df.astype(float).interpolate()
        df = df.reindex(dt_index)

        ct = ~pd.isna(df)

        df = df.interpolate().bfill()
        df = df.interpolate().ffill()

        ct = ct.reindex(dt_index)
        ct = ct.fillna(0)
        ct = ct.astype(int)

        if first:
            adf = df.copy()
            ctdf = ct.copy()
            first = False
        else:
            adf = pd.concat([adf, df], axis=0, ignore_index=False, sort=True)
            ctdf = pd.concat([ctdf, ct], axis=0, ignore_index=False, sort=True)

    adf.to_csv(out_csv)
    ctdf.to_csv(out_csv_ct)


def clustered_landsat_time_series_nc(image_df, start_yr=2000, end_yr=2024, feature_id='FID', var_name=None):
    """
    Intended to process Earth Engine extracts of clustered field data. See e.g., ndvi_export.clustered_field_ndvi()
    to generate such data. The output of this function should be the same format and structure as that from
    landsat_time_series_image() and landsat_time_series_station().
    """
    # adf, ctdf, first = None, None, True

    dt_index = pd.date_range('{}-01-01'.format(start_yr), '{}-12-31'.format(end_yr), freq='D')

    # try:
    #     f = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir) if
    #          x.endswith('.csv') and '_{}'.format(yr) in x][0]
    # except IndexError as e:
    #     print(e, yr)
    #     continue

    # field = pd.read_csv(f)
    field = image_df
    # field.index = field[feature_id]
    cols = [c for c in field.columns if len(c.split('_')) == 3]
    f_idx = [c.split('_')[-1] for c in cols]
    f_idx = [pd.to_datetime(i) for i in f_idx]
    field = pd.DataFrame(columns=field.index, data=field[cols].values.T, index=f_idx)
    duplicates = field[field.index.duplicated(keep=False)]
    if not duplicates.empty:
        field = field.resample('D').max()
    field = field.sort_index()

    # field = field[['043_000128']]

    field[field.values == 0.00] = np.nan

    # for both NDVI and ETf, values in agriculture and the vegetated land surface generally,
    # should not go below about 0.01
    # captures of these low values are likely small pixel samples on SLC OFF Landsat 7 or
    # on bad polygons that include water or some other land cover we don't want to use
    # see e.g., https://code.earthengine.google.com/5ea8bc8c6134845a8c0c81a4cdb99fc0
    # TODO: examine these thresholds, prob better to extract pixel count to filter data

    diff_back = field.diff().values
    field = pd.DataFrame(index=field.index, columns=field.columns,
                         data=np.where(diff_back < -0.1, np.nan, field.values))

    diff_for = field.shift(periods=2).diff()
    diff_for = diff_for.shift(periods=-3).values
    field = pd.DataFrame(index=field.index, columns=field.columns,
                         data=np.where(diff_for > 0.1, np.nan, field.values))

    # # Allows for different thresholds for different variables.
    # if 'etf' in csv_dir:
    #     field[field.values < 0.2] = np.nan
    #
    # if 'ndvi' in csv_dir:
    #     field[field.values < 0.2] = np.nan

    field[field.values < 0.2] = np.nan

    ct = ~pd.isna(field)
    # print()
    # print(ct, ct.sum())
    # print()

    df = field.copy()
    df = df.astype(float).interpolate()
    df = df.reindex(dt_index)

    # ct = ~pd.isna(df)
    # print()
    # print(ct, ct.sum())
    # print()

    df = df.interpolate().bfill()
    df = df.interpolate().ffill()

    ct = ct.reindex(dt_index)
    ct = ct.fillna(0)
    # print(ct.max())
    # ct = ct.astype(int)
    ct = ct.astype(bool)

    adf = df.copy()
    ctdf = ct.copy()

    # print(adf)
    # format dfs
    adf = adf.melt(
        var_name="FID",
        value_name=var_name,
        ignore_index=False
    )
    adf.index = adf.index.set_names('date')
    adf = adf.set_index('FID', append=True)

    ctdf = ctdf.melt(
        var_name="FID",
        value_name="{}_ct".format(var_name),
        ignore_index=False
    )
    ctdf.index = ctdf.index.set_names('date')
    ctdf = ctdf.set_index('FID', append=True)

    # print()
    # print(adf)
    # print(ctdf)

    axr = adf.to_xarray()
    ctxr = ctdf.to_xarray()

    return axr, ctxr


def join_remote_sensing(_dir, dst):
    l = [os.path.join(_dir, f) for f in os.listdir(_dir) if f.endswith('.csv')]
    first = True

    params = ['etf_irr',
              'etf_inv_irr',
              'ndvi_inv_irr',
              'ndvi_irr']

    params += ['{}_ct'.format(p) for p in params]

    for f in l:
        param = [p for p in params if p in os.path.basename(f)]
        if len(param) > 1:
            param = param[1]
        else:
            param = param[0]

        if first:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            cols = ['{}_{}'.format(c, param) for c in df.columns]
            df.columns = cols
            first = False
            print(param)
        else:
            csv = pd.read_csv(f, index_col=0, parse_dates=True)
            cols = ['{}_{}'.format(c, param) for c in csv.columns]
            csv.columns = cols
            df = pd.concat([csv, df], axis=1)
            print(param)

    df.to_csv(dst)


def get_tif_list(tif_dir, year):
    """ Pass list in place of tif_dir optionally """
    l = [os.path.join(tif_dir, x) for x in os.listdir(tif_dir) if
         x.endswith('.tif') and '_{}'.format(year) in x]
    dt_str = [f[-12:-4] for f in l]
    dates_ = [pd.to_datetime(d, format='%Y%m%d') for d in dt_str]
    tup_ = sorted([(f, d) for f, d in zip(l, dates_)], key=lambda x: x[1])
    l, dates_ = [t[0] for t in tup_], [t[1] for t in tup_]
    return l, dates_


def detect_cuttings(landsat, irr_csv, out_json, irr_threshold=0.1, select=None):
    lst = pd.read_csv(landsat, index_col=0, parse_dates=True)
    years = list(set([i.year for i in lst.index]))
    irr = pd.read_csv(irr_csv, index_col=0)
    irr.drop(columns=['LAT', 'LON'], inplace=True)

    try:
        _ = float(irr.index[0])
        irr.index = [str(i) for i in irr.index]
    except (TypeError, ValueError):
        pass

    irrigated, fields = False, {}
    for c in tqdm(irr.index, desc='Analyzing Irrigation', total=len(irr.index)):

        if select and c not in select:
            continue

        if c not in irr.index:
            print('{} not in index'.format(c))
            continue

        if np.all(np.isnan(irr.loc[c])):
            print('{} is all nan in {}'.format(c, irr_csv))
            continue

        fields[c] = {}

        selector = '{}_ndvi_irr'.format(c)

        fallow = []

        for yr in years:

            irr_doys, periods = [], 0

            try:
                f_irr = irr.at[c, 'irr_{}'.format(yr)]
            except (ValueError, KeyError):
                f_irr = irr.at[c, 'irr_{}'.format(yr)]

            irrigated = f_irr > irr_threshold

            if not irrigated:
                fallow.append(yr)
                fields[c][yr] = {'irr_doys': irr_doys,
                                 'irrigated': int(irrigated),
                                 'f_irr': f_irr}
                continue

            df = lst.loc['{}-01-01'.format(yr): '{}-12-31'.format(yr), [selector]]
            df['doy'] = [i.dayofyear for i in df.index]
            df[selector] = df[selector].rolling(window=10, center=True).mean()
            # df[selector] = savgol_filter(df[selector], window_length=7, polyorder=2)

            df['diff'] = df[selector].diff()

            nan_ct = np.count_nonzero(np.isnan(df.values))
            if nan_ct > 200:
                print('{}: {} has {}/{} nan'.format(c, yr, nan_ct, df.values.size))
                fallow.append(yr)
                continue

            local_min_indices = df[(df['diff'] > 0) & (df['diff'].shift(1) < 0)].index

            # Find periods with positive slope for more than 10 days
            positive_slope = (df['diff'] > 0)
            groups = (positive_slope != positive_slope.shift()).cumsum()
            df['groups'] = groups
            group_counts = positive_slope.groupby(groups).sum()
            long_positive_slope_groups = group_counts[group_counts >= 10].index

            for group in long_positive_slope_groups:
                # Find the start and end indices of the group
                group_indices = positive_slope[groups == group].index
                start_index = group_indices[0]
                end_index = group_indices[-1]

                # Check if this group follows a local minimum
                if start_index in local_min_indices:
                    start_doy = (start_index - pd.Timedelta(days=5)).dayofyear
                    end_doy = (end_index - pd.Timedelta(days=5)).dayofyear
                    irr_doys.extend(range(start_doy, end_doy + 1))
                    periods += 1

                else:
                    # Otherwise, just add the days within the group
                    start_doy = start_index.dayofyear
                    end_doy = (end_index - pd.Timedelta(days=5)).dayofyear
                    irr_doys.extend(range(start_doy, end_doy + 1))
                    periods += 1

            irr_doys = sorted(list(set(irr_doys)))

            fields[c][yr] = {'irr_doys': irr_doys,
                             'irrigated': int(irrigated),
                             'f_irr': f_irr}

        fields[c]['fallow_years'] = fallow

    with open(out_json, 'w') as fp:
        json.dump(fields, fp, indent=4)
    print('wrote {}'.format(out_json))


def detect_cuttings_nc(landsat, irr_csv, irr_threshold=0.1, select=None):
    # lst = xarray.open_dataset(landsat)  # To pull out ndvi data
    lst = landsat  # To pull out ndvi data
    years = list(set([i.year for i in pd.to_datetime(lst.date.values)]))  # In case of dropped years?
    irr = xarray.open_dataset(irr_csv)  # to extract irrmapper data

    # Create a dataframe that will be transitioned into an xarray/netcdf variable at the end.
    # Does it need 3 values? nans for no irrigation that year, then ones and zeros filling each irrigated year?
    mi = pd.MultiIndex.from_product([lst.date.values, lst.FID.values], names=["date", "FID"])
    irr_days = pd.DataFrame(np.zeros(len(mi), dtype='bool'), index=mi, columns=['irr_days'])
    # print()
    # print(irr_days.sum())
    # print(irr_days)

    for field in lst.FID.values:
        for yr in years:
            # Why do I need to loop over years? Can I do the whole time series at once?
            irrigated = irr.irr.sel(FID=field, year=yr) > irr_threshold
            if not irrigated:  # keep everything as zeros
                continue
            # otherwise, calculate irrigation days.
            irr_doys = []

            df = lst.ndvi_irr.loc['{}-01-01'.format(yr): '{}-12-31'.format(yr), field].to_dataframe()
            df['doy'] = [i.dayofyear for i in df.index]
            df['ndvi_irr'] = df['ndvi_irr'].rolling(window=10, center=True).mean()
            df['diff'] = df['ndvi_irr'].diff()

            nan_ct = np.count_nonzero(np.isnan(df.values))
            if nan_ct > 200:
                print('{}: {} has {}/{} nan'.format(field, yr, nan_ct, df.values.size))
                continue

            local_min_indices = df[(df['diff'] > 0) & (df['diff'].shift(1) < 0)].index

            # Find periods with positive slope for more than 10 days
            positive_slope = (df['diff'] > 0)
            groups = (positive_slope != positive_slope.shift()).cumsum()
            df['groups'] = groups
            group_counts = positive_slope.groupby(groups).sum()
            long_positive_slope_groups = group_counts[group_counts >= 10].index

            for group in long_positive_slope_groups:
                # Find the start and end indices of the group
                group_indices = positive_slope[groups == group].index
                start_index = group_indices[0]
                end_index = group_indices[-1]

                # Push the start back if this group follows a local minimum
                # (i.e. assume irrigation starts before seeing ndvi response)
                # also assume that irrigation stops before the end of the upward trend
                if start_index in local_min_indices:
                    start_day = start_index - pd.Timedelta(days=5)
                    end_day = end_index - pd.Timedelta(days=5)
                    irr_doys.extend(pd.date_range(start_day, end_day))

                else:  # Why would it not follow a local minimum?
                    # Otherwise, just add the days within the group
                    end_day = end_index - pd.Timedelta(days=5)
                    irr_doys.extend(pd.date_range(start_index, end_day))

            irr_doys = sorted(list(set(irr_doys)))
            indices = [(i, field) for i in irr_doys]
            # print()
            # print(len(irr_doys), irr_doys)
            # print(len(indices), indices)
            # print(irr_days.index.isin(indices))

            # turn irr_doys into True values in irr_days
            irr_days['irr_days'] = irr_days['irr_days'].where(~irr_days.index.isin(indices), True)

    # print()
    # print(irr_days.sum())
    # print(irr_days)

    irr_days = irr_days.to_xarray()  # melt index when building df, not at the end?

    # print()
    # print(irr_days)

    return irr_days


if __name__ == '__main__':

    root = '/home/dgketchum/PycharmProjects/swim-rs'

    # data = os.path.join(root, 'tutorials', '2_Fort_Peck', 'data')
    data = os.path.join(root, 'tutorials', '3_Crane', 'data')
    shapefile_path = os.path.join(data, 'gis', 'flux_fields.shp')

    # input properties files
    irr = os.path.join(data, 'properties', 'calibration_irr.csv')
    ssurgo = os.path.join(data, 'properties', 'calibration_ssurgo.csv')

    landsat = os.path.join(data, 'landsat')
    extracts = os.path.join(landsat, 'extracts')
    tables = os.path.join(landsat, 'tables')

    remote_sensing_file = os.path.join(landsat, 'remote_sensing.csv')

    FEATURE_ID = 'field_1'
    selected_feature = 'S2'

    # types_ = ['inv_irr', 'irr']

    types_ = ['irr']
    sensing_params = ['ndvi', 'etf']

    for mask_type in types_:

        for sensing_param in sensing_params:
            yrs = [x for x in range(1987, 2023)]

            ee_data = os.path.join(landsat, 'extracts', sensing_param, mask_type)
            src = os.path.join(tables, '{}_{}_{}.csv'.format('calibration', sensing_param, mask_type))
            src_ct = os.path.join(tables, '{}_{}_{}_ct.csv'.format('calibration', sensing_param, mask_type))

            # TODO: consider whether there is a case where ETf needs to be interpolated
            sparse_landsat_time_series(shapefile_path, ee_data, yrs, src, src_ct,
                                       feature_id=FEATURE_ID, select=[selected_feature])

    cuttings_json = os.path.join(landsat, 'calibration_cuttings.json')
    detect_cuttings(remote_sensing_file, irr, irr_threshold=0.1, out_json=cuttings_json, select=[selected_feature])

# ========================= EOF ================================================================================
