""" Compilation of content from landsat_sensing.py, field_properties.py, and field_timeseries.py.

Prevents the need to run each file separately and makes things easier to keep track of when preparing input data.
"""

import os

from prep import info
from prep.landsat_sensing import landsat_time_series_multipolygon, join_remote_sensing, detect_cuttings
from prep.field_properties import write_field_properties
from prep.field_timeseries import find_gridmet_points, download_gridmet, join_daily_timeseries

if __name__ == '__main__':
    # Declaring common variables
    project = info.project_name
    project_ws = info.d
    fields_shp = info.fields_shp

    # --------------------
    # From landsat_sensing
    dtype = 'extracts'
    tables = os.path.join(project_ws, 'met_timeseries', 'landsat', 'tables')
    if not os.path.exists(tables):
        os.mkdir(tables)

    types_ = ['inv_irr', 'irr']
    sensing_params = ['ndvi', 'etf']

    for mask_type in types_:
        for sensing_param in sensing_params:
            yrs = [x for x in range(1987, 2024)]
            ee_data = os.path.join(project_ws, 'met_timeseries', 'landsat', dtype, sensing_param, mask_type)
            src = os.path.join(tables, '{}_{}_{}.csv'.format(project, sensing_param, mask_type))
            src_ct = os.path.join(tables, '{}_{}_{}_ct.csv'.format(project, sensing_param, mask_type))
            landsat_time_series_multipolygon(fields_shp, ee_data, yrs, src, src_ct)
    all_landsat = os.path.join(project_ws, 'met_timeseries', 'landsat', '{}_sensing.csv'.format(project))
    join_remote_sensing(tables, all_landsat)
    irr_ = os.path.join(project_ws, 'properties', '{}_irr.csv'.format(project))
    js_ = os.path.join(project_ws, 'met_timeseries', 'landsat', '{}_cuttings.json'.format(project))
    detect_cuttings(all_landsat, irr_, irr_threshold=0.1, out_json=js_)

    # ---------------------
    # From field_properties
    irr_ = os.path.join(project_ws, 'properties', '{}_irr.csv'.format(project))
    cdl_ = os.path.join(project_ws, 'properties', '{}_cdl.csv'.format(project))
    _ssurgo = os.path.join(project_ws, 'properties', '{}_ssurgo.csv'.format(project))
    _landfire = os.path.join(project_ws, 'properties', '{}_landfire.csv'.format(project))  # apprently can't get rid of?
    jsn = os.path.join(project_ws, 'properties', '{}_props.json'.format(project))

    write_field_properties(fields_shp, irr_, cdl_, _ssurgo, _landfire, jsn, index_col='FID')

    # ---------------------
    # From field_timeseries
    gridmet = info.gridmet_dir
    rasters_ = os.path.join(gridmet, 'correction_surfaces_aea')
    grimet_cent = os.path.join(gridmet, 'gridmet_centroids_MT.shp')

    fields_gridmet = os.path.join(project_ws, 'gis', '{}_fields_gfid.shp'.format(project))
    gridmet_factors = os.path.join(project_ws, 'gis', '{}_fields_gfid.json'.format(project))
    met = os.path.join(project_ws, 'met_timeseries')

    # gridmet downloads
    find_gridmet_points(fields_shp, grimet_cent, rasters_, fields_gridmet, gridmet_factors)
    download_gridmet(fields_gridmet, gridmet_factors, met, start='1987-01-01', end='2023-12-31', overwite=False)

    # SWE unavailable
    snow_ts = 0

    dst_dir_ = os.path.join(project_ws, 'input_timeseries')
    if not os.path.exists(dst_dir_):
        os.makedirs(dst_dir_)

    params = ['etf_inv_irr',
              'ndvi_inv_irr',
              'etf_irr',
              'ndvi_irr']
    params += ['{}_ct'.format(p) for p in params]

    join_daily_timeseries(fields_gridmet, met, all_landsat, snow_ts, dst_dir_, overwrite=True,
                          start_date='2000-01-01', end_date='2020-12-31', **{'params': params})

# ========================= EOF ================================================================================
