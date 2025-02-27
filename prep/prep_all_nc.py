# # 2/26/2025
# # Prep input netcdf file for ~2000 fields in the Upper Yellowstone
# templated off of uy10_run.py

import os
import time
import geopandas as gpd
import matplotlib.pyplot as plt
import sys
import ee
import xarray
import xvec  # this is used, just tacked on to xarray stuff. Needs to be imported.
import pandas as pd
from datetime import timedelta
import pytz
import pynldas2 as nld
import numpy as np
from chmdata.thredds import GridMet, BBox

from data_extraction.ee.etf_export import clustered_sample_etf_direct_1
from data_extraction.ee.ndvi_export import clustered_sample_ndvi_direct_1
from data_extraction.ee.ee_utils import is_authorized
from data_extraction.gridmet.gridmet import air_pressure, actual_vapor_pressure
from data_extraction.ee.ee_props import get_irrigation_direct_nc, get_ssurgo_direct_nc
from prep.landsat_sensing import clustered_landsat_time_series_nc, detect_cuttings_nc

import psutil


def get_process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


def step_1():
    # Display the first few rows of the GeoDataFrame to examine structure and attributes
    gdf.head()
    print(gdf.shape[0], 'fields')
    print()

    # Plot the shapefile geometries
    gdf.plot(figsize=(10, 10), edgecolor='black')
    plt.title('Shapefile Geometry')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

    # Display the EPSG code
    epsg_code = gdf.crs
    print(f"EPSG Code: {epsg_code}")
    print()

    # List attribute fields
    attributes = gdf.columns
    print('Attributes in shapefile:')
    for attribute in attributes:
        print(attribute)
    print()

    # # Create directories, if they do not already exist
    # data_dir = os.path.join(root, 'examples', 'uy10', 'data')
    # dirs = ['snodas',
    #         'properties',
    #         'landsat',
    #         'bias_correction_tif',
    #         'gis',
    #         'met_timeseries',
    #         'input_timeseries']
    #
    # dir_paths = [os.path.join(data_dir, d) for d in dirs]
    # [os.makedirs(d, exist_ok=True) for d in dir_paths]


# Define Constants and Remote Sensing Data Paths
# TODO: remove hard-coded collections and use variables defined here
IRR = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'
ETF = 'projects/usgs-gee-nhm-ssebop/assets/ssebop/landsat/c02'
# We must specify which column in the shapefile represents the field's unique ID, in this case it is 'FID'
FEATURE_ID = 'FID'

# # Step 3
CLIMATE_COLS = {
    'etr': {
        'nc': 'agg_met_etr_1979_CurrentYear_CONUS',
        'var': 'daily_mean_reference_evapotranspiration_alfalfa',
        'col': 'etr_mm'},
    'pet': {
        'nc': 'agg_met_pet_1979_CurrentYear_CONUS',
        'var': 'daily_mean_reference_evapotranspiration_grass',
        'col': 'eto_mm'},
    'pr': {
        'nc': 'agg_met_pr_1979_CurrentYear_CONUS',
        'var': 'precipitation_amount',
        'col': 'prcp_mm'},
    'srad': {
        'nc': 'agg_met_srad_1979_CurrentYear_CONUS',
        'var': 'daily_mean_shortwave_radiation_at_surface',
        'col': 'srad_wm2'},
    'tmmx': {
        'nc': 'agg_met_tmmx_1979_CurrentYear_CONUS',
        'var': 'daily_maximum_temperature',
        'col': 'tmax_k'},
    'tmmn': {
        'nc': 'agg_met_tmmn_1979_CurrentYear_CONUS',
        'var': 'daily_minimum_temperature',
        'col': 'tmin_k'},
    'vs': {
        'nc': 'agg_met_tmmn_1979_CurrentYear_CONUS',
        'var': 'daily_minimum_temperature',
        'col': 'u2_ms'},
    'sph': {
        'nc': 'agg_met_tmmn_1979_CurrentYear_CONUS',
        'var': 'daily_minimum_temperature',
        'col': 'q'},
}


def elevation_from_coordinate(lat, lon):
    """ Use Earth Engine API to get elevation data (in meters) from decimal degree coordinates.
    Dataset referenced is NASA SRTM Digital Elevation 30m. """
    img = ee.Image("USGS/SRTMGL1_003")
    point = ee.Geometry.Point(lon, lat)
    sample = img.sample(point).getInfo()
    elev = sample['features'][0]['properties']['elevation']
    # print(lon, lat, elev)
    return elev


def step_3(fields, gm_out, out_file):
    """ Contents of step 3 - gridmet and NLDAS (and step 2b!) data to NetCDF.
    fields: str, location of gee field shapefile asset, form of 'projects/cloud_project/assets/asset_filename'
    out_file:
    """
    # TODO: create intermediate ncs.

    print("Begin step 3 processing:")

    # if os.path.exists(out_file):
    #     print('{} exists, skipping'.format(out_file))
    #     return 1

    all_3_start = time.time()

    gmet_list = []  # empty list for storing gridmet data for each variable.

    start = '1987-01-01'
    end = '2023-12-31'
    # start = '2018-01-01'
    # end = '2018-12-31'

    start_time = time.time()

    if os.path.exists(gm_out):
        print('{} exists, skipping'.format(gm_out))
        # extracting coordinates for corrections below
        ds = xarray.open_dataset(gm_out)
        ds_coords = xarray.Coordinates(ds.coords)
    else:
        for p, col in CLIMATE_COLS.items():  # 6s
            # No buffer added in GridMet from chmdata, so any desired buffer needs to be added here.
            gmet = GridMet(variable=p, start=start, end=end,
                           bbox=BBox(bnds[0] - 0.1, bnds[2] - 0.1, bnds[3] + 0.1, bnds[1] + 0.1))
            gmet = gmet.subset_nc(return_array=True)  # returns an xarray.Dataset w/ dimentions(lat, lon, time)
            gmet_input = gmet[list(gmet.data_vars)[0]]  # indexes xarray.Dataset for the data variable
            gmet_list.append(gmet_input)
        ds = xarray.merge(gmet_list)
        ds = ds.rename({'time': 'date'})

        ds = ds.xvec.extract_points(centroids, x_coords="lon", y_coords="lat", index=True)  # very fast
        ds = ds.swap_dims({"geometry": "FID"})
        ds = ds.reset_coords("geometry", drop=True)  # Get rid of geometry index

        renaming = {'daily_mean_reference_evapotranspiration_alfalfa': 'etr_mm',
                    'daily_mean_reference_evapotranspiration_grass': 'eto_mm',
                    'precipitation_amount': 'prcp_mm',
                    'daily_mean_shortwave_radiation_at_surface': 'srad_wm2',
                    'daily_maximum_temperature': 'tmax_c',  # still need conversion, but start w/ eventually correct name.
                    'daily_minimum_temperature': 'tmin_c',  # still need conversion, but start w/ eventually correct name.
                    'daily_mean_wind_speed': 'u2_ms',
                    'daily_mean_specific_humidity': 'q'}
        ds = ds.rename(renaming)

        # Additional variables: elevation (need ee) and vapor pressure
        elevs = [elevation_from_coordinate(lat=point.coords[0][1], lon=point.coords[0][0]) for point in centroids]
        ds['elevation'] = xarray.Variable('FID', elevs, {'units': 'm'})
        p_air = air_pressure(ds['elevation'])
        ea_kpa = actual_vapor_pressure(ds['q'], p_air)
        ds['ea_kpa'] = xarray.Variable(['date', 'FID'], ea_kpa.copy(),
                                       {'units': 'kPa', 'description': 'Actual vapor pressure'})  # This takes a bit.
        # Adjusting temperature data (started in K, turn to deg C)
        for i in ['tmax_c', 'tmin_c']:
            temp_attr = ds[i].attrs
            temp_attr['units'] = 'C'
            ds[i] = ds[i] - 273.15
            # ds[i] = ds[i].assign_attrs(units='C')
            ds[i].attrs.update(temp_attr)

        # print()
        # print(list(ds.keys()))
        # print()
        print("1/6 Gridmet: {:.2f} seconds".format(time.time() - start_time))
        print(f"  Current process memory: {get_process_memory() / (1024 ** 2):.2f} MB")

    # memory too full to continue after this point, but we need to be able to access info for the corrections...
    # pull out ET variables from dataset
    et_df = ds[['etr_mm', 'eto_mm']].to_dataframe()
    start_time = time.time()
    ds.to_netcdf(gm_out)  # memory too full to save to netcdf?
    print("Gridmet saving to nc: {:.2f} seconds".format(time.time() - start_time))
    # print()
    # print(ds)
    ds_coords = xarray.Coordinates(ds.coords)  # save coords for initializing next ds.
    ds.close()  # release resources - memory above and below read the same. Were resources not released?

    # create new ds with coordinates saved above
    # gm_ds = xarray.open_dataset(gm_out)  # to access variables
    ds = xarray.Dataset(coords=ds_coords)
    # print(ds)
    # print()
    # print(et_df)

    # # Gridmet ET corrections
    # correction rasters are in EPSG:5071, so use original gdf to match.
    start_time = time.time()
    gridmet_ras = os.path.join(main_dir, 'openet_pilot/gridmet/correction_surfaces_aea')
    for etvar in ['etr_mm', 'eto_mm']:
        rasters = [os.path.join(gridmet_ras, 'gridmet_corrected_{}_{}.tif'
                                .format(etvar[:3], m)) for m in range(1, 13)]
        gridmet_factors = []
        for r in rasters:
            ras = xarray.open_dataset(r, engine='rasterio')
            vals = ras.xvec.extract_points(gdf['centroids'], x_coords="x", y_coords="y",
                                           index=False)['band_data'].values[0]
            gridmet_factors.append(vals)
        gridmet_factors = np.asarray(gridmet_factors)
        # print(np.shape(gridmet_factors))
        # print(gridmet_factors)
        df = et_df[etvar]
        # df = gm_ds[etvar].to_dataframe()
        corr = "{}_corrected".format(etvar)
        num = 0
        for point in df.index.levels[1]:
            # print(point)
            for month in range(1, 13):
                corr_factor = gridmet_factors[month-1][num]
                idx = [i for i in df.index if (i[0].month == month and i[1] == point)]
                df.loc[idx] = df.loc[idx] * corr_factor
            num += 1
        # print(df)
        # Do I need additional attributes?
        ds[corr] = xarray.Variable(['date', 'FID'], df.to_xarray(), {'units': 'mm'})

    # print()
    # print(ds)
    # print(ds['date'].values)
    # print()
    print("2/6 ET corrections: {:.2f} seconds".format(time.time() - start_time))
    print(f"  Current process memory: {get_process_memory() / (1024 ** 2):.2f} MB")
    # print()
    # print(ds)

    # # Getting NLDAS precip
    start_time = time.time()
    # gridmet is utc-6, US/Central, NLDAS is UTC-0
    # shifting NLDAS to UTC-6 is the most straightforward alignment
    s = pd.to_datetime(start) - timedelta(days=1)
    e = pd.to_datetime(end) + timedelta(days=2)
    temp = centroids.index
    centroids.index = np.arange(10)  # It failed earlier because it had a non-zero-starting index.
    nldas = nld.get_bycoords(centroids, start_date=s, end_date=e, variables=['prcp'], source='grib')  # pd df, 11s
    centroids.index = temp  # Revert back to FID so it doesn't screw anything up later.
    # I don't know how to check that this preserves order...
    nldas.index = nldas.index.rename('date')
    hr_cols = ['prcp_hr_{}'.format(str(i).rjust(2, '0')) for i in range(0, 24)]

    central = pytz.timezone('US/Central')
    nldas = nldas.tz_convert(central)
    hourly_ppt = nldas.pivot_table(columns=nldas.index.hour, index=nldas.index.date)  # works!
    hourly_ppt = hourly_ppt.loc[ds['date'].values]  # got rid of like 3 rows.

    # rename columns
    hourly_ppt = hourly_ppt.droplevel(level='variable', axis=1)
    hourly_ppt = hourly_ppt.rename(columns=dict(zip(range(24), hr_cols)), level='date')
    # hourly_ppt.columns = hourly_ppt.columns.set_names('variable', level=1)  # I think this also works?
    hourly_ppt.columns = hourly_ppt.columns.set_names(['FID', 'variable'])
    hourly_ppt.index = hourly_ppt.index.rename('date')

    # print()
    # print(hourly_ppt)

    for i in hourly_ppt.columns.levels[0]:
        nan_ct = np.sum(np.isnan(hourly_ppt[i].values), axis=0)
        if sum(nan_ct) > 100:
            # raise ValueError('Too many NaN in NLDAS data')  # Highest number is approaching 2% of data... Fine?
            print('{}: Too many NaN in NLDAS data ({} NaN)'.format(i, sum(nan_ct)))
        if np.any(nan_ct):
            hourly_ppt[i] = hourly_ppt[i].fillna(0.)
        # How to supress "fragmented" warnings below?
        hourly_ppt[i, 'nld_ppt_d'] = hourly_ppt[i].sum(axis=1)  # Many same entries, there are 2 cells for 10 fields.
    hourly_ppt = hourly_ppt.stack(level=0)  # moves multiindex column level to the index.
    nldas = hourly_ppt.to_xarray()
    # Make coords match the gridmet data
    nldas = nldas.assign_coords({'date': ds['date'], 'FID': ds['FID']})

    # print()
    print("3/6 nldas: {:.2f} seconds".format(time.time() - start_time))
    print(f"  Current process memory: {get_process_memory() / (1024 ** 2):.2f} MB")

    ds = ds.merge(nldas)
    # print()
    # print(ds)

    # SNODAS
    start_time = time.time()
    snow_yrs = []
    for y in range(2005, 2024):
        snow_yr = xarray.open_dataset(os.path.join(main_dir, "snodas/netcdf2/{}WGS84MT.nc".format(y)))
        # Extract field locations
        snow_yr = snow_yr.xvec.extract_points(centroids, x_coords="lon", y_coords="lat", index=True)
        snow_yr = snow_yr.drop_vars(['crs', 'Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7', 'Band8'])
        snow_yr = snow_yr.rename({'time': 'date'})
        snow_yrs.append(snow_yr)
    snow = xarray.concat(snow_yrs, "date")
    snow = snow.rename({'Band1': 'swe_m'})
    # Mess with file so it can be saved as a netcdf.
    snow = snow.swap_dims({"geometry": "FID"})
    snow = snow.reset_coords("geometry", drop=True)  # Get rid of geometry index

    # print()
    # print(snow)
    # print()
    print("4/6 snodas: {:.2f} seconds".format(time.time() - start_time))  # 10.27 seconds for 19 years!
    print(f"  Current process memory: {get_process_memory() / (1024 ** 2):.2f} MB")

    ds = ds.merge(snow)  # What about the Sept-May thing?
    # print()
    # print(ds)

    # Soil and irrigation properties
    start_time = time.time()
    irr = get_irrigation_direct_nc(fields, debug=False, selector=FEATURE_ID)
    ssurgo = get_ssurgo_direct_nc(fields, debug=False, selector=FEATURE_ID)
    props = irr.merge(ssurgo)
    # print(props)

    print("5/6 soil and irrigation properties: {:.2f} seconds".format(time.time() - start_time))  # 1-ish seconds
    print(f"  Current process memory: {get_process_memory() / (1024 ** 2):.2f} MB")

    ds = ds.merge(props)
    # print()
    # print(ds)

    start_time = time.time()
    ds.to_netcdf(out_file)
    print()
    print("6/6 Saving netcdf: {:.2f} seconds".format(time.time() - start_time))
    print(f"  Current process memory: {get_process_memory() / (1024 ** 2):.2f} MB")

    print("Total Step 3 processing time: {:.2f} seconds".format(time.time() - all_3_start))
    # 3 minutes. Where can I save time? Make saving faster, that's the slowest bit.


def step_4(fields, step3_out, out_file, do_inv_irr=True):
    """
    fields: str, location of gee field shapefile asset, form of 'projects/cloud_project/assets/asset_filename'
    out_file:
    """
    # Step 4.1 is not needed, we've done that with the netcdfs.

    # Upload shapefile to GEE on your own

    # Can I change to using bounds, and then use xvec to get the polygon-specific stuff?
    # That might work, but I was not able to get it going right now. Future problem there.
    # Potential problem: might be slower for larger areas with low field density?

    if do_inv_irr:
        types_ = ['inv_irr', 'irr']
    else:
        types_ = ['irr']
    sensing_params = ['ndvi', 'etf']
    strt_yr, end_yr = 2004, 2023

    ndvi_irr = None

    # every sample is divided into a 'purely' irrigated section (i.e., 'irr') and an unirrigated one (i.e., 'inv_irr')
    # this allows us to build a model for irrigated areas that aren't contaminated by unirrigated areas.

    if os.path.exists(out_file):
        print('{} exists, skipping'.format(out_file))
    else:
        rs_xrs = []
        start_time = time.time()
        for mask_type in types_:
            for sensing_param in sensing_params:
                # This bit is slow.
                if sensing_param == 'etf':
                    imgs = clustered_sample_etf_direct_1(fields, debug=False, mask_type=mask_type, start_yr=strt_yr,
                                                         end_yr=end_yr, feature_id=FEATURE_ID, drops=list(gdf.columns))
                elif sensing_param == 'ndvi':
                    imgs = clustered_sample_ndvi_direct_1(fields, debug=False, mask_type=mask_type, start_yr=strt_yr,
                                                          end_yr=end_yr, feature_id=FEATURE_ID, drops=list(gdf.columns))
                else:
                    imgs = None
                # print()
                # print(result1)

                # This bit is fast.
                ts, count = clustered_landsat_time_series_nc(imgs, start_yr=strt_yr, end_yr=end_yr,
                                                             feature_id=FEATURE_ID,
                                                             var_name='{}_{}'.format(sensing_param, mask_type))
                # print()
                # print(ts)
                # print(count)

                # plt.figure()
                # for i in range(10):
                #     plt.plot(ts.to_dataarray().values[0, :, i])
                #     plt.plot(count.to_dataarray().values[0, :, i])
                # plt.show()

                rs_xrs.append(ts)
                rs_xrs.append(count)  # What does count end up being used for?
                if mask_type == 'irr' and sensing_param == 'ndvi':
                    ndvi_irr = ts

        print("EE etf and ndvi exports: {:.2f} seconds".format(time.time() - start_time))  # 70 seconds for 3 years.
        print(f"  Current process memory: {get_process_memory() / (1024 ** 2):.2f} MB")

        if ndvi_irr:
            # Finally, we use both the irrigation and NDVI data to run an analysis to infer
            # simple agricultural information and get an estimate of the potential irrigation dates.
            irr = step3_out
            # cuttings_nc = os.path.join(landsat, 'uy10_cuttings.nc')
            irr_days = detect_cuttings_nc(ndvi_irr, irr, irr_threshold=0.1)
            rs_xrs.append(irr_days)
        else:
            print("No irrigated ndvi info, skipping detect_cuttings")

        # Next, join the daily remote sensing data to a single file.
        # This will be a single, large file to hold all the NDVI and ETf data.
        start_time = time.time()
        rs = xarray.merge(rs_xrs)
        # print()
        # print(rs)
        rs.to_netcdf(out_file)
        print("Saving EE exports: {:.2f}".format(time.time() - start_time))
        print(f"  Current process memory: {get_process_memory() / (1024 ** 2):.2f} MB")


if __name__ == '__main__':
    # Establish paths
    if os.path.exists('F:/FileShare'):
        main_dir = 'F:/FileShare'  # on remote server
    else:
        main_dir = 'F:'  # on local computer
    root = os.path.join(main_dir, 'BOR_UYWS_2025')

    # Tracking memory to get a better handle on what's going on.
    memory = psutil.virtual_memory()
    # print(f"Total memory: {memory.total / (1024 ** 3):.2f} GB")
    # print(f"Available memory: {memory.available / (1024 ** 3):.2f} GB")
    # print(f"Memory percentage used: {memory.percent}%")
    print(f"Memory available: {memory.available / (1024 ** 2):.2f} MB ({100 - memory.percent:.2f}%)")
    print(f"  Current process memory: {get_process_memory() / (1024 ** 2):.2f} MB")

    shp_name = '067_Park_fields'  # all 2000 fields
    # shp_name = 'mt_sid_uy10'  # switching to smaller fields for testing.
    ee_fields = 'projects/ee-hehaugen/assets/{}'.format(shp_name)
    shapefile_path = os.path.join(root, '{}.shp'.format(shp_name))
    gdf = gpd.read_file(shapefile_path)
    gdf.index = gdf['FID']
    # print(gdf)
    # step_1()  # Visualizing the study area

    # Need bounds for steps 3 and 4
    # Convert to correct coordinate system. Need bounds and field centroids.
    gdf_4326 = gdf.to_crs("EPSG:4326")
    bnds = gdf_4326.total_bounds
    gdf['centroids'] = gdf.geometry.centroid  # doesn't like this? Well, we'll keep moving forward.
    centroids = gdf['centroids'].to_crs('EPSG:4326')
    # centroids = gdf_4326.geometry.centroid  # Does it like this one better? Nope...

    # output file locations
    gm_nc = os.path.join(root, 'swim', 'uy_all_gm.nc')
    step3 = os.path.join(root, 'swim', 'uy_all_step3.nc')
    step4 = os.path.join(root, 'swim', 'uy_all_remote_sensing.nc')
    final = os.path.join(root, 'swim', 'uy_all_input.nc')

    sys.path.append(root)
    sys.path.insert(0, os.path.abspath('../..'))
    sys.setrecursionlimit(5000)  # What does this do?

    print()

    all_start = time.time()

    if not is_authorized():
        ee.Authenticate()  # cannot reach this line?
    ee.Initialize()

    # both steps will only run if out_file is not detected.
    step_3(ee_fields, gm_nc, step3)
    step_4(ee_fields, step3, step4)

    # merging the resulting files
    start_t = time.time()
    step3 = xarray.open_dataset(step3)
    step4 = xarray.open_dataset(step4)
    all_input = xarray.merge([step3, step4])  # causes datetime alignment and introduces nans, making dtype=float.
    print()
    print(all_input)
    all_input.to_netcdf(final)
    print()
    print("Merging files: {:.2f}".format(time.time() - start_t))  # Fast.

    all_end = time.time()
    print()
    print("Total input netcdf processing time: {:.0f}".format(all_end - all_start))

    # # ------------------------------------
    # # Take this stuff out to a "run_all" file or something.
    # # Now actually run the model. (Step 5)
    # from swim.config import ProjectConfig
    # from swim.input import SamplePlots
    #
    # # Our project workspace will replace the "{project_root}" in the paths in the config file,
    # # several directories will be placed there. Let's use the top level directory of this tutorial
    # project_ws = os.path.join(root, 'examples', 'uy10')
    # print(f'Setting project root to {project_ws}')
    #
    # config_file = os.path.join(root, 'examples', 'uy10', 'uy10_config.toml')
    # config = ProjectConfig()
    # config.read_config(config_file, project_ws)
    #
    # fields = SamplePlots()
    # fields.initialize_plot_data_nc(config)  # loads
    # var = list(fields.input.keys())
    # for i in range(len(var)):
    #     print(i, var[i])
    # print(fields.input)
    #
    # from model.etd import obs_field_cycle
    #
    # # Let's time this run - slow! :(
    # start_time = time.time()
    # fields.output = obs_field_cycle.field_day_loop_nc_1(config, fields, debug_flag=True)
    # end_time = time.time()
    # print('\nExecution time: {:.2f} seconds\n'.format(end_time - start_time))
    #
    # # print(fields.output)
    #
    # # Save the results for a given field and plot them.
    # field = 354
    # out_df = fields.output[field].copy()
    #
    # print()
    # print(out_df.head())
    # print(out_df.columns)
    #
    # out_data_loc = os.path.join(root, 'examples', 'uy10', 'combined_output_{}.csv'.format('354'))
    #
    # in_df = fields.input_to_dataframe(field)
    # df = pd.concat([out_df, in_df], axis=1, join='inner', ignore_index=False)
    # df.to_csv(out_data_loc)
    # print(df.shape)

# ========================= EOF ====================================================================
