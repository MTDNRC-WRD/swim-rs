# # 1/13/2025
# # Recreating tutorial 1_Boulder with 10 fields in the Upper Yellowstone,
# # and changing the GEE download to load directly into python objects.

# Step 1 imports
import os
import time
import geopandas as gpd
import matplotlib.pyplot as plt

# Step 2 imports
# 2a
import sys
import ee
# from data_extraction.ee.etf_export import clustered_sample_etf_direct
# from data_extraction.ee.ndvi_export import clustered_sample_ndvi_direct
from data_extraction.ee.ee_utils import is_authorized
# 2b
# from data_extraction.ee.snodas_export import sample_snodas_swe_direct
# from data_extraction.snodas.snodas import create_timeseries_json
# from data_extraction.ee.ee_props import get_irrigation_direct, get_ssurgo_direct

# Step 3 imports
import xarray
import xvec  # this is used, just tacked on to xarray stuff.
import pandas as pd
from datetime import timedelta
import pytz
import pynldas2 as nld
import numpy as np
# from data_extraction.gridmet.thredds import GridMet, BBox
from data_extraction.gridmet.gridmet import air_pressure, actual_vapor_pressure
# from GRIDtools.zonal import sample_raster_points
from chmdata.thredds import GridMet, BBox


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

    # Create directories, if they do not already exist
    data_dir = os.path.join(root, 'examples', 'uy10', 'data')
    dirs = ['snodas',
            'properties',
            'landsat',
            'bias_correction_tif',
            'gis',
            'met_timeseries',
            'input_timeseries']

    dir_paths = [os.path.join(data_dir, d) for d in dirs]
    [os.makedirs(d, exist_ok=True) for d in dir_paths]


# # Step 2
# Define Constants and Remote Sensing Data Paths
# TODO: remove hard-coded collections and use variables defined here
IRR = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'
ETF = 'projects/usgs-gee-nhm-ssebop/assets/ssebop/landsat/c02'
# We must specify which column in the shapefile represents the field's unique ID, in this case it is 'FID'
FEATURE_ID = 'FID'


def step_2():
    # # -------------------------------------------------
    # # Contents of step 2a

    # Upload shapefile to GEE on your own
    # Can I change to using bounds, and then use xvec to get the polygon-specific stuff? - yeah, I think so.

    # Change this to your own
    ee_account = 'ee-hehaugen'

    # # If you don't have gsutil, there is a workaround described below
    # command = os.path.join(home, 'google-cloud-sdk', 'bin', 'gsutil')

    # bucket = 'wudr'
    fields = 'projects/ee-hehaugen/assets/mt_sid_uy10'

    etf_dst = os.path.join(root, 'examples', 'uy10', 'data', 'landsat', 'extracts', 'etf')

    # # Here, we run the clustered_field_etf function on the uploaded asset.
    #
    # # every sample is divided into a 'purely' irrigated section (i.e., 'irr') and an unirrigated one (i.e., 'inv_irr')
    # # this allows us to build a model for irrigated areas that aren't contaminated by unirrigated areas.
    # # for this tutorial, we're going to use both
    #
    # for mask in ['inv_irr', 'irr']:
    #
    #     # the 'check_dir' will check the planned directory for the existence of the data
    #     # if a run fails for some reason, move what is complete from the bucket to the directory, then rerun
    #     # this will skip what's already there
    #     chk = os.path.join(etf_dst, '{}'.format(mask))
    #
    #     # write the directory if it's not already there
    #     if not os.path.exists(chk):
    #         os.makedirs(chk, exist_ok=True)
    #
    #     # This has a few extra columns that might need to be dropped, but we'll see!
    #     clustered_sample_etf_direct(fields, chk, debug=False, mask_type=mask, start_yr=2004, end_yr=2023,
    #                                 feature_id=FEATURE_ID, drops=list(gdf.columns))
    #
    # ndvi_dst = os.path.join(root, 'examples', 'uy10', 'data', 'landsat', 'extracts', 'ndvi')
    #
    # # Just like before, but with 'ndvi' instead of 'etf':
    # for mask in ['inv_irr', 'irr']:
    #
    #     # the 'check_dir' will check the planned directory for the existence of the data
    #     # if a run fails for some reason, move what is complete from the bucket to the directory, then rerun
    #     # this will skip what's already there
    #     chk = os.path.join(ndvi_dst, '{}'.format(mask))
    #
    #     # write the directory if it's not already there
    #     if not os.path.exists(chk):
    #         os.makedirs(chk, exist_ok=True)
    #
    #     clustered_sample_ndvi_direct(fields, chk, debug=False, mask_type=mask, start_yr=2004, end_yr=2023,
    #                                  feature_id=FEATURE_ID, drops=list(gdf.columns))

    # # # -------------------------------------------------
    # # # Contents of step 2b
    #
    # SWE = 'projects/earthengine-legacy/assets/projects/climate-engine/snodas/daily'
    #
    # # let's send all the data we get to our tutorial directories
    # snow_dst = os.path.join(root, 'examples', 'uy10', 'data', 'snodas', 'extracts')
    # if not os.path.isdir(snow_dst):
    #     os.makedirs(snow_dst, exist_ok=True)
    #
    # start_time = time.time()
    #
    # # 7.5 minutes for SWE and soils (i think the soils are really fast)
    #
    # # Here, we run the sample_snodas_swe function on the uploaded asset.
    # # Note we use 'check_dir' to check if it's already written to the directory,
    # # and 'overwrite=False' so we don't write it again if it is.
    # sample_snodas_swe_direct(fields, snow_dst, debug=False, overwrite=False, feature_id=FEATURE_ID)
    #
    # snow_out = os.path.join(root, 'examples/uy10/data/snodas/snodas.json')
    # create_timeseries_json(snow_dst, snow_out, feature_id=FEATURE_ID)
    #
    # # description = 'tutorial_irr'
    # dst = os.path.join(root, 'examples', 'uy10', 'data', 'properties')
    # get_irrigation_direct(fields, dst, debug=False, selector=FEATURE_ID)
    #
    # # description = 'tutorial_ssurgo'
    # dst = os.path.join(root, 'examples', 'uy10', 'data', 'properties')
    # get_ssurgo_direct(fields, dst, debug=False, selector=FEATURE_ID)
    #
    # print("{:.0f} seconds".format(time.time() - start_time))


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


def step_3():
    """ Contents of step 3 - gridmet and NLDAS met data to NetCDF. """
    all_start = time.time()

    gmet_list = []  # empty list for storing gridmet data for each variable.

    # Convert to correct coordinate system. Need bounds and field centroids.
    gdf_4326 = gdf.to_crs("EPSG:4326")
    bnds = gdf_4326.total_bounds
    # print(bnds)

    gdf['centroids'] = gdf.geometry.centroid
    centroids = gdf['centroids'].to_crs('EPSG:4326')  # Likes this. Same result as above...

    start = '1987-01-01'
    end = '2023-12-31'

    start_time = time.time()
    # for p in ['etr', 'pet']:  # faster for testing
    for p, col in CLIMATE_COLS.items():  # 6s
        # No buffer added in GridMet from chmdata, so any desired buffer needs to be added here.
        gmet = GridMet(variable=p, start=start, end=end,
                       bbox=BBox(bnds[0] - 0.1, bnds[2] - 0.1, bnds[3] + 0.1, bnds[1] + 0.1))
        gmet = gmet.subset_nc(return_array=True)  # returns an xarray.Dataset w/ dimentions(lat, lon, time)
        gmet_input = gmet[list(gmet.data_vars)[0]]  # indexes xarray.Dataset for the data variable
        gmet_list.append(gmet_input)
    ds = xarray.merge(gmet_list)

    # ds = ds.xvec.zonal_stats(geometry=gdf_4326.geometry, x_coords="lon", y_coords="lat",
    #                          stats="mean(coverage_weight=none)", method="exactextract")  # Slowwww... >30 mins.
    ds = ds.xvec.extract_points(centroids, x_coords="lon", y_coords="lat", index=True)  # very fast, 0.039 seconds.
    ds = ds.swap_dims({"geometry": "FID"})
    ds = ds.reset_coords("geometry", drop=True)  # Get rid of geometry index

    # Additional variables: elevation (need ee) and vapor pressure
    elevs = [elevation_from_coordinate(lat=point.coords[0][1], lon=point.coords[0][0]) for point in centroids]
    ds['elevation'] = xarray.Variable('FID', elevs, {'units': 'm'})
    p_air = air_pressure(ds['elevation'])
    ea_kpa = actual_vapor_pressure(ds['daily_mean_specific_humidity'], p_air)
    ds['ea_kpa'] = xarray.Variable(['time', 'FID'], ea_kpa.copy(),
                                   {'units': 'kPa', 'description': 'Actual vapor pressure'})  # This takes a bit.
    # Adjusting temperature data
    for i in ['daily_maximum_temperature', 'daily_minimum_temperature']:
        temp_attr = ds[i].attrs
        temp_attr['units'] = 'C'
        ds[i] = ds[i] - 273.15
        # ds[i] = ds[i].assign_attrs(units='C')
        ds[i].attrs.update(temp_attr)

    # print()
    # print(list(ds.keys()))
    # print()
    print("Gridmet: {:.2f} seconds".format(time.time() - start_time))

    # print(ds)

    # # Gridmet ET corrections
    # correction rasters are in EPSG:5071, so use original gdf to match.
    start_time = time.time()
    gridmet_ras = 'F:/openet_pilot/gridmet/correction_surfaces_aea'

    # So slow! But I think it does work.
    for long_var in ['daily_mean_reference_evapotranspiration_alfalfa',
                     'daily_mean_reference_evapotranspiration_grass']:
        if long_var == 'daily_mean_reference_evapotranspiration_alfalfa':
            var = 'etr'
        else:
            var = 'eto'
        rasters = [os.path.join(gridmet_ras, 'gridmet_corrected_{}_{}.tif'.format(var, m)) for m in range(1, 13)]
        gridmet_factors = []
        for r in rasters:
            ras = xarray.open_dataset(r, engine='rasterio')
            vals = ras.xvec.extract_points(gdf['centroids'], x_coords="x", y_coords="y",
                                           index=False)['band_data'].values[0]
            gridmet_factors.append(vals)
        gridmet_factors = np.asarray(gridmet_factors)
        # print(np.shape(gridmet_factors))
        # print(gridmet_factors)
        df = ds[long_var].to_dataframe()
        # print(df)
        corr = "{}_corrected".format(var)
        num = 0
        for point in df.index.levels[1]:
            # print(point)  # Looks good.
            for month in range(1, 13):
                corr_factor = gridmet_factors[month-1][num]
                idx = [i for i in df.index if (i[0].month == month and i[1] == point)]
                df.loc[idx, corr] = df.loc[idx, long_var] * corr_factor
            num += 1
        # print(df)
        # temp_attr = ds[long_var].attrs
        # print(temp_attr)
        # It might not like this...
        # Do I need additional attributes?
        ds[corr] = xarray.Variable(['time', 'FID'], df[corr].to_xarray(), {'units': 'mm'})

    # df = df[out_cols]  # Do I need to reduce the columns present in the final dataset?

    # print(ds)
    # print(ds['time'].values)
    # print()
    print("ET corrections: {:.2f} seconds".format(time.time() - start_time))
    # print()
    # print(ds)

    # # Getting NLDAS precip
    start_time = time.time()
    # gridmet is utc-6, US/Central, NLDAS is UTC-0
    # shifting NLDAS to UTC-6 is the most straightforward alignment
    s = pd.to_datetime(start) - timedelta(days=1)
    e = pd.to_datetime(end) + timedelta(days=2)
    centroids.index = np.arange(10)  # It failed earlier because it had a non-zero-starting index.
    nldas = nld.get_bycoords(centroids, start_date=s, end_date=e, variables=['prcp'], source='grib')  # pd df, 11s
    # print(nldas)
    hr_cols = ['prcp_hr_{}'.format(str(i).rjust(2, '0')) for i in range(0, 24)]

    central = pytz.timezone('US/Central')
    nldas = nldas.tz_convert(central)
    hourly_ppt = nldas.pivot_table(columns=nldas.index.hour, index=nldas.index.date)  # works!
    hourly_ppt = hourly_ppt.loc[ds['time'].values]  # got rid of like 3 rows.

    # rename columns
    hourly_ppt = hourly_ppt.droplevel(level='variable', axis=1)
    hourly_ppt = hourly_ppt.rename(columns=dict(zip(range(24), hr_cols)), level='time')
    # hourly_ppt.columns = hourly_ppt.columns.set_names('variable', level=1)  # I think this also works?
    hourly_ppt.columns = hourly_ppt.columns.set_names(['FID', 'variable'])
    hourly_ppt.index = hourly_ppt.index.rename('time')

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
    nldas = nldas.assign_coords({'time': ds['time'], 'FID': ds['FID']})

    # print()
    print("nldas: {:.2f} seconds".format(time.time() - start_time))

    ds = ds.merge(nldas)
    print()
    print(ds)

    # Then save to netcdf and see how to input it into swim...
    start_time = time.time()
    ds.to_netcdf("C:/Users/CND571/PycharmProjects/swim-rs1/examples/uy10/data/met_timeseries/uy10_step3.nc")
    print()
    print("Saving netcdf: {:.2f} seconds".format(time.time() - start_time))

    print("Total Step 3 processing time: {:.2f} seconds".format(time.time() - all_start))
    # 3 minutes. Where can I save time? Make saving faster, that's the slowest bit.


if __name__ == '__main__':
    # # Step 1, required for other steps
    # Load the shapefile
    print(os.getcwd())
    print()

    home = os.path.expanduser('~')
    root = os.path.join(home, 'PycharmProjects', 'swim-rs1')
    print(root)

    shapefile_path = os.path.join(root, 'examples', 'uy10', 'data', 'gis', 'mt_sid_uy10.shp')
    gdf = gpd.read_file(shapefile_path)
    # print(gdf)
    gdf.index = gdf['FID']

    # step_1()

    # # Step 2, required for other steps
    sys.path.append(root)
    sys.path.insert(0, os.path.abspath('../..'))
    sys.setrecursionlimit(5000)

    if not is_authorized():
        ee.Authenticate()
    ee.Initialize()

    # step_2()

    # step_3()
# ========================= EOF ====================================================================
