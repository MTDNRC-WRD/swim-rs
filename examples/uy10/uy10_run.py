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
from data_extraction.ee.etf_export import clustered_sample_etf_direct_1
from data_extraction.ee.ndvi_export import clustered_sample_ndvi_direct_1
from data_extraction.ee.ee_utils import is_authorized

# Step 3 imports
import xarray
import xvec  # this is used, just tacked on to xarray stuff. Needs to be imported.
import pandas as pd
from datetime import timedelta
import pytz
import pynldas2 as nld
import numpy as np
from data_extraction.gridmet.gridmet import air_pressure, actual_vapor_pressure
from chmdata.thredds import GridMet, BBox
# 2b
from data_extraction.ee.ee_props import get_irrigation_direct_nc, get_ssurgo_direct_nc

# Step 4 imports
# 4.2
from prep.landsat_sensing import clustered_landsat_time_series_nc, detect_cuttings_nc


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
# SWE = 'projects/earthengine-legacy/assets/projects/climate-engine/snodas/daily'
# We must specify which column in the shapefile represents the field's unique ID, in this case it is 'FID'
FEATURE_ID = 'FID'

# Step 2a is taken care of in step 4
# Step 2b is taken care of in step 3

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
    """ Contents of step 3 - gridmet and NLDAS (and step 2b!) data to NetCDF. """
    print("Begin step 3 processing:")
    all_start = time.time()

    gmet_list = []  # empty list for storing gridmet data for each variable.

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
    # rename time coordinate?
    ds = ds.rename({'time': 'date'})

    # ds = ds.xvec.zonal_stats(geometry=gdf_4326.geometry, x_coords="lon", y_coords="lat",
    #                          stats="mean(coverage_weight=none)", method="exactextract")  # Slowwww... >30 mins.
    ds = ds.xvec.extract_points(centroids, x_coords="lon", y_coords="lat", index=True)  # very fast, 0.039 seconds.
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

    # print(ds)

    # # Gridmet ET corrections
    # correction rasters are in EPSG:5071, so use original gdf to match.
    start_time = time.time()
    gridmet_ras = 'F:/openet_pilot/gridmet/correction_surfaces_aea'
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
        df = ds[etvar].to_dataframe()
        # print(df)
        corr = "{}_corrected".format(etvar)
        num = 0
        for point in df.index.levels[1]:
            # print(point)  # Looks good.
            for month in range(1, 13):
                corr_factor = gridmet_factors[month-1][num]
                idx = [i for i in df.index if (i[0].month == month and i[1] == point)]
                df.loc[idx, corr] = df.loc[idx, etvar] * corr_factor
            num += 1
        # print(df)
        # temp_attr = ds[long_var].attrs
        # print(temp_attr)
        # It might not like this...
        # Do I need additional attributes?
        ds[corr] = xarray.Variable(['date', 'FID'], df[corr].to_xarray(), {'units': 'mm'})
    # df = df[out_cols]  # Do I need to reduce the columns present in the final dataset?

    # print(ds)
    # print(ds['date'].values)
    # print()
    print("2/6 ET corrections: {:.2f} seconds".format(time.time() - start_time))
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

    ds = ds.merge(nldas)
    print()
    print(ds)

    # SNODAS
    start_time = time.time()
    snow_yrs = []
    for y in range(2005, 2024):
        snow_yr = xarray.open_dataset("F:/snodas/netcdf2/{}WGS84MT.nc".format(y))
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

    print()
    print(snow)
    # print()
    print("4/6 snodas: {:.2f} seconds".format(time.time() - start_time))  # 10.27 seconds for 19 years!

    ds = ds.merge(snow)  # What about the Sept-May thing?
    # ValueError: cannot reindex or align along dimension 'date' because of conflicting dimension sizes: {13514, 19} (note: an index is found along that dimension with size=13514)
    # print()
    # print(ds)

    # Soil and irrigation properties
    start_time = time.time()
    fields = 'projects/ee-hehaugen/assets/mt_sid_uy10'
    irr = get_irrigation_direct_nc(fields, debug=False, selector=FEATURE_ID)
    ssurgo = get_ssurgo_direct_nc(fields, debug=False, selector=FEATURE_ID)
    props = irr.merge(ssurgo)
    print(props)

    print("5/6 soil and irrigation properties: {:.2f} seconds".format(time.time() - start_time))  # 1-ish seconds

    ds = ds.merge(props)
    print()
    print(ds)

    # Then save to netcdf and see how to input it into swim...
    start_time = time.time()
    ds.to_netcdf("C:/Users/CND571/PycharmProjects/swim-rs1/examples/uy10/data/met_timeseries/uy10_step3.nc")
    print()
    print("6/6 Saving netcdf: {:.2f} seconds".format(time.time() - start_time))

    print("Total Step 3 processing time: {:.2f} seconds".format(time.time() - all_start))
    # 3 minutes. Where can I save time? Make saving faster, that's the slowest bit.


def step_4(do_inv_irr=True):
    """ """
    # Step 4.1 is not needed, we've done that with the netcdfs.

    # Upload shapefile to GEE on your own

    # Can I change to using bounds, and then use xvec to get the polygon-specific stuff?
    # That might work, but I was not able to get it going right now. Future problem there.
    # Potential problem: might be slower for larger areas with low field density?

    fields = 'projects/ee-hehaugen/assets/mt_sid_uy10'

    tutorial_dir = os.path.join(root, 'examples', 'uy10')
    landsat = os.path.join(tutorial_dir, 'data', 'landsat')

    remote_sensing_file = os.path.join(landsat, 'remote_sensing.nc')

    if do_inv_irr:
        types_ = ['inv_irr', 'irr']
    else:
        types_ = ['irr']
    sensing_params = ['ndvi', 'etf']
    strt_yr, end_yr = 2004, 2023

    ndvi_irr = None

    # every sample is divided into a 'purely' irrigated section (i.e., 'irr') and an unirrigated one (i.e., 'inv_irr')
    # this allows us to build a model for irrigated areas that aren't contaminated by unirrigated areas.

    if os.path.exists(remote_sensing_file):
        print('{} exists, skipping'.format(remote_sensing_file))
    else:
        rs_xrs = []
        start = time.time()
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

        print("EE etf and ndvi exports: {:.2f} seconds".format(time.time() - start))  # 70 seconds for 3 years.

        if ndvi_irr:
            # Finally, we use both the irrigation and NDVI data to run an analysis to infer
            # simple agricultural information and get an estimate of the potential irrigation dates.
            irr = "C:/Users/CND571/PycharmProjects/swim-rs1/examples/uy10/data/met_timeseries/uy10_step3.nc"  # from step 3
            # cuttings_nc = os.path.join(landsat, 'uy10_cuttings.nc')
            irr_days = detect_cuttings_nc(ndvi_irr, irr, irr_threshold=0.1)
            rs_xrs.append(irr_days)
        else:
            print("No irrigated ndvi info, skipping detect_cuttings")

        # Next, join the daily remote sensing data to a single file.
        # This will be a single, large file to hold all the NDVI and ETf data.
        start = time.time()
        rs = xarray.merge(rs_xrs)
        print()
        print(rs)
        rs.to_netcdf(remote_sensing_file)
        print("Saving EE exports: {:.2f}".format(time.time() - start))


if __name__ == '__main__':
    # # Step 1, required for other steps
    # # Load the shapefile
    # print(os.getcwd())
    # print()

    home = os.path.expanduser('~')
    root = os.path.join(home, 'PycharmProjects', 'swim-rs1')
    # print(root)

    shapefile_path = os.path.join(root, 'examples', 'uy10', 'data', 'gis', 'mt_sid_uy10.shp')
    gdf = gpd.read_file(shapefile_path)
    # print(gdf)
    gdf.index = gdf['FID']

    # step_1()  # Visualizing the study area

    # Need bounds for both steps
    # Convert to correct coordinate system. Need bounds and field centroids.
    gdf_4326 = gdf.to_crs("EPSG:4326")
    bnds = gdf_4326.total_bounds
    # print(bnds)
    gdf['centroids'] = gdf.geometry.centroid
    centroids = gdf['centroids'].to_crs('EPSG:4326')  # Likes this. Same result as above...

    # # Step 2, required for other steps
    sys.path.append(root)
    sys.path.insert(0, os.path.abspath('../..'))
    sys.setrecursionlimit(5000)  # What does this do?

    print()
    # if not is_authorized():
    #     ee.Authenticate()
    # ee.Initialize()
    #
    # all_start = time.time()
    #
    # step_3()  # will always run?
    # step_4()  # will only run if file isn't detected
    #
    # start = time.time()
    # step3 = 'C:/Users/CND571/PycharmProjects/swim-rs1/examples/uy10/data/met_timeseries/uy10_step3.nc'
    # step3 = xarray.open_dataset(step3)
    # step4 = 'C:/Users/CND571/PycharmProjects/swim-rs1/examples/uy10/data/landsat/remote_sensing.nc'
    # step4 = xarray.open_dataset(step4)
    # all_input = xarray.merge([step3, step4])  # causes datetime alignment and introduces nans, making dtype=float.
    # print()
    # print(all_input)
    # all_input.to_netcdf('C:/Users/CND571/PycharmProjects/swim-rs1/examples/uy10/data/uy10_input.nc')
    # print()
    # print("Merging files: {:.2f}".format(time.time() - start))  # Fast.
    #
    # all_end = time.time()
    # print()
    # print("Total input netcdf processing time: {:.0f}".format(all_end - all_start))

    # ------------------------------------
    # Now actually run the model. (Step 5)
    from swim.config import ProjectConfig
    from swim.input import SamplePlots

    # Our project workspace will replace the "{project_root}" in the paths in the config file,
    # several directories will be placed there. Let's use the top level directory of this tutorial
    project_ws = os.path.join(root, 'examples', 'uy10')
    print(f'Setting project root to {project_ws}')

    config_file = os.path.join(root, 'examples', 'uy10', 'uy10_config.toml')
    config = ProjectConfig()
    config.read_config(config_file, project_ws)

    fields = SamplePlots()
    fields.initialize_plot_data_nc(config)  # loads
    var = list(fields.input.keys())
    for i in range(len(var)):
        print(i, var[i])
    print(fields.input)

    from model.etd import obs_field_cycle

    # Let's time this run - slow! :(
    start_time = time.time()
    fields.output = obs_field_cycle.field_day_loop_nc_1(config, fields, debug_flag=True)
    end_time = time.time()
    print('\nExecution time: {:.2f} seconds\n'.format(end_time - start_time))

    # print(fields.output)

    # Save the results for a given field and plot them.
    field = 354
    out_df = fields.output[field].copy()

    print()
    print(out_df.head())
    print(out_df.columns)

    out_data_loc = os.path.join(root, 'examples', 'uy10', 'combined_output_{}.csv'.format('354'))

    in_df = fields.input_to_dataframe(field)
    df = pd.concat([out_df, in_df], axis=1, join='inner', ignore_index=False)
    df.to_csv(out_data_loc)
    print(df.shape)

# ========================= EOF ====================================================================
