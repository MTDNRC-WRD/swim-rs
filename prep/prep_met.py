# # 1/28/26 hannah.haugen@mt.gov

import os
import time
import geopandas as gpd
import matplotlib.pyplot as plt
import sys
import xarray
import xvec  # this is used, just tacked on to xarray stuff. Does need to be imported.
import pandas as pd
import numpy as np
from tqdm import tqdm
from chmdata.thredds import GridMet, BBox
import requests
import urllib
import sqlite3
import cmethods

from prep.reference_et import pm_fao56_ref
from prep.reference_et import extraterrestrial_r, calc_rso


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


# Specify which column in the shapefile represents the field's unique ID
FEATURE_ID = 'FID_1'

# # for GridMET
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
    # 'vs': {
    #     'nc': 'agg_met_tmmn_1979_CurrentYear_CONUS',
    #     'var': 'daily_minimum_temperature',
    #     'col': 'u2_ms'},
    # 'sph': {
    #     'nc': 'agg_met_tmmn_1979_CurrentYear_CONUS',
    #     'var': 'daily_minimum_temperature',
    #     'col': 'q'},
}


# SLOOOOOOW
def elevation_from_coordinate(lat: float, lon: float):
    """ Get elevation in meters from decimal degree coordinates using USGS National Map services.
    Args:
        lat: latitude in decimal degrees.
        lon: longtidue in decimal degrees.

    Returns:
        elev: float, elevation in meters.
    """
    params = {"output": "json", "x": lon, "y": lat, "units": "Meters"}

    url = r"https://epqs.nationalmap.gov/v1/json?"
    result = requests.get(url + urllib.parse.urlencode(params), timeout=20)
    try:
        print(result.json())
        elev = float(result.json()["value"])
    except requests.exceptions.JSONDecodeError:
        print(result.text)

    return elev


def wind_2m(uz, zw=10):
    """ Convert wind measured at a height of zw meters to 2 meters. For measurements over clipped grass.

    See https://www.apogeeinstruments.com/content/EWRI-ASCE-Reference-ET-Appendices.pdf
    pdf page 47, page B-10 if more detailed information is needed.

    Parameters
    ----------
    uz: wind speed measured at zw meters, m/s
    zw: height of wind measurement, meters
    """
    u2 = (uz * 4.87) / np.log(67.8 * zw - 5.42)
    return u2


def est_srad_wm2(start_dts, centroid_series, elevations):
    """Estimate annual solar radiation time series. Probably an overestimate."""
    tindex = pd.date_range('2000-01-01', '2000-12-31')
    # latitude doesn't change much throughout a county. Just take the mean.
    lat = np.mean([point.coords[0][1] * np.pi / 180 for point in centroid_series])  # convert to rads
    r_a = extraterrestrial_r(tindex, lat=lat, shape=[366])
    elev = np.mean(elevations)  # also take mean elevation, this has much less effect than latitude.
    srad = calc_rso(r_a, elev) * 11.6 * 0.8  # from MJ / m2 d to wm2, plus a factor to account for clouds.

    long_srad = pd.DataFrame(columns=['srad_wm2'], index=pd.date_range(start_dts, '1978-12-31', freq='D'))
    long_srad.index = pd.to_datetime(long_srad.index)
    for j in range(366):
        long_srad.loc[long_srad.index.dayofyear == j + 1, 'srad_wm2'] = srad[j]

    # plt.figure()
    # plt.plot(long_srad)
    # plt.show()

    return long_srad


def met_data(centers, met_out, start=1987, end=2024, verbose=2):
    """ Download Livneh and GridMET data to NetCDF and apply corrections.

    Params:
        centers: geoseries of field centroids in EPSG:4326.
        out_file: str, filepath to save resulting netcdf to.
        start: int, first year of period of study.
        end: int, last year of period of study.
        verbose: int, what progress reports to print. 3: everything (including tqdm), 2: a lot, 1: some, 0: nothing.
    """

    met_start = time.time()

    beg_dts = '{}-01-01'.format(start)
    end_dts = '{}-12-31'.format(end)

    start_time = time.time()
    if os.path.exists(met_out):
        print('Met data NetCDF {} exists, skipping'.format(met_out))  # I always want this to print.
        # print()
        # extracting coordinates and data for corrections below
        # (not required when I'm doing just met data? Why do I need these for other steps?)
        # ds = xarray.open_dataset(met_out)
        # etdf = ds[['etr_mm', 'eto_mm']].to_dataframe()
        # ds_coords = xarray.Coordinates(ds.coords)
    else:
        if verbose > 0:
            # print("Begin met data processing:")
            print(time.ctime())
            print("GridMET processing:")

        # # gridmet processing
        bnds = centers.total_bounds
        gmet_list = []  # empty list for storing gridmet data for each variable.
        for p, col in CLIMATE_COLS.items():  # 6s
            # No buffer added in GridMet from chmdata, so any desired buffer needs to be added here.
            gmet = GridMet(variable=p, start=max(beg_dts, '1979-01-01'), end=end_dts,
                           bbox=BBox(bnds[0] - 0.1, bnds[2] - 0.1, bnds[3] + 0.1, bnds[1] + 0.1))
            gmet = gmet.subset_nc(return_array=True)  # returns an xarray.Dataset w/ dimentions(lat, lon, time)
            gmet_input = gmet[list(gmet.data_vars)[0]]  # indexes xarray.Dataset for the data variable
            gmet_list.append(gmet_input)
        ds = xarray.merge(gmet_list)
        # ds = ds.rename({'time': 'date'})

        ds = ds.xvec.extract_points(centers, x_coords="lon", y_coords="lat", index=True)  # very fast
        ds = ds.swap_dims({"geometry": FEATURE_ID})
        ds = ds.reset_coords("geometry", drop=True)  # Get rid of geometry index

        renaming = {'daily_mean_reference_evapotranspiration_alfalfa': 'etr_mm',
                    'daily_mean_reference_evapotranspiration_grass': 'eto_mm',
                    'precipitation_amount': 'prcp_mm',
                    'daily_mean_shortwave_radiation_at_surface': 'srad_wm2',
                    'daily_maximum_temperature': 'tmax_c',  # needs conversion, but start w/ eventually correct name.
                    'daily_minimum_temperature': 'tmin_c',  # needs conversion, but start w/ eventually correct name.
                    # 'daily_mean_wind_speed': 'u2_ms',
                    # 'daily_mean_specific_humidity': 'q'
                    }
        ds = ds.rename(renaming)

        # # Additional variables: elevation and vapor pressure - NOT NEEDED
        # elevs = [elevation_from_coordinate(lat=point.coords[0][1], lon=point.coords[0][0]) for point in centers]
        # ds['elevation'] = xarray.Variable(FEATURE_ID, elevs, {'units': 'm'})
        # p_air = air_pressure(ds['elevation'])
        # ea_kpa = actual_vapor_pressure(ds['q'], p_air)
        # ds['ea_kpa'] = xarray.Variable(['date', FEATURE_ID], ea_kpa.copy(),
        #                                {'units': 'kPa', 'description': 'Actual vapor pressure'})  # This takes a bit.

        # Adjusting temperature data (started in K, turn to deg C)
        for i in ['tmax_c', 'tmin_c']:
            temp_attr = ds[i].attrs
            temp_attr['units'] = 'C'
            ds[i] = ds[i] - 273.15
            # ds[i] = ds[i].assign_attrs(units='C')
            ds[i].attrs.update(temp_attr)

        # print()
        if verbose > 1:
            # print("Gridmet: {:.0f} seconds".format(time.time() - start_time))
            print("{:.0f} seconds".format(time.time() - start_time))

        # -------------------------------
        # pull in Livneh data.
        # Fill in record with Livneh data (get enough to bias-correct!)
        start_time = time.time()  # reset count
        if start < 1979:
            if verbose > 1:
                print('Livneh processing:')
            # get livneh data
            ln_yrs = []
            for y in range(start, min(end + 1, 2013)):  # change to 2013 for bias-correction with overlap.
                met_file = os.path.join(main_dir, f'Livneh/daily_MT/livneh_MT_{y}.nc')
                ln_yr = xarray.load_dataset(met_file)
                # ln_yr = ln_yr.rename({'time': 'date'})

                ln_yr = ln_yr.xvec.extract_points(centers, x_coords="lon", y_coords="lat", index=True)
                ln_yr = ln_yr.swap_dims({"geometry": FEATURE_ID})  # FEATURE_ID
                ln_yr = ln_yr.reset_coords("geometry", drop=True)  # Get rid of geometry index

                renaming = {'Prec': 'prcp_mm',
                            'Tmax': 'tmax_c',
                            'Tmin': 'tmin_c',
                            'wind': 'u2_ms',  # needs conversion, but start w/ eventually correct name
                            }
                ln_yr = ln_yr.rename(renaming)
                ln_yrs.append(ln_yr)

            livneh = xarray.concat(ln_yrs, "time")
            # Up to this point, it takes a second, but not too long for 15 fields. 30 seconds?

            # print(livneh)

            lats_rad = [point.coords[0][1] * np.pi / 180 for point in centers]  # fao56 uses latitude in radians.

            # convert 10m to 2m wind speed
            livneh['u2_ms'] = wind_2m(livneh['u2_ms'])

            # Need elevations for et calculations.
            # Current fastest solution is to fetch already-calculated gm_elevations from the OpenET database.
            conec = sqlite3.connect(os.path.join(r"F:\openet_pilot\opnt_analysis_03042024_Copy1.db"))
            elevs = []
            for i in livneh['FID_1'].values:
                # print(i)
                # print(i, pd.read_sql(f"SELECT elev_gm FROM field_data where fid='{i}'", conec).values[0,0])
                elevs.append(pd.read_sql(f"SELECT elev_gm FROM field_data where fid='{i}'", conec).values[0, 0])
            conec.close()

            # print(livneh)  # looking good!
            # print(time.time() - start_time, f"seconds for Livneh base")

            livneh['eto_mm'] = pm_fao56_ref(tmean=None, wind=livneh['u2_ms'], tmax=livneh['tmax_c'],
                                            tmin=livneh['tmin_c'], elevation=np.asarray(elevs),
                                            lat=lats_rad)
            livneh['etr_mm'] = pm_fao56_ref(tmean=None, wind=livneh['u2_ms'], tmax=livneh['tmax_c'],
                                            tmin=livneh['tmin_c'], elevation=np.asarray(elevs),
                                            lat=lats_rad, ref='alfalfa')

            # wind not needed after ET calculation
            livneh = livneh.drop_vars(['u2_ms'])

            # print(livneh)

            # print(time.time() - start_time, f"seconds for Livneh base and ET")
            if verbose > 1:
                # print("Livneh: {:.0f} seconds".format(time.time() - start_time))
                print("{:.0f} seconds".format(time.time() - start_time))

            start_time = time.time()
            if verbose > 1:
                # print("Livneh PPT bias correction")  # if we do temp too, this needs to happen before the ET calcs.
                print("Livneh + Gridmet bias correction:")

            # var = 'prcp_mm'  # adding et bias correction.
            # Does this work independently on different fields? I think so, based on documentation.
            for var in ['prcp_mm', 'eto_mm', 'etr_mm']:
                livneh[var].loc[{'time': slice(beg_dts, '1978-12-31')}] = cmethods.adjust(
                    method='quantile_mapping',
                    obs=ds[var].sel(time=slice('1979-01-01', min('2012-12-31', end_dts))),
                    simh=livneh[var].sel(time=slice('1979-01-01', min('2012-12-31', end_dts))),
                    # simulation historical, representing the overlapping period
                    simp=livneh[var].sel(time=slice(beg_dts, '1978-12-31')),
                    # simulation predicted, representing the projection period
                    n_quantiles=100,  # lowered from example 250.  Looks to do the same thing?
                    kind='*',  # since we're doing precip. Otherwise, probably do '+'. ET also '*'?
                )[var]

            if verbose > 1:
                print("{:.0f} seconds".format(time.time() - start_time))
                start_time = time.time()
                print("Livneh + Gridmet srad and concat:")

            # # srad method 1: median of available years of data (before concat)
            # # This one stalls forever, don't use.
            # med_srad = ds['srad_wm2']
            # med_srad['time'] = [i.dayofyear for i in pd.to_datetime(med_srad['time'].values)]
            # med_srad = med_srad.groupby('time').median()  # 366-length time series for all fields.
            #
            # long_srad = pd.DataFrame(columns=ds[FEATURE_ID].values,
            #                          index=pd.date_range(beg_dts, '1978-12-31', freq='D'))
            # for j in range(366):
            #     long_srad.loc[long_srad.index.dayofyear == j + 1] = med_srad[j]
            #
            # livneh['srad_wm2'] = long_srad  # results in no nans to fill.

            # concatenating the data (takes a long time)
            ds = xarray.concat([livneh.sel({'time': slice(beg_dts, '1978-12-31')}), ds], dim='time')

            # srad method 2: 80% clear sky curve (after concat)
            srad = est_srad_wm2(beg_dts, centers, elevs)  # returns generalized time series
            # copying the srad time series to all FIDs
            srad_many = np.zeros((len(srad), len(ds[FEATURE_ID])))
            for i in range(len(ds[FEATURE_ID])):
                srad_many[:, i] = np.reshape(srad, (1, -1))
            # print(srad_many.shape)
            ds['srad_wm2'].loc[{'time': srad.index}] = srad_many  # filling in missing Livneh data

            if verbose > 1:
                print("{:.0f} seconds".format(time.time() - start_time))
            # That's all the Livneh-specific stuff.

        # ET corrections on combined dataset
        if verbose > 1:
            print("ET corrections:")

        start_time = time.time()

        # pull out ET variables from dataset
        etdf = ds[['etr_mm', 'eto_mm']].to_dataframe()  # this is the right length.

        etdf['time'] = etdf.index.get_level_values(0)  # slow at the beginning, but I think it's worth it.
        etdf['month'] = [i.month for i in etdf['time']]
        if isinstance(etdf.index[0][1], str):  # str in fid will slow things dramatically in 'for point' loop below.
            etdf[FEATURE_ID] = [int(i[-4:]) for i in etdf.index.get_level_values(1)]  # save as int, assuming SID formating.
        else:
            etdf[FEATURE_ID] = etdf.index.get_level_values(1)  # take value as-is.

        # print(len(etdf))
        # print("reshuffling dataframe to allow vectorization: {:.2f} seconds".format(time.time() - start_time))  # ~1min
        # print(etdf)

        # correction rasters are in EPSG:5071, so make sure to match.
        # gridmet_ras = os.path.join(main_dir, 'openet_pilot/gridmet/correction_surfaces_aea')
        gridmet_ras = 'C:/Users/CND571/Documents/Data/swim/gridmet/gridmet_corrected/correction_surfaces_aea'
        for etvar in ['etr_mm', 'eto_mm']:
            rasters = [os.path.join(gridmet_ras, 'gridmet_corrected_{}_{}.tif'
                                    .format(etvar[:3], m)) for m in range(1, 13)]
            gridmet_factors = []
            for r in rasters:
                ras = xarray.open_dataset(r, engine='rasterio')
                vals = ras.xvec.extract_points(centers.to_crs("EPSG:5071"), x_coords="x", y_coords="y",
                                               index=False)['band_data'].values[0]
                gridmet_factors.append(vals)
            gridmet_factors = np.asarray(gridmet_factors)
            # print(np.shape(gridmet_factors))
            # print(gridmet_factors)
            etdf['factor'] = np.zeros_like(etdf[etvar])  # overwritten for the other variable
            corr = "{}_corrected".format(etvar)
            num = 0
            if verbose > 2:
                for point in tqdm(etdf[FEATURE_ID].unique(), total=len(etdf[FEATURE_ID].unique())):
                    for month in range(1, 13):
                        corr_factor = gridmet_factors[month - 1][num]
                        mask = (etdf['month'] == month) & (etdf[FEATURE_ID] == point)
                        etdf.loc[mask, 'factor'] = corr_factor
                    num += 1
            else:
                for point in etdf[FEATURE_ID].unique():
                    for month in range(1, 13):
                        corr_factor = gridmet_factors[month - 1][num]
                        mask = (etdf['month'] == month) & (etdf[FEATURE_ID] == point)
                        etdf.loc[mask, 'factor'] = corr_factor
                    num += 1
            etdf[corr] = etdf[etvar] * etdf['factor']
            # print(etdf[corr])
            ds[corr] = xarray.Variable(['time', FEATURE_ID], etdf[corr].to_xarray(), {'units': 'mm'})
            # ds[corr] = etdf[corr].to_xarray()  # this seems to work just fine...

        if verbose > 1:
            # print("ET corrections: {:.0f} seconds".format(time.time() - start_time))
            print("{:.0f} seconds".format(time.time() - start_time))

        if verbose > 2:
            print(ds)

        if verbose > 1:
            print("Saving netcdf:")
        start_time = time.time()
        ds.to_netcdf(met_out, engine="netcdf4")
        if verbose > 1:
            print(f"{time.time() - start_time:.0f} seconds")
            # print("  met data saving to nc: {:.0f} seconds".format(time.time() - start_time))

        if verbose > 0:
            print("Total met data processing time: {:.0f} seconds".format(time.time() - met_start))


if __name__ == '__main__':

    # Establish paths
    if os.path.exists('F:/FileShare'):
        main_dir = 'F:/FileShare'  # on remote server
    else:
        main_dir = 'F:'  # on local computer
    gis_dir = r"F:\SWIM_SID\statewide_irrigation_dataset_20240408\cleaner_counties_4326"

    # still not sure what this chunk is doing.  # what happens when I take this out? It failed. Was it because of this?
    sys.path.append(main_dir)
    sys.path.insert(0, os.path.abspath('../..'))
    sys.setrecursionlimit(5000)

    # Inclusive
    beg_year = 1963  # 1 extra year to account for the first year having bad model results due to initial conditions.
    end_year = 2023

    print(time.ctime())

    # location-specific processing:

    # county defines what files to read and write to.
    # county = '19'
    counties = ['19', '33', '61', '101', '51', '41', '91', '53', '15', '93', '55', '75', '37', '23', '69', '45', '79',
                '107', '21', '27', '89', '39', '35', '85', '43', '65', '63', '59', '77', '29', '17', '87', '103', '7',
                '95', '13', '1', '83', '49', '57', '5', '9', '3', '97', '67', '71', '31', '105', '73', '81', '99',
                '111', '47']  # all counties in increasing order of fields

    for county in tqdm(counties, total=len(counties)):  # try the next 5 counties, see what's up
        gis_path = os.path.join(gis_dir, f'COUNTY_NO_{county}.geojson')  # should be saved in EPSG:4326.
        # # this is fine in vscode/qgis, but here it's not the right crs (unless EPSG:4326). proj and geojson issue?
        # print(gis_path)
        print(f"County {county}")

        gdf = gpd.read_file(gis_path)
        gdf.index = gdf[FEATURE_ID]
        gdf = gdf.to_crs('EPSG:5071')
        # step_1()  # printing some statistics of gdf

        # Convert to correct coordinate system. Need bounds and field centroids.
        gdf_4326 = gdf.to_crs("EPSG:4326")
        gdf['centroids'] = gdf.geometry.centroid  # Not good?
        centroids = gdf['centroids'].to_crs('EPSG:4326')
        # print(centroids.iloc[0])  # should be decimal degrees

        # output file location
        met_nc = os.path.join(main_dir, 'SWIM_SID', 'met', f'{county:>03}_{beg_year}_{end_year}_ln_gm_corr.nc')

        print()

        met_data(centroids, met_nc, beg_year, end_year)

    print(time.ctime())

# ========================= EOF ====================================================================
