import os

# import pandas as pd
import time
import requests
import gzip

# These two endpoints are working from the same quota bin. I think.
# I don't how that works, as they might have different limits/permissions.

# API_KEY = 'C:/Users/CND571/Documents/OpenET_API.txt'
API_KEY = 'C:/Users/CND571/Documents/New_OpenET_API.txt'
ENDPOINT = 'general'

# API_KEY = 'C:/Users/CND571/Documents/Haugen_Montana_API.txt'
# ENDPOINT = 'montana'


def openet_get_fields_export(fields, start, end, attrs='FID', et_too=False, show=False,
                             api_key=API_KEY):
    """ Uses OpenET API multipolygon export endpoint to get etof data given a Google Earth Engine asset.

    Files will be exported to user's Google Drive; this is dependent on linking OpenET account to a Google account
    (I think).

    Parameters
    ----------
    fields: path to gee asset, form of 'projects/cloud_project/assets/asset_filename'
    start: beginning of period of study, 'YYYY-MM-DD' format
    end: end of period of study, 'YYYY-MM-DD' format
    attrs: str, which column/attribute in the asset should be retained in the output file.
    et_too: bool, optional; if True, also download OpenET ensemble ET over same time period and set of fields
    show: bool, optional; if True, prints the arguments for the API call
    api_key: str, optional; path to local .txt file where API key from user's OpenET account is stored.
    Key is first line in file.
    """

    # Use earth engine and gsutil to upload shapefiles to earth engine and then download resulting files from bucket.

    # This is apparently better, as it makes sure to close the file.
    with open(api_key, 'r') as f:
        api_key = f.readline()

    # key_file = open(api_key, "r")
    # api_key = key_file.readline()

    # set your API key before making the request
    header = {"Authorization": api_key}

    # endpoint arguments
    args = {
        "date_range": [
            start,
            end
        ],
        "interval": "daily",
        "overpass": True,
        "asset_id": fields,
        "attributes": [
            attrs
        ],
        "reducer": "mean",
        "model": "SSEBop",
        "variable": "ETof",  # Available variable depends on model.
        "reference_et": "gridMET",
        "units": "in"
    }

    if show:
        print(args)

    # query the api
    resp = requests.post(
        headers=header,
        json=args,
        # url="https://openet-api-montana-ic5gyecbva-uw.a.run.app/raster/export/multipolygon"
        url="https://openet-api.org/raster/export/multipolygon"
    )
    print(resp.json())
    # response = resp.json()
    # tag = response['name']
    # print(tag)

    if et_too:
        # getting et variable too, in separate file.
        args.update({"variable": "ET"})
        resp = requests.post(
            headers=header,
            json=args,
            # url="https://openet-api-montana-ic5gyecbva-uw.a.run.app/raster/export/multipolygon"
            url="https://openet-api.org/raster/export/multipolygon"
        )
        print(resp.json())


def openet_get_fields_export_1(fields, start, end, attrs='FID', et_too=False, show=False,
                               api_key=API_KEY):
    """ Uses OpenET API multipolygon export endpoint to get etof data given a Google Earth Engine asset.

    Files will be exported to user's Google Drive; this is dependent on linking OpenET account to a Google account
    (I think).

    Parameters
    ----------
    fields: path to gee asset, form of 'projects/cloud_project/assets/asset_filename'
    start: beginning of period of study, 'YYYY-MM-DD' format
    end: end of period of study, 'YYYY-MM-DD' format
    attrs: str, which column/attribute in the asset should be retained in the output file.
    et_too: bool, optional; if True, also download OpenET ensemble ET over same time period and set of fields
    show: bool, optional; if True, prints the arguments for the API call
    api_key: str, optional; path to local .txt file where API key from user's OpenET account is stored.
    Key is first line in file.
    """

    # Use earth engine and gsutil to upload shapefiles to earth engine and then download resulting files from bucket.

    # This is apparently better, as it makes sure to close the file.
    with open(api_key, 'r') as f:
        api_key = f.readline()

    # key_file = open(api_key, "r")
    # api_key = key_file.readline()

    # set your API key before making the request
    header = {"Authorization": api_key}

    # endpoint arguments
    args = {
        "date_range": [
            start,
            end
        ],
        "interval": "monthly",
        "asset_id": fields,
        "attributes": [
            attrs
        ],
        "reducer": "mean",
        "model": "Ensemble",
        "variable": "ET",  # Available variable depends on model.
        "reference_et": "gridMET",
        "units": "in"
    }

    if show:
        print(args)

    # query the api
    resp = requests.post(
        headers=header,
        json=args,
        url="https://openet-api-montana-ic5gyecbva-uw.a.run.app/raster/export/multipolygon"
        # url="https://openet-api.org/raster/export/multipolygon"
    )
    print(resp.json())
    # response = resp.json()
    # tag = response['name']
    # print(tag)

    if et_too:
        # getting et variable too, in separate file.
        args.update({"variable": "ET"})
        resp = requests.post(
            headers=header,
            json=args,
            url="https://openet-api-montana-ic5gyecbva-uw.a.run.app/raster/export/multipolygon"
            # url="https://openet-api.org/raster/export/multipolygon"
        )
        print(resp.json())


def track(track_id, api_key=API_KEY):
    # set your API key before making the request
    with open(api_key, 'r') as f:
        api_key = f.readline()
    header = {"Authorization": api_key}

    if ENDPOINT == 'montana':
        url = "https://openet-api-montana-ic5gyecbva-uw.a.run.app/raster/export/track"
    else:
        url = "https://openet-api.org/raster/export/track"

    # endpoint arguments
    args = {
        "tracking_id": track_id
    }

    # query the api
    resp = requests.get(
        headers=header,
        params=args,
        url=url
        # url="https://openet-api.org/raster/export/track"
        # url="https://openet-api-montana-ic5gyecbva-uw.a.run.app/raster/export/track"
    )

    print(resp.json())


def storage(api_key=API_KEY):
    """Retrieves information on all current files exported to OpenET's Google Bucket.

    If your OpenET account is not synced with your personal Google Earth Engine account, all exported data will be
    stored in a secure private Google Cloud Bucket. The following endpoint allows you to retrieve downloadable links
    to these files. This endpoint must be called to continue exporting if there are over 100 files which have not
    been retrieved.

    Once this endpoint has been called, all existing files will be moved to a public bucket with a 24 hour lifecycle.
    Data return is in a JSON format.

    WARNING: exported files will be automatically deleted after 7 days if not retrieved.
    """
    # set your API key before making the request
    with open(api_key, 'r') as f:
        api_key = f.readline()
    header = {"Authorization": api_key}

    if ENDPOINT == 'montana':
        url = "https://openet-api-montana-ic5gyecbva-uw.a.run.app/account/storage"
    else:
        url = "https://openet-api.org/account/storage"

    # query the api
    resp = requests.get(
        headers=header,
        url=url
        # url="https://openet-api.org/account/storage"
        # url="https://openet-api-montana-ic5gyecbva-uw.a.run.app/account/storage"
    )

    print(resp.json())


def upload(filepath, api_key=API_KEY):
    """Upload a temporary GeoJSON file to use with OpenET.

    As an alternative to storing shapefiles on Google Earth Engine, OpenET allows you to generate a temporary
    asset id by uploading a RFC 7946 formatted GeoJSON file to delineate boundaries for data extractions.

    Each file uploaded has a maximum file size of 25mb and expires after 72 hours. During this time you can use
    the generated temporary asset id for the corresponding parameter. Data return is in a JSON format.
    """
    # set your API key before making the request
    with open(api_key, 'r') as f:
        api_key = f.readline()
    header = {"Authorization": api_key}

    if ENDPOINT == 'montana':
        url = "https://openet-api-montana-ic5gyecbva-uw.a.run.app/account/upload"
    else:
        url = "https://openet-api.org/account/upload"

    # endpoint arguments
    args = {
        'file': (filepath, open(filepath, 'rb'), 'application/geo+json')
    }

    # query the api
    resp = requests.post(
        headers=header,
        files=args,
        url=url
    )

    print(resp.json())

    return resp.json()['asset_id']


def check(api_key=API_KEY):
    # set your API key before making the request
    with open(api_key, 'r') as f:
        api_key = f.readline()
    header = {"Authorization": api_key}

    if ENDPOINT == 'montana':
        url = "https://openet-api-montana-ic5gyecbva-uw.a.run.app/account/status"
    else:
        url = "https://openet-api.org/account/status"

    # query the api
    resp = requests.get(
        headers=header,
        url=url
    )

    try:
        print(resp.json())
    except requests.exceptions.JSONDecodeError:
        print(resp)


def export_stack(fields, start='2020-01-01', end='2024-12-31', api_key=API_KEY):
    """ Export GeoTIFFs of OpenET data to Google Drive. Use GEE asset as study area.

    Export is limited to 31 time steps per request. Each time step is a sepearate file.
    """
    # set your API key before making the request
    with open(api_key, 'r') as f:
        api_key = f.readline()
    header = {"Authorization": api_key}

    if ENDPOINT == 'montana':
        url = "https://openet-api-montana-ic5gyecbva-uw.a.run.app/raster/export/stack"
    else:
        url = "https://openet-api.org/raster/export/stack"

    # endpoint arguments
    args = {
        "date_range": [
            start,
            end
        ],
        "interval": "monthly",
        "asset_id": fields,
        "model": "Ensemble",
        "variable": "ET",
        "reference_et": "gridMET",
        "units": "mm",
        "encrypt": False
    }

    # query the api
    resp = requests.post(
        headers=header,
        json=args,
        url=url
    )

    print(resp.json())


def export_multipolygon(fields, start='2020-01-01', end='2024-12-31',
                        api_key=API_KEY):
    """ """
    # set your API key before making the request
    with open(api_key, 'r') as f:
        api_key = f.readline()
    header = {"Authorization": api_key}

    if ENDPOINT == 'montana':
        url = "https://openet-api-montana-ic5gyecbva-uw.a.run.app/raster/export/multipolygon"
    else:
        url = "https://openet-api.org/raster/export/multipolygon"

    # endpoint arguments
    args = {
        "date_range": [
            start,
            end
        ],
        "interval": "monthly",
        "asset_id": fields,
        "attributes": ["FID_1"],
        "model": "Ensemble",
        "variable": "ETof",
        "reducer": "mean",
        "reference_et": "gridMET",
        "units": "mm",
        "encrypt": False,
        "version": 2.0
    }

    # query the api
    resp = requests.post(
        headers=header,
        json=args,
        url=url
    )

    print(resp.json())

    try:
        return resp.json()['tracking_id'], resp.json()['name'], ''
    except KeyError:
        return 0, 0, resp.json()['detail']


def export_multipolygon_1(fields, start='2020-01-01', end='2024-12-31',
                          api_key=API_KEY):
    """ """
    # set your API key before making the request
    with open(api_key, 'r') as f:
        api_key = f.readline()
    header = {"Authorization": api_key}

    if ENDPOINT == 'montana':
        url = "https://openet-api-montana-ic5gyecbva-uw.a.run.app/raster/export/multipolygon"
    else:
        url = "https://openet-api.org/raster/export/multipolygon"

    # endpoint arguments
    args = {
        "date_range": [
            start,
            end
        ],
        "interval": "daily",
        "asset_id": fields,
        "attributes": ["Field"],
        "model": "Ensemble",
        "variable": "ET",
        "reducer": "mean",
        "reference_et": "gridMET",
        "units": "mm",
        "encrypt": False,
        "version": 2.1
    }

    # query the api
    resp = requests.post(
        headers=header,
        json=args,
        url=url
    )

    print(resp.json())

    try:
        return resp.json()['tracking_id'], resp.json()['name'], ''
    except KeyError:
        return 0, 0, resp.json()['detail']


def export_stack_rect(bounds, start='2020-01-01', end='2024-12-31', api_key=API_KEY):
    """ Export GeoTIFFs of OpenET data to Google Drive. Use rectangle as study area.

    Export is limited to 31 time steps per request. Each time step is a sepearate file.
    """
    # set your API key before making the request
    with open(api_key, 'r') as f:
        api_key = f.readline()
    header = {"Authorization": api_key}

    xmin, ymin, xmax, ymax = bounds  # tuple of length 4 in EPSG:4326
    geometry = [xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]
    print(geometry)

    if ENDPOINT == 'montana':
        url = "https://openet-api-montana-ic5gyecbva-uw.a.run.app/raster/export/stack"
    else:
        url = "https://openet-api.org/raster/export/stack"

    # endpoint arguments
    args = {
        "date_range": [
            start,
            end
        ],
        "interval": "monthly",
        "geometry": geometry,
        "model": "Ensemble",
        "variable": "ET",
        "reference_et": "gridMET",
        "units": "mm",
        "encrypt": False
    }

    # query the api
    resp = requests.post(
        headers=header,
        json=args,
        url=url
    )

    print(resp.json())


def timeseries_point(point, start='2024-01-01', end='2025-12-31', api_key=API_KEY):
    """ Uses OpenET's raster/timeseries/point endpoint.

    point: [lon, lat] location er
    """

    # set your API key before making the request
    with open(api_key, 'r') as f:
        api_key = f.readline()
    header = {"Authorization": api_key}

    if ENDPOINT == 'montana':
        url = "https://openet-api-montana-ic5gyecbva-uw.a.run.app/raster/timeseries/point"
    else:
        url = "https://openet-api.org/raster/timeseries/point"

    # endpoint arguments
    args = {
        "date_range": [
            start,
            end
        ],
        "interval": "daily",
        "geometry": point,
        "model": "Ensemble",
        "variable": "ET",
        "reference_et": "gridMET",
        "units": "mm",
        "file_format": "JSON"
    }

    # query the api
    resp = requests.post(
        headers=header,
        json=args,
        url=url
    )

    print(resp.json())


def gdb_ts(start="2024-01-01", end="2024-12-31", api_key=API_KEY):
    """Provides support for retrieving timeseries from OpenET's pre-computed database.

    Allows the user to define a list of field ids and export a subset of the OpenET geodatabase to retrieve
    timeseries data. Extractions will only include data within one US State at a time, however, multi-model and
    variable queries are supported in list format.

    For large queries, uncompressed csv output is not an option and json must be selected.

    NOTE: data in geodatabase are stored in metric (mm & hectares).
    """
    # set your API key before making the request
    with open(api_key, 'r') as f:
        api_key = f.readline()
    header = {"Authorization": api_key}

    if ENDPOINT == 'montana':
        url = "https://openet-api-montana-ic5gyecbva-uw.a.run.app/geodatabase/timeseries"
    else:
        url = "https://openet-api.org/geodatabase/timeseries"

    # endpoint arguments
    args = {
        "date_range": [
            start,
            end
        ],
        "interval": "daily",
        "field_ids": [
            "21130019358"  # GC alfalfa (NP and FCP), 21130019393 is upper terrace on west pivot
        ],
        "models": [
            "Ensemble"
        ],
        "variables": [
            "ET"
        ],
        "file_format": "JSON"
    }

    # query the api
    resp = requests.post(
        headers=header,
        json=args,
        url=url
    )

    # unzip the data
    try:
        data = eval(gzip.decompress(resp.content).decode())
        print(data)
    except gzip.BadGzipFile:
        print(resp.content)


if __name__ == '__main__':

    # Testing things

    # check()

    storage()  # returns all active links, even if they have been retrieved already.

    # gdb_ts()

    # upload(r"S:\Water_Management\Clark Fork\Gold Creek\Gold Creek Return Flow Study\ANALYSIS\gis\GoldCreekFields.geojson")

    # # # Worked on regular server, on MT server it can produce:
    # # # ImageCollection asset 'projects/openet/assets/ensemble/conus/gridmet/landsat/c02' not found
    # # # (does not exist or caller does not have access)
    # ee_fields = 'projects/ee-hehaugen/assets/GoldCreekFields1'
    # # export_multipolygon(ee_fields, start='2024-01-01', end='2024-12-31')
    # export_multipolygon_1(ee_fields, start='2025-01-01', end='2025-12-31')

    # track('IM75AXPWFLSNF5KPYEBFBE2G')  # running on MT server, 558e9

    # # Trying to get Evan's watersheds to work.

    # this is too big. 'Single query area limit exceeded. Region must not exceed 200000 acres.'
    # bnds = (-116.619478, 45.470972, -114.320293, 46.740637)  # largest watershed boundary in Evan's shapefile.

    # # what about this? Still too big? But I added a buffer!
    # bnds = [-116.23331945, 42.17297118, -115.82741815, 42.47022458]  # 28km grid
    # bnds = [-116.22040773, 42.13800772, -115.82924409, 42.42461931]  # 27km grid

    # # 'There are 0 rasters which meet the request criteria.' - Are they missing historical data? Looks like it.
    # # 2000-2001 didn't work, but 2022-2023 did
    # bnds = [-116.202862, 47.412161, -116.148276, 47.441068]  # the smallest watershed, to see if my code is wrong.
    #
    # export_stack_rect(bnds, '2022-01-01', '2023-12-31')

    # ----------------------------------------------

    # New procedure for getting data from OpenET:

    import csv

    logfile = r"F:\SWIM_SID\OpenET\OpenET_export_logging.csv"
    gis_dir = r"F:\SWIM_SID\statewide_irrigation_dataset_20240408\Cleaner_Counties"
    gis_dir = r"C:\Users\CND571\Downloads"

    # for exporting <10-year chunks of data:
    starts = ['1991-01-01', '2000-01-01', '2009-01-01', '2018-01-01']
    ends = ['1999-12-31', '2008-12-31', '2017-12-31', '2024-12-31']

    # [19, 33, 61, 101, 51, 41, 91, 53, 15, 93]  # first 10 counties
    # OpenET has a global rate limit for individual users of 20 per minute with a maximum of 500 per hour.
    # for county in [19]:
    for section in [163]:
        # print('\n', county)
        # gis_path = os.path.join(gis_dir, f'COUNTY_NO_{county}.geojson')
        gis_path = os.path.join(gis_dir, f"Section_{section}.geojson")  # I think these need to be in epsg:4326

        asset = upload(gis_path)
        # asset = 'https://storage.googleapis.com/openet-api-public/Hannah_Haugen_2433/9e553aeca4fe49169573f2c7c7ba59c3'
        # asset = 'https://storage.googleapis.com/openet-api-public/Hannah_Haugen_2433/7b23b397a3014a3ba6b9a4b71de6bf12'  # section 357a

        # Can I get the full record at once? No. Date range cannot exceed 10 years. :(
        # trackid, filename = export_multipolygon(asset, start='1991-01-01', end='2024-12-31')

        for s, e in zip(starts[::-1], ends[::-1]):
            # print(s, e)
            trackid, filename, message = export_multipolygon(asset, start=s, end=e)

            now = time.strftime("%x %X")  # date and time

            print(asset, trackid, filename)

            with open(logfile, mode='a', newline='') as log:
                writer = csv.writer(log)
                # writer.writerow([county, gis_path, asset, trackid, filename, s, e, now, 'submitted'])
                writer.writerow([section, gis_path, asset, trackid, filename, s, e, now, 'submitted', message])

        # asset = "mynewasset"
        # trackid, filename = 'ABCDEF123', 'ab123'

    # --------------------------------------------------

    # # How to use the timeseries endpoint. Actually returns data, requires immediate formatting.
    # gc_lys = {'FCP-LYS1': {'loc': [-112.910480, 46.573291], 'deg': '104'},
    #           'FCP-LYS2': {'loc': [-112.912107, 46.574847], 'deg': '22'},
    #           'NP-LYS1': {'loc': [-112.908331, 46.577076], 'deg': '212'},
    #           'NP-LYS2': {'loc': [-112.906849, 46.579788], 'deg': '3'}}
    # lys_et = []
    # for k, v in gc_lys.items():
    #     lys_et.append({'name': k, 'data': timeseries_point(v['loc'], '2024-01-01', '2025-10-01')})
    # lys_et = pd.json_normalize(lys_et, 'data', ['name'])
    # lys_et = lys_et.pivot(index='time', columns='name', values='et')
    # lys_et.index = pd.to_datetime(lys_et.index)
    # lys_et.to_csv(r"C:\Users\CND571\Downloads\gc_lys_OpenET_ensemble_et_daily_in_20240101_2025101.csv")

    # lys_et = lys_et['2025-01-01':'2025-09-30']
    # lys_et = lys_et.groupby(lys_et.index.month).sum()

    # ee_fields = 'projects/ee-hehaugen/assets/UpperYellowstoneBasin'
    # export_stack(ee_fields, start='2020-01-01', end='2022-06-30')
    # export_stack(ee_fields, start='2022-07-01', end='2024-12-31')

    # export_stack_rect(start='2024-06-01', end='2024-10-31')
    # export_stack_rect(start='2025-04-01', end='2025-05-31')

    # # Get the data from Openet to Google Drive
    # shp = '067_Park'  # all 1968 fields from 01/30/24 version of SID
    # # shp = 'mt_sid_uy10'  # smaller set of fields for testing
    # ee_fields = 'projects/ee-hehaugen/assets/{}'.format(shp)
    # openet_get_fields_export_1(ee_fields, "2020-01-01", "2024-12-31", attrs='fid')

    # gee_asset_1 = 'projects/ee-hehaugen/assets/067_Park_A'
    # gee_asset_2 = 'projects/ee-hehaugen/assets/067_Park_B'
    # openet_get_fields_export(gee_asset_1, "2020-01-01", "2024-12-31", attrs='fid')
    # openet_get_fields_export(gee_asset_2, "2020-01-01", "2024-12-31", attrs='fid')

    # # 99 fields per file works! Maybe, kinda.
    # # tried running these again, with existence checking. The memory failure is pretty reliable.
    # for i in range(1, 20):
    #     print("{}:".format(i))
    #     loc = 'F:/BOR_UYWS_2025/swim/ssebop_etof_park_{}.csv'.format(i)
    #     if os.path.exists(loc):
    #         print("{} exists, skipping".format(loc))
    #     else:
    #         gee_asset = 'projects/ee-hehaugen/assets/group_{}'.format(i)
    #         openet_get_fields_export(gee_asset, "2020-01-01", "2024-12-31", attrs='fid')

    # # Same deal as above, just less printing.
    # for i in range(1, 20):
    #     loc = 'F:/BOR_UYWS_2025/swim/ssebop_etof_park_{}.csv'.format(i)
    #     if not os.path.exists(loc):
    #         print("{}:".format(i))
    #         gee_asset = 'projects/ee-hehaugen/assets/group_{}'.format(i)
    #         openet_get_fields_export(gee_asset, "2020-01-01", "2024-12-31", attrs='fid')

    import geopandas as gpd

    # park_incomp = gpd.read_file("C:/Users/CND571/Documents/Data/sid_30JAN2024/Park_incomplete_12/park_incomplete.shp")

    # # All of these downloads worked!
    # for i in park_incomp['group'].unique():
    #     print("{}:".format(i))
    #     gee_asset = 'projects/ee-hehaugen/assets/park_incomplete/group_{}'.format(i)
    #     openet_get_fields_export(gee_asset, "2020-01-01", "2024-12-31", attrs='fid')

    # # Come back and concat all the smaller files.
    # groups = []
    # for i in [0, 1, 5, 9, 10, 11, 13, 15, 17, 18]:
    #     this_group = pd.read_csv('F:/BOR_UYWS_2025/swim/ssebop_etof_park_{}.csv'.format(i))
    #     groups.append(this_group)
    # for i in FILES.values():
    #     this_group = pd.read_csv('F:/BOR_UYWS_2025/swim/ssebop_etof_{}.csv'.format(i['name']))
    #     groups.append(this_group)
    # all_groups = pd.concat(groups)
    # all_groups.to_csv("F:/BOR_UYWS_2025/swim/uy_all_ssebop_etof.csv", index=False)
    # # print(all_groups)  # I don't need to sort by FID right? What is the format for things?

    # old_start, old_end = "1985-01-01", "1989-12-31"
    # old_start, old_end = "1985-01-01", "2015-12-31"
    # old_start, old_end = "1985-01-01", "2015-12-31"

    # gee_asset = 'projects/ee-hehaugen/assets/059_Meagher_subset_20250317'
    # for year in range(1985, 2014, 5):
    #     old_start, old_end = f"{year}-01-01", f"{year+4}-12-31"
    #     # print(old_start, old_end)
    #     openet_get_fields_export_1(gee_asset, old_start, old_end, et_too=True)
    # old_start, old_end = "2015-01-01", "2015-12-31"
    # openet_get_fields_export_1(gee_asset, old_start, old_end, et_too=True)
    # new_start, new_end = "2016-01-01", "2023-12-31"
    # openet_get_fields_export_1(gee_asset, new_start, new_end, et_too=True)

    # ets = ['2bc40', '6c8b7', '487ef', '1611a', 'd9e5a', 'd164a', 'e79d1', 'fa588']
    # etofs = ['9e58d', '77d7e', '85bf3', '193d2', '292fc', '402c9', 'ca1cf', 'f0827']
    #
    # et_dfs = []
    # etof_dfs = []
    # for i in range(8):
    #     et = pd.read_csv(f'C:/Users/CND571/Downloads/ensemble_et_{ets[i]}.csv')
    #     etof = pd.read_csv(f'C:/Users/CND571/Downloads/ensemble_etof_{etofs[i]}.csv')
    #     et_dfs.append(et)
    #     etof_dfs.append(etof)
    # et_df = pd.concat(et_dfs)
    # etof_df = pd.concat(etof_dfs)
    # et_df['time'] = pd.to_datetime(et_df['time'])
    # etof_df['time'] = pd.to_datetime(etof_df['time'])
    # et_df = et_df.sort_values('time')
    # etof_df = etof_df.sort_values('time')
    # et_ind = pd.MultiIndex.from_frame(et_df[['time', 'FID']])
    # et_df.index = et_ind
    # et_df = et_df.drop(columns=['time', 'FID'])
    # etof_df.index = et_ind  # should be the same
    # etof_df = etof_df.drop(columns=['time', 'FID'])
    # # print(et_df)
    # # print(etof_df)
    #
    # opnt = pd.concat([et_df, etof_df], axis=1)
    # print(opnt)
    #
    # opnt.to_csv('C:/Users/CND571/Documents/Data/059_Meagher_openet_1985_2023.csv')
    #
    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # one_field = opnt.xs('059_000585', level='FID')
    # plt.plot(one_field.index, one_field['etof'])
    # plt.show()

# ========================= EOF ====================================================================
