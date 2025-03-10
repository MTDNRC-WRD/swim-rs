import pandas as pd
import requests


def openet_get_fields_export(fields, start, end, attrs='FID', et_too=False, show=False,
                             api_key='C:/Users/CND571/Documents/OpenET_API.txt'):
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


def track(track_id, api_key='C:/Users/CND571/Documents/Haugen_Montana_API.txt'):
    # set your API key before making the request
    with open(api_key, 'r') as f:
        api_key = f.readline()
    header = {"Authorization": api_key}

    # endpoint arguments
    args = {
        "tracking_id": track_id
    }

    # query the api
    resp = requests.get(
        headers=header,
        params=args,
        url="https://openet-api.org/raster/export/track"
    )

    print(resp.json())


def upload(filepath, api_key='C:/Users/CND571/Documents/OpenET_API.txt'):
    import requests

    # set your API key before making the request
    header = {"Authorization": api_key}

    # endpoint arguments
    args = {
        'file': (filepath, open(filepath, 'rb'), 'application/geo+json')
    }

    # query the api
    resp = requests.post(
        headers=header,
        files=args,
        url="https://openet-api-montana-ic5gyecbva-uw.a.run.app/account/upload"
        # url="https://openet-api.org/account/upload"
    )

    print(resp.json())


def check(api_key='C:/Users/CND571/Documents/OpenET_API.txt'):
    # set your API key before making the request
    with open(api_key, 'r') as f:
        api_key = f.readline()
    header = {"Authorization": api_key}

    # query the api
    resp = requests.get(
        headers=header,
        url="https://openet-api.org/account/status"  # both urls read the same thing.
        # url="https://openet-api-montana-ic5gyecbva-uw.a.run.app/account/status"
    )

    print(resp.json())


if __name__ == '__main__':
    # Get the data from Openet to Google Drive
    # shp = '067_Park'  # all 1968 fields from 01/30/24 version of SID
    shp = 'mt_sid_uy10'  # smaller set of fields for testing
    ee_fields = 'projects/ee-hehaugen/assets/{}'.format(shp)
    # openet_get_fields_export(ee_fields, "2020-01-01", "2024-12-31")  # This works! Now how to do more fields?

    # gee_asset_1 = 'projects/ee-hehaugen/assets/067_Park_A'
    # gee_asset_2 = 'projects/ee-hehaugen/assets/067_Park_B'
    # openet_get_fields_export(gee_asset_1, "2020-01-01", "2024-12-31", attrs='fid')
    # openet_get_fields_export(gee_asset_2, "2020-01-01", "2024-12-31", attrs='fid')

    # 99 fields per file works!
    for i in range(1, 20):
        gee_asset = 'projects/ee-hehaugen/assets/group_{}'.format(i)
        print("{}:".format(i))
        openet_get_fields_export(gee_asset, "2020-01-01", "2024-12-31", attrs='fid')

    # upload("C:/Users/CND571/Documents/Data/sid_30JAN2024/Park_20_geojson/group_0.geojson")

    # track()
    # track()

    # check()  # Do my fancy credentials just mean that I can do whatever, and these won't increment?

    # Come back and concat all the smaller files.
    groups = []
    for i in range(20):
        this_group = pd.read_csv('F:/BOR_UYWS_2025/swim/ssebop_etof_park_{}.csv'.format(i))
        groups.append(this_group)
    all_groups = pd.concat(groups)
    all_groups.to_csv("F:/BOR_UYWS_2025/swim/uy_all_ssebop_etof.csv")

    print(all_groups)


# ========================= EOF ====================================================================
