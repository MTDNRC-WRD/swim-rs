import os.path

import numpy as np
import pandas as pd
import requests

FILES = {
    99: {'tracking_id': 'GB7QTW36UASFOYGLP6LFTWTL', 'encrypted': False, 'name': '40fa2', 'destination': 'drive'},
    100: {'tracking_id': 'RTWV2VMUBE6XU6I2UHQRKELU', 'encrypted': False, 'name': 'f70f0', 'destination': 'drive'},
    101: {'tracking_id': 'HZIW33WT3TDQY4QK7UZBDQFH', 'encrypted': False, 'name': 'd4bc3', 'destination': 'drive'},
    102: {'tracking_id': 'HDOGJZFFYB4UTJ57EJKX5A3B', 'encrypted': False, 'name': 'e4077', 'destination': 'drive'},
    103: {'tracking_id': 'IU2E5FP3PZPAEMTT5HIJVYP5', 'encrypted': False, 'name': 'b8d06', 'destination': 'drive'},
    104: {'tracking_id': 'WCMDDX3KGC7HNCOY7XFS6O7F', 'encrypted': False, 'name': 'af989', 'destination': 'drive'},
    105: {'tracking_id': 'XDNAUNYON5RL4O636MWZLIZW', 'encrypted': False, 'name': 'd5a04', 'destination': 'drive'},
    106: {'tracking_id': 'GOS6T6RZA22SEGWIZVBFIH3N', 'encrypted': False, 'name': '67ef0', 'destination': 'drive'},
    107: {'tracking_id': 'UNDNX73JXYIXXVDLJJV4EQE3', 'encrypted': False, 'name': 'c7f04', 'destination': 'drive'},
    115: {'tracking_id': 'IIDNEUK65377Z2S3ZXU6YJRV', 'encrypted': False, 'name': '81128', 'destination': 'drive'},
    116: {'tracking_id': 'ADNXBMUZWEHJ3YRLQKLG4QIR', 'encrypted': False, 'name': 'cb40a', 'destination': 'drive'},
    117: {'tracking_id': '46CJ4W5HKFNXJKRBMSDLGD6S', 'encrypted': False, 'name': '58593', 'destination': 'drive'},
    118: {'tracking_id': 'XX2LWYR5NJQFNV7ZEVCJE7XW', 'encrypted': False, 'name': 'f905b', 'destination': 'drive'},
    119: {'tracking_id': 'P4XXERHI43CUOG4YA6NJIRJQ', 'encrypted': False, 'name': '16008', 'destination': 'drive'},
    120: {'tracking_id': '4H6UV2Y65Y5W7ET3EKDEBHGH', 'encrypted': False, 'name': 'f4a8a', 'destination': 'drive'},
    121: {'tracking_id': 'MAXSQZYNTKUR4WZGPYNUFL63', 'encrypted': False, 'name': '09856', 'destination': 'drive'},
    122: {'tracking_id': 'BMFKNNLMOEFZ6T424HEU76BB', 'encrypted': False, 'name': 'd949f', 'destination': 'drive'},
    123: {'tracking_id': 'GJ3ETD6DS2IFTMMMJ2WGUDM3', 'encrypted': False, 'name': '6e47d', 'destination': 'drive'},
    132: {'tracking_id': 'FOKMU4HA2CUJHROITBXNL2Q3', 'encrypted': False, 'name': '6bdfc', 'destination': 'drive'},
    133: {'tracking_id': 'RAYWGOHUSIO7ZWIBDGC6IAUP', 'encrypted': False, 'name': '6403b', 'destination': 'drive'},
    134: {'tracking_id': 'PYROONNGQ3OVQYM4O6BKFSFB', 'encrypted': False, 'name': '23439', 'destination': 'drive'},
    135: {'tracking_id': 'HTAQBDZ6OKJSFAOEQ4WMDDAK', 'encrypted': False, 'name': 'f9c24', 'destination': 'drive'},
    136: {'tracking_id': 'GQNTG3NW43UZQZ36FGFWFMI4', 'encrypted': False, 'name': '150bd', 'destination': 'drive'},
    137: {'tracking_id': 'PD5BZMBOXORNX6JLXDYJ3STM', 'encrypted': False, 'name': '62a13', 'destination': 'drive'},
    138: {'tracking_id': 'JNVLUSSMVIGIXF3CEUHVNDAN', 'encrypted': False, 'name': '25644', 'destination': 'drive'},
    139: {'tracking_id': 'OD7J5QQBDFIMZLPPPL6NVDZU', 'encrypted': False, 'name': 'ade37', 'destination': 'drive'},
    140: {'tracking_id': 'T4YAHP4EBLT22P4AX5TJWT7S', 'encrypted': False, 'name': '71f50', 'destination': 'drive'},
    156: {'tracking_id': '3J6XMHADJUT3HEF4RWA3FDKY', 'encrypted': False, 'name': '9cebc', 'destination': 'drive'},
    157: {'tracking_id': 'SRNBFNMPG3WVN7I5322DKFDJ', 'encrypted': False, 'name': 'ac0c6', 'destination': 'drive'},
    158: {'tracking_id': 'BFYIEV3LYMY6NPJ7C2TAM7ZV', 'encrypted': False, 'name': '12462', 'destination': 'drive'},
    159: {'tracking_id': 'CMX2KEYOTBLQH352AW7CMAYQ', 'encrypted': False, 'name': '4284d', 'destination': 'drive'},
    160: {'tracking_id': '4VPAZ776A4VNLAEGD3C3XBQH', 'encrypted': False, 'name': '35d2f', 'destination': 'drive'},
    161: {'tracking_id': 'D5CBQ5Z6MAHXC3WJY3DC2MB6', 'encrypted': False, 'name': '8f8c8', 'destination': 'drive'},
    162: {'tracking_id': 'RS2W7QMCQMANIGXZ3YLEP2TF', 'encrypted': False, 'name': '2f1ad', 'destination': 'drive'},
    163: {'tracking_id': '4OENS76L3OAG42ZHBBKB3UTQ', 'encrypted': False, 'name': 'fd7b3', 'destination': 'drive'},
    164: {'tracking_id': 'E3FHCK5HAZQR3IUDSEAF3T5M', 'encrypted': False, 'name': 'b8057', 'destination': 'drive'},
    16: {'tracking_id': 'Z7VSQJQ7GIK4PVGDSTQ6VSCM', 'encrypted': False, 'name': 'e891e', 'destination': 'drive'},
    17: {'tracking_id': 'C7WCDGMTG2UC25OS6ZIUWVYX', 'encrypted': False, 'name': 'c5f44', 'destination': 'drive'},
    18: {'tracking_id': 'TJRF4HBWMEWRETUYKKYJG3AK', 'encrypted': False, 'name': 'a121e', 'destination': 'drive'},
    19: {'tracking_id': '43Q3X3MU6NBHEIA3LSF5PTKO', 'encrypted': False, 'name': 'b918a', 'destination': 'drive'},
    20: {'tracking_id': 'IKTUEKTZOM2AYON65HVBRKFB', 'encrypted': False, 'name': 'f6c9d', 'destination': 'drive'},
    21: {'tracking_id': 'P4CM7VEMXWM25D3H2OFHMN4I', 'encrypted': False, 'name': 'd8a0e', 'destination': 'drive'},
    22: {'tracking_id': '4E6ASZDQWZ6EREQXP7NVJBNH', 'encrypted': False, 'name': '77d50', 'destination': 'drive'},
    23: {'tracking_id': 'NBRUS67LH3SRIOPTHCBDC5Z5', 'encrypted': False, 'name': '89e4f', 'destination': 'drive'},
    24: {'tracking_id': 'SKHVHM25XMDVB3GALQZZ2RXI', 'encrypted': False, 'name': '1dcbe', 'destination': 'drive'},
    25: {'tracking_id': 'KLZ4IH64LZYL7NJVDSCGO3QL', 'encrypted': False, 'name': '17d8d', 'destination': 'drive'},
    26: {'tracking_id': 'WUD57QKWX7MQUPRW2GU3K7ST', 'encrypted': False, 'name': '67b48', 'destination': 'drive'},
    27: {'tracking_id': 'TO5OMOAIFAQV4C3BDRV73GGC', 'encrypted': False, 'name': '1342d', 'destination': 'drive'},
    28: {'tracking_id': '5BQ5Y5VEODELQLIG2XJJ6PLQ', 'encrypted': False, 'name': 'bead4', 'destination': 'drive'},
    29: {'tracking_id': 'N5L73WIFGCROTQ56ZPJGGDGI', 'encrypted': False, 'name': '1026d', 'destination': 'drive'},
    30: {'tracking_id': 'SHAM2JSG7SUFZN6UUVSVD577', 'encrypted': False, 'name': '343a6', 'destination': 'drive'},
    31: {'tracking_id': 'I2HRHHXUZWHLIPBDEV62VFH2', 'encrypted': False, 'name': '60486', 'destination': 'drive'},
    32: {'tracking_id': 'BZU4GYFVK5WZWETU6OX6YSHS', 'encrypted': False, 'name': '41ea5', 'destination': 'drive'},
    33: {'tracking_id': 'PHKKZMN3PFDEX5PD4FYVDM6U', 'encrypted': False, 'name': '94089', 'destination': 'drive'},
    34: {'tracking_id': 'NL4ZLXG2ZZFXM6JB3DHNMBT6', 'encrypted': False, 'name': '6f61e', 'destination': 'drive'},
    35: {'tracking_id': 'FRK3GVLW6ID4LIYTOADBWGK4', 'encrypted': False, 'name': '80991', 'destination': 'drive'},
    36: {'tracking_id': 'BAB7FUDRKLG4CG5ZQITMZPH7', 'encrypted': False, 'name': '57893', 'destination': 'drive'},
    37: {'tracking_id': 'UDZ5SJUGP6ND53IDD4AJ3NEZ', 'encrypted': False, 'name': '0389a', 'destination': 'drive'},
    38: {'tracking_id': 'CF72S37ZOO7ZDYHA7XMURRSA', 'encrypted': False, 'name': '04add', 'destination': 'drive'},
    39: {'tracking_id': 'ON7P76KZROG6KUVLC6FM6IGT', 'encrypted': False, 'name': 'ba9e7', 'destination': 'drive'},
    40: {'tracking_id': 'HMKOQJT45FKLMLNYIKMQQR7Y', 'encrypted': False, 'name': 'e9a30', 'destination': 'drive'},
    41: {'tracking_id': '4MFYOCG63GQAHQA7KGSSZ7CO', 'encrypted': False, 'name': 'd9538', 'destination': 'drive'},
    49: {'tracking_id': 'PV66UCSPMNSJYNJL7PMPBNRD', 'encrypted': False, 'name': '6b47d', 'destination': 'drive'},
    50: {'tracking_id': 'QQ5RSZQLCLFEYCDTRLUC374B', 'encrypted': False, 'name': 'c29db', 'destination': 'drive'},
    51: {'tracking_id': '4RLNOPEN5Z4HKFZSLPA57LQK', 'encrypted': False, 'name': '067c7', 'destination': 'drive'},
    52: {'tracking_id': 'K7QI3MKIZBZTHMG72KUGSGSX', 'encrypted': False, 'name': '6f0c5', 'destination': 'drive'},
    53: {'tracking_id': 'OSWDSYUQFNXBNM3MZHB7ID33', 'encrypted': False, 'name': '237bc', 'destination': 'drive'},
    54: {'tracking_id': 'PTWHBSPV626X25EKD5AGC5Y4', 'encrypted': False, 'name': 'c8c63', 'destination': 'drive'},
    55: {'tracking_id': 'VHEDAEDEYXIELECLAM7VAAZN', 'encrypted': False, 'name': 'd1b88', 'destination': 'drive'},
    56: {'tracking_id': 'E6YEQDLSWEQGXDK7H6HC7GHC', 'encrypted': False, 'name': '844ba', 'destination': 'drive'},
    57: {'tracking_id': 'B4EK2GBAS5QAIKAZJC65V3UB', 'encrypted': False, 'name': 'c6938', 'destination': 'drive'},
    58: {'tracking_id': 'SCB55OQPNDTV2NMTKOVWLMLE', 'encrypted': False, 'name': '26666', 'destination': 'drive'},
    59: {'tracking_id': 'KH6UNK6LEF7HKB7RR3VHU5GP', 'encrypted': False, 'name': 'd7f75', 'destination': 'drive'},
    60: {'tracking_id': 'NZ4KVUPYIVBKCQ2GBPP3Q7DT', 'encrypted': False, 'name': 'd9f07', 'destination': 'drive'},
    61: {'tracking_id': 'OH4OKDHOSLGOOM63ILHP63IF', 'encrypted': False, 'name': '48a36', 'destination': 'drive'},
    62: {'tracking_id': 'JXBJ23RQYCQ6HMXNFPAAFALO', 'encrypted': False, 'name': '41c30', 'destination': 'drive'},
    63: {'tracking_id': 'QCBXUTB6CCDY3AWQ6YDSWJO2', 'encrypted': False, 'name': '76955', 'destination': 'drive'},
    64: {'tracking_id': 'I5TTJXKWWBWJPWPLWPCE5IKJ', 'encrypted': False, 'name': 'dadf3', 'destination': 'drive'},
    65: {'tracking_id': 'TXI4EARFSUHE5SRVXFPBC76F', 'encrypted': False, 'name': '4a18b', 'destination': 'drive'},
    66: {'tracking_id': 'JUIPIDE2Z753WALRHVKJENO4', 'encrypted': False, 'name': 'dbaf7', 'destination': 'drive'},
    67: {'tracking_id': 'JQTTJXB3ERI3T6A3AOEPM7MM', 'encrypted': False, 'name': '286d4', 'destination': 'drive'},
    68: {'tracking_id': 'JXSCRPFGEI3IZY524W7Z2WZA', 'encrypted': False, 'name': '7a84c', 'destination': 'drive'},
    69: {'tracking_id': 'QOIWWCGG2TJ3MIMMK3GORJYM', 'encrypted': False, 'name': 'adecb', 'destination': 'drive'},
    70: {'tracking_id': 'MB3TSYGXJPYFEZB5DVWVBVUT', 'encrypted': False, 'name': '23e48', 'destination': 'drive'},
    71: {'tracking_id': 'JP6LOFY7ECZLWNH2SMQUAAPQ', 'encrypted': False, 'name': 'a4b7f', 'destination': 'drive'},
    72: {'tracking_id': '4OQNKVUKPITB4NAD3OAG7UOP', 'encrypted': False, 'name': '0a080', 'destination': 'drive'},
    73: {'tracking_id': 'R6BC6QCSGZ4TXRI6CBZDLDQ7', 'encrypted': False, 'name': '176c8', 'destination': 'drive'},
    74: {'tracking_id': 'FAD3RN43ISRZNJ7BIJ32AJNI', 'encrypted': False, 'name': '515a0', 'destination': 'drive'}
}


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


def openet_get_fields_export_1(fields, start, end, attrs='FID', et_too=False, show=False,
                               api_key='C:/Users/CND571/Documents/Haugen_Montana_API.txt'):
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
    # shp = 'mt_sid_uy10'  # smaller set of fields for testing
    # ee_fields = 'projects/ee-hehaugen/assets/{}'.format(shp)
    # openet_get_fields_export(ee_fields, "2020-01-01", "2024-12-31")  # This works! Now how to do more fields?

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
