import os
import sys

import ee
import pandas as pd
from datetime import datetime as dt
import geopandas as gpd
import xarray
from tqdm import tqdm

from data_extraction.ee.ee_utils import is_authorized

sys.path.insert(0, os.path.abspath('../..'))
sys.setrecursionlimit(5000)

IRR = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'

ETF = 'projects/usgs-gee-nhm-ssebop/assets/ssebop/landsat/c02'

EC_POINTS = 'users/dgketchum/fields/flux'

STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']


def get_flynn():
    return ee.FeatureCollection(ee.Feature(ee.Geometry.Polygon([[-106.63372199162623, 46.235698473362476],
                                                                [-106.49124304875514, 46.235698473362476],
                                                                [-106.49124304875514, 46.31472036075997],
                                                                [-106.63372199162623, 46.31472036075997],
                                                                [-106.63372199162623, 46.235698473362476]]),
                                           {'key': 'Flynn_Ex'}))


def export_etf_images(feature_coll, year=2015, bucket=None, debug=False, mask_type='irr'):
    s, e = '1987-01-01', '2021-12-31'
    irr_coll = ee.ImageCollection(IRR)
    coll = irr_coll.filterDate(s, e).select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(5)
    irr = irr_coll.filterDate('{}-01-01'.format(year),
                              '{}-12-31'.format(year)).select('classification').mosaic()
    irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))

    coll = ee.ImageCollection(ETF).filterDate('{}-01-01'.format(year), '{}-12-31'.format(year))
    coll = coll.filterBounds(feature_coll)
    scenes = coll.aggregate_histogram('system:index').getInfo()

    for img_id in scenes:

        splt = img_id.split('_')
        _name = '_'.join(splt[-3:])

        img = ee.Image(os.path.join(ETF, img_id))

        if mask_type == 'no_mask':
            img = img.clip(feature_coll.geometry()).int()
        elif mask_type == 'irr':
            img = img.clip(feature_coll.geometry()).mask(irr_mask).int()
        elif mask_type == 'inv_irr':
            img = img.clip(feature_coll.geometry()).mask(irr.gt(0)).int()

        if debug:
            point = ee.Geometry.Point([-106.576, 46.26])
            data = img.sample(point, 30).getInfo()
            print(data['features'])

        task = ee.batch.Export.image.toCloudStorage(
            img,
            description='ETF_{}_{}'.format(mask_type, _name),
            bucket=bucket,
            region=feature_coll.geometry(),
            crs='EPSG:5070',
            scale=30)

        task.start()
        print(_name)


def sparse_sample_etf(shapefile, bucket=None, debug=False, mask_type='irr', check_dir=None, feature_id='FID',
                      select=None, start_yr=2000, end_yr=2024):
    df = gpd.read_file(shapefile)
    df.index = df[feature_id]

    assert df.crs.srs == 'EPSG:5071'

    df = df.to_crs(epsg=4326)

    s, e = '1987-01-01', '2021-12-31'
    irr_coll = ee.ImageCollection(IRR)
    coll = irr_coll.filterDate(s, e).select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(5)

    for fid, row in df.iterrows():

        for year in range(start_yr, end_yr + 1):

            if select is not None and fid not in select:
                continue

            state = row['field_3']
            if state not in STATES:
                continue

            site = row[feature_id]

            desc = 'etf_{}_{}_{}'.format(site, mask_type, year)
            if check_dir:
                f = os.path.join(check_dir, '{}.csv'.format(desc))
                if os.path.exists(f):
                    print(desc, 'exists, skipping')
                    continue

            irr = irr_coll.filterDate('{}-01-01'.format(year),
                                      '{}-12-31'.format(year)).select('classification').mosaic()
            irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))

            point = ee.Geometry.Point([row['field_8'], row['field_7']])
            geo = point.buffer(150.)
            fc = ee.FeatureCollection(ee.Feature(geo, {feature_id: site}))

            etf_coll = ee.ImageCollection(ETF).filterDate('{}-01-01'.format(year),
                                                          '{}-12-31'.format(year))
            etf_coll = etf_coll.filterBounds(geo)
            etf_scenes = etf_coll.aggregate_histogram('system:index').getInfo()

            first, bands = True, None
            selectors = [site]

            for img_id in etf_scenes:

                splt = img_id.split('_')
                _name = '_'.join(splt[-3:])

                selectors.append(_name)

                etf_img = ee.Image(os.path.join(ETF, img_id)).rename(_name)
                etf_img = etf_img.divide(10000)

                if mask_type == 'no_mask':
                    etf_img = etf_img.clip(fc.geometry())
                elif mask_type == 'irr':
                    etf_img = etf_img.clip(fc.geometry()).mask(irr_mask)
                elif mask_type == 'inv_irr':
                    etf_img = etf_img.clip(fc.geometry()).mask(irr.gt(0))

                if first:
                    bands = etf_img
                    first = False
                else:
                    bands = bands.addBands([etf_img])

                if debug:
                    data = etf_img.sample(fc, 30).getInfo()
                    print(data['features'])

            data = bands.reduceRegions(collection=fc,
                                       reducer=ee.Reducer.mean(),
                                       scale=30)

            task = ee.batch.Export.table.toCloudStorage(
                data,
                description=desc,
                bucket=bucket,
                fileNamePrefix=desc,
                fileFormat='CSV',
                selectors=selectors)

            task.start()
            print(desc)


def clustered_sample_etf(feature_coll, bucket=None, debug=False, mask_type='irr', check_dir=None,
                         start_yr=2000, end_yr=2024, feature_id='FID'):

    feature_coll = ee.FeatureCollection(feature_coll)

    s, e = '1987-01-01', '2021-12-31'
    irr_coll = ee.ImageCollection(IRR)
    coll = irr_coll.filterDate(s, e).select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(5)

    for year in range(start_yr, end_yr + 1):

        irr = irr_coll.filterDate('{}-01-01'.format(year),
                                  '{}-12-31'.format(year)).select('classification').mosaic()
        irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))

        desc = 'etf_{}_{}'.format(mask_type, year)

        if check_dir:
            f = os.path.join(check_dir, '{}.csv'.format(desc))
            if os.path.exists(f):
                print(desc, 'exists, skipping')
                continue

        coll = ee.ImageCollection(ETF).filterDate('{}-01-01'.format(year), '{}-12-31'.format(year))
        coll = coll.filterBounds(feature_coll)
        scenes = coll.aggregate_histogram('system:index').getInfo()

        first, bands = True, None
        selectors = [feature_id]

        for img_id in scenes:

            # if img_id != 'lt05_036029_20000623':
            #     continue

            splt = img_id.split('_')
            _name = '_'.join(splt[-3:])

            selectors.append(_name)

            etf_img = ee.Image(os.path.join(ETF, img_id)).rename(_name)
            etf_img = etf_img.divide(10000)

            if mask_type == 'no_mask':
                etf_img = etf_img.clip(feature_coll.geometry())
            elif mask_type == 'irr':
                etf_img = etf_img.clip(feature_coll.geometry()).mask(irr_mask)
            elif mask_type == 'inv_irr':
                etf_img = etf_img.clip(feature_coll.geometry()).mask(irr.gt(0))

            if first:
                bands = etf_img
                first = False
            else:
                bands = bands.addBands([etf_img])

            if debug:
                point = ee.Geometry.Point([-107.188225, 44.9011])
                data = etf_img.sample(point, 30).getInfo()
                print(data['features'])

        # TODO extract pixel count to filter data
        data = bands.reduceRegions(collection=feature_coll,
                                   reducer=ee.Reducer.mean(),
                                   scale=30)

        task = ee.batch.Export.table.toCloudStorage(
            data,
            description=desc,
            bucket=bucket,
            fileNamePrefix=desc,
            fileFormat='CSV',
            selectors=selectors)

        task.start()
        print(desc)


def clustered_sample_etf_direct(feature_coll, dest_dir, debug=False, mask_type='irr',
                                start_yr=2000, end_yr=2024, feature_id='FID', drops=None):
    """ Process GEE SEEBOP etf data and save to local csv file.

    Combined behavior of clustered_sample_etf and list_and_copy_gcs_bucket"""
    feature_coll = ee.FeatureCollection(feature_coll)

    s, e = '1987-01-01', '2021-12-31'
    irr_coll = ee.ImageCollection(IRR)
    coll = irr_coll.filterDate(s, e).select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(5)

    for year in range(start_yr, end_yr + 1):

        irr = irr_coll.filterDate('{}-01-01'.format(year),
                                  '{}-12-31'.format(year)).select('classification').mosaic()
        irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))

        desc = 'etf_{}_{}'.format(mask_type, year)

        # Check that file has not already been created.
        f = os.path.join(dest_dir, '{}.csv'.format(desc))
        if os.path.exists(f):
            print(desc, 'exists, skipping')
            continue

        coll = ee.ImageCollection(ETF).filterDate('{}-01-01'.format(year), '{}-12-31'.format(year))
        coll = coll.filterBounds(feature_coll)
        scenes = coll.aggregate_histogram('system:index').getInfo()

        first, bands = True, None
        selectors = [feature_id]

        for img_id in scenes:

            # if img_id != 'lt05_036029_20000623':
            #     continue

            splt = img_id.split('_')
            _name = '_'.join(splt[-3:])

            selectors.append(_name)

            full_path = '{}/{}'.format(ETF, img_id)
            etf_img = ee.Image(full_path).rename(_name)  # fixed slash
            etf_img = etf_img.divide(10000)

            if mask_type == 'no_mask':
                etf_img = etf_img.clip(feature_coll.geometry())
            elif mask_type == 'irr':
                etf_img = etf_img.clip(feature_coll.geometry()).mask(irr_mask)
            elif mask_type == 'inv_irr':
                etf_img = etf_img.clip(feature_coll.geometry()).mask(irr.gt(0))

            if first:
                bands = etf_img
                first = False
            else:
                bands = bands.addBands([etf_img])

            if debug:
                point = ee.Geometry.Point([-107.188225, 44.9011])
                data = etf_img.sample(point, 30).getInfo()
                print(data['features'])

        # TODO extract pixel count to filter data
        data = bands.reduceRegions(collection=feature_coll,
                                   reducer=ee.Reducer.mean(),
                                   scale=30)

        data_df = ee.data.computeFeatures({
            'expression': data,
            'fileFormat': 'PANDAS_DATAFRAME'
        })

        print(desc)
        # Drop all columns that are not FID or a landsat image.
        data_df.index = data_df[feature_id]
        if drops:
            drops.append('geo')
            data_df.drop(columns=drops, inplace=True, errors='ignore')
        # print(data_df.head())
        data_df.to_csv(f)


def clustered_sample_etf_direct_1(feature_coll, debug=False, mask_type='irr',
                                  start_yr=2000, end_yr=2024, feature_id='FID', drops=None):
    """ Process GEE SEEBOP etf data and return as pd df.

    Combined behavior of clustered_sample_etf and list_and_copy_gcs_bucket"""
    feature_coll = ee.FeatureCollection(feature_coll)

    s, e = '1987-01-01', '2021-12-31'
    irr_coll = ee.ImageCollection(IRR)
    coll = irr_coll.filterDate(s, e).select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(5)

    dfs = []
    print('etf_{} {}-{}:'.format(mask_type, start_yr, end_yr))
    for year in tqdm(range(start_yr, end_yr + 1), total=end_yr - start_yr):

        irr = irr_coll.filterDate('{}-01-01'.format(year),
                                  '{}-12-31'.format(year)).select('classification').mosaic()
        irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))

        desc = 'etf_{}_{}'.format(mask_type, year)

        # # Check that file has not already been created.
        # f = os.path.join(dest_dir, '{}.csv'.format(desc))
        # if os.path.exists(f):
        #     print(desc, 'exists, skipping')
        #     continue

        coll = ee.ImageCollection(ETF).filterDate('{}-01-01'.format(year), '{}-12-31'.format(year))
        coll = coll.filterBounds(feature_coll)
        scenes = coll.aggregate_histogram('system:index').getInfo()

        first, bands = True, None
        selectors = [feature_id]

        for img_id in scenes:

            # if img_id != 'lt05_036029_20000623':
            #     continue

            splt = img_id.split('_')
            _name = '_'.join(splt[-3:])

            selectors.append(_name)

            full_path = '{}/{}'.format(ETF, img_id)
            etf_img = ee.Image(full_path).rename(_name)  # fixed slash
            etf_img = etf_img.divide(10000)

            if mask_type == 'no_mask':
                etf_img = etf_img.clip(feature_coll.geometry())
            elif mask_type == 'irr':
                etf_img = etf_img.clip(feature_coll.geometry()).mask(irr_mask)
            elif mask_type == 'inv_irr':
                etf_img = etf_img.clip(feature_coll.geometry()).mask(irr.gt(0))

            if first:
                bands = etf_img
                first = False
            else:
                bands = bands.addBands([etf_img])

            if debug:
                point = ee.Geometry.Point([-107.188225, 44.9011])
                data = etf_img.sample(point, 30).getInfo()
                print(data['features'])

        # TODO extract pixel count to filter data
        data = bands.reduceRegions(collection=feature_coll,
                                   reducer=ee.Reducer.mean(),
                                   scale=30)

        data_df = ee.data.computeFeatures({
            'expression': data,
            'fileFormat': 'PANDAS_DATAFRAME'
        })

        # print(desc)
        # Drop all columns that are not FID or a landsat image.
        data_df.index = data_df[feature_id]
        if drops:
            drops.append('geo')
            data_df.drop(columns=drops, inplace=True, errors='ignore')
        # print(data_df.head())
        # data_df.to_csv(f)
        dfs.append(data_df)
    all_yrs = pd.concat(dfs, axis=1)
    return all_yrs


def clustered_sample_etf_direct_nc(feature_coll, debug=False, mask_type='irr',
                                   start_yr=2000, end_yr=2024, feature_id='FID', drops=None):
    """ Process GEE SEEBOP etf data and ... do something with netcdfs?

    Combined behavior of clustered_sample_etf and list_and_copy_gcs_bucket"""
    feature_coll = ee.FeatureCollection(feature_coll)

    s, e = '1987-01-01', '2021-12-31'
    irr_coll = ee.ImageCollection(IRR)
    coll = irr_coll.filterDate(s, e).select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(5)

    dfs = []

    for year in range(start_yr, end_yr + 1):

        irr = irr_coll.filterDate('{}-01-01'.format(year),
                                  '{}-12-31'.format(year)).select('classification').mosaic()
        irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))

        desc = 'etf_{}_{}'.format(mask_type, year)

        # # Check that file has not already been created.
        # f = os.path.join(dest_dir, '{}.csv'.format(desc))
        # if os.path.exists(f):
        #     print(desc, 'exists, skipping')
        #     continue

        coll = ee.ImageCollection(ETF).filterDate('{}-01-01'.format(year), '{}-12-31'.format(year))
        coll = coll.filterBounds(feature_coll)
        scenes = coll.aggregate_histogram('system:index').getInfo()

        first, bands = True, None
        selectors = [feature_id]

        for img_id in scenes:

            # if img_id != 'lt05_036029_20000623':
            #     continue

            splt = img_id.split('_')
            _name = '_'.join(splt[-3:])

            selectors.append(_name)

            full_path = '{}/{}'.format(ETF, img_id)
            etf_img = ee.Image(full_path).rename(_name)  # fixed slash
            etf_img = etf_img.divide(10000)

            if mask_type == 'no_mask':
                etf_img = etf_img.clip(feature_coll.geometry())
            elif mask_type == 'irr':
                etf_img = etf_img.clip(feature_coll.geometry()).mask(irr_mask)
            elif mask_type == 'inv_irr':
                etf_img = etf_img.clip(feature_coll.geometry()).mask(irr.gt(0))

            if first:
                bands = etf_img
                first = False
            else:
                bands = bands.addBands([etf_img])

            if debug:
                point = ee.Geometry.Point([-107.188225, 44.9011])
                data = etf_img.sample(point, 30).getInfo()
                print(data['features'])

        data = bands.reduceRegions(collection=feature_coll,
                                   reducer=ee.Reducer.mean(),
                                   scale=30)

        # extract pixel count to filter data
        # We don't actually need that! We get irrigated fraction somewhere else.
        count = bands.reduceRegions(collection=feature_coll,
                                    reducer=ee.Reducer.count(),
                                    scale=30)

        data_df = ee.data.computeFeatures({
            'expression': data,
            'fileFormat': 'PANDAS_DATAFRAME'
        })

        count_df = ee.data.computeFeatures({
            'expression': count,
            'fileFormat': 'PANDAS_DATAFRAME'
        })

        print(desc)

        data_df = data_df.melt(
            id_vars=["FID"],
            value_vars=scenes,
            var_name="image",
            value_name="etf_{}".format(mask_type),
        )

        count_df = count_df.melt(
            id_vars=["FID"],
            value_vars=scenes,
            var_name="image",
            value_name="etf_{}_ct".format(mask_type),
        )

        print(data_df)
        print(count_df)

        # remove 'irr_' from beginning of band names.
        data_df['date'] = [dt.strptime(i[-8:], '%Y%m%d') for i in data_df['image']]
        data_df['image'] = [i[:-9] for i in data_df['image']]

        # Create multiindex for xarray formatting
        mi = pd.MultiIndex.from_frame(data_df[['FID', 'date', 'image']])  # Do I want this much in indices?
        # print("{:.0f} duplicated out of {:.0f}. ({:.2f}%)".format(mi.duplicated().sum(), len(mi),
        #                                                           100*(mi.duplicated().sum()/len(mi))))
        # print(data_df['image'].unique())
        data_df.index = mi
        data_df = data_df.drop(columns=['FID', 'date', 'image'])
        data_df = data_df.sort_index()

        print()
        print(data_df)

        # print(data_df.head())
        # data_df.to_csv(f)
        data_df = data_df.to_xarray()
        dfs.append(data_df)
        print()
        print(data_df)
    dfs = xarray.merge(dfs)
    return dfs


def clustered_sample_etf_direct_nc1(bounds, debug=False, mask_type='irr',
                                    start_yr=2000, end_yr=2024, feature_id='FID', drops=None):
    """ Process GEE SEEBOP etf data and ... do something with netcdfs?
    This is an experiment looking at the efficiency of dowloading a region's worth of data and then using xvec
    later to do the zonal statistics. Not working at the moment, returns single values, all zeros, always.
    This is closer now, but I don't think it is saving any time, and it is much more complicated.

    Combined behavior of clustered_sample_etf and list_and_copy_gcs_bucket
    bounds: the length-4 list of boundaries of the field area in EPSG:4326 decimal degrees.
    """
    # feature_coll = ee.FeatureCollection(feature_coll)
    # [110.0, 0.0, 113.0, 0.0, 110.0, 3.0]
    # print(bounds)
    # print([bounds[0], bounds[1], bounds[0], bounds[3], bounds[2], bounds[3], bounds[2], bounds[1]])
    # print([[[bounds[0], bounds[1]], [bounds[0], bounds[3]], [bounds[2], bounds[3]],
    #         [bounds[2], bounds[1]], [bounds[0], bounds[1]]]])
    # bounds = ee.Geometry.Polygon([[[bounds[0], bounds[1]], [bounds[0], bounds[3]],
    #                               [bounds[2], bounds[3]], [bounds[2], bounds[1]], [bounds[0], bounds[1]]]])
    bounds = ee.Geometry.Polygon([bounds[0], bounds[1], bounds[0], bounds[3],
                                  bounds[2], bounds[3], bounds[2], bounds[1]])

    s, e = '1987-01-01', '2021-12-31'
    irr_coll = ee.ImageCollection(IRR)
    coll = irr_coll.filterDate(s, e).select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(5)

    dfs = []

    for year in range(start_yr, end_yr + 1):

        irr = irr_coll.filterDate('{}-01-01'.format(year),
                                  '{}-12-31'.format(year)).select('classification').mosaic()
        irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))

        desc = 'etf_{}_{}'.format(mask_type, year)

        # # Check that file has not already been created.
        # f = os.path.join(dest_dir, '{}.nc'.format(desc))
        # if os.path.exists(f):
        #     print(desc, 'exists, skipping')
        #     continue

        coll = ee.ImageCollection(ETF).filterDate('{}-01-01'.format(year), '{}-12-31'.format(year))
        coll = coll.filterBounds(bounds)

        scenes = coll.aggregate_histogram('system:index').getInfo()

        first, bands = True, None
        # selectors = [feature_id]
        selectors = []

        for img_id in scenes:

            # if img_id != 'lt05_036029_20000623':
            #     continue

            splt = img_id.split('_')
            _name = '_'.join(splt[-3:])

            selectors.append(_name)

            full_path = '{}/{}'.format(ETF, img_id)  # fixed slash
            etf_img = ee.Image(full_path).rename(_name)  # is this the right place?
            etf_img = etf_img.divide(10000)

            # if mask_type == 'no_mask':
            #     etf_img = etf_img.clip(feature_coll.geometry())
            # elif mask_type == 'irr':
            #     etf_img = etf_img.clip(feature_coll.geometry()).mask(irr_mask)
            # elif mask_type == 'inv_irr':
            #     etf_img = etf_img.clip(feature_coll.geometry()).mask(irr.gt(0))

            if mask_type == 'irr':
                etf_img = etf_img.mask(irr_mask).reproject(crs='EPSG:4326', scale=30).clip(bounds)
            elif mask_type == 'inv_irr':
                etf_img = etf_img.mask(irr.gt(0)).reproject(crs='EPSG:4326', scale=30).clip(bounds)

            if first:
                bands = etf_img
                first = False
            else:
                bands = bands.addBands([etf_img])

            if debug:
                point = ee.Geometry.Point([-107.188225, 44.9011])
                data = etf_img.sample(point, 30).getInfo()
                print(data['features'])

        # # TODO extract pixel count to filter data
        # data = bands.reduceRegions(collection=feature_coll,
        #                            reducer=ee.Reducer.mean(),
        #                            scale=30)

        data_df = ee.data.computePixels({
            'expression': bands,
            'fileFormat': 'NUMPY_NDARRAY'
        })

        print(desc)
        # # Drop all columns that are not FID or a landsat image.
        # data_df.index = data_df[feature_id]
        # if drops:
        #     drops.append('geo')
        #     data_df.drop(columns=drops, inplace=True, errors='ignore')
        # # print(data_df.head())
        # data_df.to_xarray()
        dfs.append(data_df)
    return dfs


if __name__ == '__main__':

    d = '/media/research/IrrigationGIS/swim'
    if not os.path.exists(d):
        d = '/home/dgketchum/data/IrrigationGIS/swim'

    is_authorized()
    bucket_ = 'wudr'
    fields = 'users/dgketchum/fields/tongue_annex_20OCT2023'
    for mask in ['inv_irr', 'irr']:
        chk = os.path.join(d, 'examples/tongue/landsat/extracts/etf/{}'.format(mask))
        clustered_sample_etf(fields, bucket_, debug=False, mask_type=mask, check_dir=None)

# ========================= EOF ====================================================================
