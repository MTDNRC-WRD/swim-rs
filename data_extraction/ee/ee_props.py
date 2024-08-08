import os
import ee

from prep import info
from data_extraction.ee.ee_utils import is_authorized
from etf_export import clustered_field_etf
from ndvi_export import clustered_field_ndvi

IRR = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'

# See https://websoilsurvey.nrcs.usda.gov/app/WebSoilSurvey.aspx
# to check soil parameters

# OpenET AWC is in cm/cm
AWC = 'projects/openet/soil/ssurgo_AWC_WTA_0to152cm_composite'
# OpenET KSAT is in micrometers/sec
KSAT = 'projects/openet/soil/ssurgo_Ksat_WTA_0to152cm_composite'
CLAY = 'projects/openet/soil/ssurgo_Clay_WTA_0to152cm_composite'
SAND = 'projects/openet/soil/ssurgo_Sand_WTA_0to152cm_composite'


def get_cdl(fields, desc, bucket, selector='FID'):
    plots = ee.FeatureCollection(fields)
    crops, first = None, True
    cdl_years = [x for x in range(2008, 2023)]

    _selectors = [selector]

    for y in cdl_years:

        image = ee.Image('USDA/NASS/CDL/{}'.format(y))
        crop = image.select('cropland')
        _name = 'crop_{}'.format(y)
        _selectors.append(_name)
        if first:
            crops = crop.rename(_name)
            first = False
        else:
            crops = crops.addBands(crop.rename(_name))

    modes = crops.reduceRegions(collection=plots,
                                reducer=ee.Reducer.mode(),
                                scale=30)

    out_ = '{}'.format(desc)
    task = ee.batch.Export.table.toCloudStorage(
        modes,
        description=out_,
        bucket=bucket,
        fileNamePrefix=out_,
        fileFormat='CSV',
        selectors=_selectors)

    task.start()


def get_irrigation(fields, desc, bucket, debug=False, selector='FID'):
    plots = ee.FeatureCollection(fields)
    irr_coll = ee.ImageCollection(IRR)

    _selectors = [selector, 'LAT', 'LON']  # do I need lat and lon?
    first = True

    area, irr_img = ee.Image.pixelArea(), None

    for year in range(1987, 2022):

        irr = irr_coll.filterDate('{}-01-01'.format(year),
                                  '{}-12-31'.format(year)).select('classification').mosaic()

        irr = irr.lt(1)

        _name = 'irr_{}'.format(year)
        _selectors.append(_name)

        if first:
            irr_img = irr.rename(_name)
            first = False
        else:
            irr_img = irr_img.addBands(irr.rename(_name))

    means = irr_img.reduceRegions(collection=plots,
                                  reducer=ee.Reducer.mean(),
                                  scale=30)

    if debug:
        debug = means.filterMetadata('FID', 'equals', 1789).getInfo()

    task = ee.batch.Export.table.toCloudStorage(
        means,
        description=desc,
        bucket=bucket,
        fileNamePrefix=desc,
        fileFormat='CSV',
        selectors=_selectors)

    task.start()


def get_ssurgo(fields, desc, bucket, debug=False, selector='FID'):
    plots = ee.FeatureCollection(fields)

    ksat = ee.Image(KSAT).select('b1').rename('ksat')
    awc = ee.Image(AWC).select('b1').rename('awc')
    clay = ee.Image(CLAY).select('b1').rename('clay')
    sand = ee.Image(SAND).select('b1').rename('sand')

    img = ksat.addBands([awc, clay, sand])

    _selectors = [selector, 'LAT', 'LON'] + ['awc', 'ksat', 'clay', 'sand']

    means = img.reduceRegions(collection=plots,
                              reducer=ee.Reducer.mean(),
                              scale=30)

    if debug:
        debug = means.filterMetadata('FID', 'equals', 1789).getInfo()

    task = ee.batch.Export.table.toCloudStorage(
        means,
        description=desc,
        bucket=bucket,
        fileNamePrefix=desc,
        fileFormat='CSV',
        selectors=_selectors)

    task.start()
    print(desc)


def get_landfire(fields, desc, bucket, debug=False, selector='FID'):
    plots = ee.FeatureCollection(fields)

    height = ee.ImageCollection('LANDFIRE/Vegetation/EVH/v1_4_0').select('EVH').first().rename('plant_height')

    img = height

    _selectors = [selector, 'LAT', 'LON'] + ['height']

    means = img.reduceRegions(collection=plots,
                              reducer=ee.Reducer.mean(),
                              scale=30)

    if debug:
        debug = means.filterMetadata('FID', 'equals', 1789).getInfo()

    task = ee.batch.Export.table.toCloudStorage(
        means,
        description=desc,
        bucket=bucket,
        fileNamePrefix=desc,
        fileFormat='CSV',
        selectors=_selectors)

    task.start()
    print(desc)


if __name__ == '__main__':
    ee.Initialize()

    d = 'C:/Users/CND571/Documents/Data/swim'

    # bucket_ = 'mt_cu_2024'
    # project_ = 'haugen'
    # index_col = 'FID'
    # fields_ = 'projects/ee-hehaugen/assets/029_Flathead_Fields_Subset'
    bucket_ = info.gcs_bucket
    project_ = info.project_name
    index_col = info.index_col
    fields_ = info.ee_fields

    # fields_ = 'C:/Users/CND571/Documents/Data/swim/examples/haugen/gis/029_Flathead_Fields_Subset.shp'

    description = '{}_cdl'.format(project_)
    get_cdl(fields_, description, bucket_, selector=index_col)

    description = '{}_irr'.format(project_)
    get_irrigation(fields_, description, bucket_, debug=False, selector=index_col)

    description = '{}_ssurgo'.format(project_)
    get_ssurgo(fields_, description, bucket_, debug=False, selector=index_col)  # soil properties!

    description = '{}_landfire'.format(project_)
    get_landfire(fields_, description, bucket_, debug=False, selector=index_col)  # What is it?

    # Export 8 files per year to gcs for etf and ndvi data.
    # For each year, ndvi and etf, irr and inv_rr, and data and pixel count. (2x2x2=8)
    is_authorized()
    for mask in ['inv_irr', 'irr']:
        chk = os.path.join(d, 'examples/{}/met_timeseries/landsat/extracts/etf/{}'.format(project_, mask))
        clustered_field_etf(fields_, project_, bucket_, debug=False, mask_type=mask, check_dir=chk)
        chk = os.path.join(d, 'examples/{}/met_timeseries/landsat/extracts/ndvi/{}'.format(project_, mask))
        clustered_field_ndvi(fields_, project_, bucket_, debug=False, mask_type=mask, check_dir=chk)

# ========================= EOF ====================================================================
