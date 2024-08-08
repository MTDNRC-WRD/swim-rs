""" A file to control variables used throughout the project.

Limits/eliminates need to re-initialize the same variables in each file. Primarily features file locations
that may change between users.
"""

import os

# Google Cloud Storage bucket
gcs_bucket = 'mt_cu_2024'

# Name of project, used throughout project in paths and filenames
project_name = 'haugen'

# Instance of field shapefile in Google Earth Engine
ee_fields = 'projects/ee-hehaugen/assets/029_Flathead_Fields_Subset'
# name of column in shapefile to use as index
index_col = 'FID'

# Local directory where all project-specific data is stored.
d = 'C:/Users/CND571/Documents/Data/swim/examples/{}'.format(project_name)
# d = 'C:/Users/CND571/Documents/Data/swim'

# Local instance of field shapefile, identical to ee_fields
fields_shp = os.path.join(d, 'examples', project_name, 'gis', '029_Flathead_Fields_Subset.shp')

# gridmet correction surface file local directory
gridmet_dir = 'C:/Users/CND571/Documents/Data/gridmet'

