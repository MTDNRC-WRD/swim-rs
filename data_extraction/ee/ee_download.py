""" Downloads files created by ee_props.py from GCS to local computer. Also creates necessary directory structure. """

import os
from google.cloud import storage

from prep import info


if __name__ == '__main__':
    # general setup
    client = storage.Client()
    bucket = client.get_bucket(info.gcs_bucket)
    project = info.project_name
    d = info.d

    # one-off files for properties: check for directory, then download 4 files from gcs
    dprop = os.path.join(d, 'properties')
    if not os.path.exists(dprop):
        os.makedirs(dprop)
    for prop in ['cdl', 'irr', 'ssurgo', 'landfire']:
        file = '{}_{}.csv'.format(project, prop)
        blob = bucket.blob(file)
        blob.download_to_filename('{}/{}'.format(dprop, file))

    # Looping through etf and ndvi files.
    for a in ['etf', 'ndvi']:
        for b in ['inv_irr', 'irr']:
            # Create nested directories if they do not already exist locally.
            # d = ('C:/Users/CND571/Documents/Data/swim/examples/{}/met_timeseries/landsat/extracts/{}/{}'
            #      .format(project, a, b))
            dpath = '{}/met_timeseries/landsat/extracts/{}/{}'.format(d, a, b)
            if not os.path.exists(dpath):
                os.makedirs(dpath)

            # Looping through years
            for y in range(1987, 2024):  # Cannot use star operator, need to loop through each file individually.
                file = '{}_{}_{}.csv'.format(a, y, b)

                blob = bucket.blob('{}/{}'.format(project, file))
                blob.download_to_filename('{}/{}'.format(dpath, file))

                # # ignoring count file for now.
                # for c in ['', '_ct']:
                #     # Do I need to download the count files?
                #     # I don't think they're being used for anything at the moment...
                #     # They might even mess things up if everything has to be in its own separate folders...
                #     file = '{}_{}_{}{}.csv'.format(a, y, b, c)
                #     blob = bucket.blob('{}/{}'.format(project, file))
                #     blob.download_to_filename('{}/{}'.format(d, file))

# ========================= EOF ====================================================================
