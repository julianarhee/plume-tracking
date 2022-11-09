#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File           : imaging.py
Created        : 2022/11/8 9:41:21
Project        : /Users/julianarhee/Repositories/plume-tracking
Author         : jyr
Email          : juliana.rhee@gmail.com
Last Modified  : 
'''
import os
import glob
import sys
import traceback
import cv2

import xmltodict
import tifffile as tf
import numpy as np
import pandas as pd
import pylab as pl

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import utils as util

class ImageStack:
    def __init__(self, tif_dir, ft_fpath='', scope='hedwig', 
                n_slices=0, n_volumes=0):
        self.name = os.path.split(tif_dir)[1]
        self.tif_dir = tif_dir
        self.tif_fpath = '{}.tif'.format(tif_dir)
        self.ft_fpath = ft_fpath
        self.scope = scope
        #self.image_size = self.get_image_size()
        self.get_metadata()
        self.stack = None

    def get_metadata(self):
        '''
        Get metadata (OME). Use tifffile library.

        https://github.com/cgohlke/tifffile/blob/master/tifffile/tifffile.py
        '''
        try:
            im_fns = sorted(glob.glob(os.path.join(self.tif_dir, '*.tif')), \
                        key=util.natsort)
            tif = tf.TiffFile(im_fns[0])
            page = tif.pages[0]
            assert tif.pages.pages[0].is_ome, "Not OME. Unknown type."
            ome_dict = xmltodict.parse(tif.ome_metadata)
            acquisition_date = ome_dict['OME']['Image']['AcquisitionDate']
            tstamps = [float(v['@DeltaT']) for v in ome_dict['OME']['Image']['Pixels']['Plane']]
            self.acquisition_date = acquisition_date
            self.timestamps = tstamps
            # image size
            sz_x = float(ome_dict['OME']['Image']['Pixels']['@PhysicalSizeX'])
            sz_y = float(ome_dict['OME']['Image']['Pixels']['@PhysicalSizeY'])
            sz_z = float(ome_dict['OME']['Image']['Pixels']['@PhysicalSizeZ'])
            self.pixel_size_um = (sz_x, sz_y)
            self.z_stepsize = sz_z
            self.n_slices = int(ome_dict['OME']['Image']['Pixels']['@SizeZ'])
            self.height = int(ome_dict['OME']['Image']['Pixels']['@SizeY']) 
            self.width = int(ome_dict['OME']['Image']['Pixels']['@SizeX']) 
            self.dtype = ome_dict['OME']['Image']['Pixels']['@Type']

            # calculate n reps
            nframes_total = len(ome_dict['OME']['Image']['Pixels']['TiffData'])
            d = pd.DataFrame([int(v['@FirstZ']) for v in ome_dict['OME']['Image']['Pixels']['TiffData']],
                columns=['slice'], index=list(range(nframes_total)))
            self.n_volumes = d.groupby('slice')['slice'].count().min()
            self.nframes_total = nframes_total

        except Exception as e:
            traceback.print_exc()
        finally:
            tif.close()

    def imread(self):
        im = tf.imread(self.tif_fpath)

        # %timeit im = cv2.imread(self.tif_fpath, -1)
        # 7.61 ms ± 799 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
        # %timeit im = cv2.imreadmulti(stk.tif_fpath, [], cv2.IMREAD_UNCHANGED)
        # 55.3 ms ± 5.43 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
        # %timeit im = tf.imread(self.tif_fpath)
        # 20.7 ms ± 83.1 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

        # reshape
        imr = np.reshape(im, (self.n_volumes, self.n_slices, self.height, self.width)).astype(self.dtype)

        return imr

    def load_stack(self):

        self.stack = self.imread()


    def average_volumes(self, dst_dir=None, save=False):
        if self.stack is None:
            im = self.imread()
        avg = im.astype(float).mean(axis=0).astype(self.dtype)

        if dst_dir is not None and save is True:
            outfn = os.path.join(dst_dir, 'avgstack_{}.tif'.format(self.name))
            tf.imwrite(outfn, avg)

        return avg

    def plot_average_slices(self, dst_dir=None, cmap='viridis',
                        col_wrap=5, fmt='png', save=False):

        if dst_dir is not None:
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
                print("Saving images to: {}".format(dst_dir))

        avg = self.average_volumes()

        # plot
        n_slices = int(avg.shape[0])
        nrows = int(np.ceil(n_slices/col_wrap))

        fig, axn = pl.subplots(nrows, col_wrap, sharex=True, sharey=True,
                         figsize=(col_wrap, nrows*2))
        for ax, ai in zip(axn.flat, range(self.n_slices)):
            ax.imshow(avg[ai, :, :], cmap=cmap)
        util.label_figure(fig, self.tif_fpath)
        pl.tight_layout()

        if dst_dir is not None and save is True:
            figname = 'slices_{}.{}'.format(self.name, fmt)
            pl.savefig(os.path.join(dst_dir, figname)) 

        pl.show()

        return fig