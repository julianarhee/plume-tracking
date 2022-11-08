#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File           : tifs_to_stacks.py
Created        : 2022/11/7 19:18:59
Project        : /Users/julianarhee/Repositories/plume-tracking
Author         : Juliana Rhee
Last Modified  : 
'''
import os
import sys
import glob
import argparse
import re
import cv2
from pathlib import Path
import tifffile as tf
import numpy as np
# import some custom funcs
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import utils as util

rootdir = '/Volumes/Julie/plume-tracking'
experiment = 'stripgrid/20221102'

def main():
    parser = argparse.ArgumentParser(description='Preprocessing steps.')
    parser.add_argument('-R', '--rootdir', type=str, 
        default='/Volumes/Julie/plume-tracking',
        help='Base name for directories. Example: /Full/path/to/server/folder')
    parser.add_argument('-S', '--session', type=str, default='',
        help='experiment name Example: 20221031')
    parser.add_argument('-v', '--verbose', type=bool, default=False,
        help='verbose, print all statements')

    args = parser.parse_args()
    rootdir = args.rootdir
    session = args.session

    src_dir = os.path.join(rootdir, session, 'raw')
    tseries = glob.glob(os.path.join(src_dir, 'TSeries*'))

    tif_dirs = sorted([p for p in tseries if Path(p).is_dir()], \
                key=util.natsort)
    already_processed = []
    for tdir in tif_dirs:   
        fname = '{}.tif'.format(os.path.split(tdir)[-1])
        tif_fname = os.path.join(src_dir, fname)
        if os.path.exists(tif_fname):
            print("Already processed: {}, skipping".format(fname))
            already_processed.append(tdir)
    tifs_to_process = [d for d in tif_dirs if d not in already_processed]

    []
    tifs_to_run = []
    for tdir in tifs_to_process:
        print(os.path.split(tdir)[-1])
        fns = sorted(glob.glob(os.path.join(tdir, '*.tif')), key=util.natsort)
        print("... processing {} images to stack.".format(len(fns)))
        fullstack = np.dstack([cv2.imread(i, -1) for i in fns])
        fullstack1 = np.swapaxes(np.swapaxes(fullstack, 0, 2), 1, 2)

        fname = '{}.tif'.format(os.path.split(tdir)[-1])
        outfn = os.path.join(src_dir, fname)
        tf.imwrite(outfn, fullstack1)
        print("... save to .tif stack: {}".format(outfn)) 
   

if __name__ == '__main__':
    main()