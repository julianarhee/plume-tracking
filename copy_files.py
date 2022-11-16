#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File           : copy_files.py
Created        : 2022/11/16 9:49:06
Project        : /Users/julianarhee/Repositories/plume-tracking
Author         : jyr
Email          : juliana.rhee@gmail.com
Last Modified  : 
'''
import os
import glob
import re
import sys
import shutil
import argparse
# import some custom funcs
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import utils as util

# 

def copy_log_files(src_dir, dst_dir, fmt='log'):
    '''
    Find all .log files (or whatever fmt is) from src_dir to dst_dir.

    Arguments:
        src_dir -- _description_
        dst_dir -- _description_

    Keyword Arguments:
        fmt -- _description_ (default: {'log'})
    '''
    fns = glob.glob(os.path.join(src_dir, '*.{}'.format(fmt)))
    for f in fns:
        new_f = os.path.join(dst_dir, os.path.split(f)[-1])
        shutil.copy(f, new_f)


def main():
    parser = argparse.ArgumentParser(description='Preprocessing steps.')
    parser.add_argument('-R', '--rootdir', type=str, 
        default='/Users/julianarhee/Library/CloudStorage/GoogleDrive-edge.tracking.ru@gmail.com/My Drive/Edge_Tracking/Data',
        help='Base name for directories. Example: /Full/path/to/data/folder/Edge_Tracking/Data')
    parser.add_argument('-E', '--experiment', type=str, default='',
        help='experiment name Example: pam-activation')
    args = parser.parse_args()

    rootdir = args.rootdir
    experiment = args.experiment

    src_dir = os.path.join(rootdir, experiment)
    dst_dir = os.path.join(rootdir, 'jyr', experiment, 'raw')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    print("Copying raw files to: {}".format(dst_dir))

    copy_log_files(src_dir, dst_dir, fmt='log')

if __name__ == '__main__':
    main()