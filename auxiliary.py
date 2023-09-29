#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File           : auxiliary.py
Created        : 2023/05/17 15:05:16
Project        : /Users/julianarhee/Repositories/plume-tracking
Author         : Juliana Rhee
Last Modified  : 
Copyright (c) 2022 Your Company
'''

import os
import re
import glob
import h5py

import pandas as pd
import numpy as np

import behavior as butil
import utils as util

def extract_fly_condition_from_filename(flyid, fpath):
    '''
    Given a filepath, extracts everything after flyX and before _000.log
    Removes special characters that are not alphanumeric.
    '''
    filename_suffix = fpath.split(flyid)[-1]
    acquisition_str = re.findall(r'(_\d{3})', filename_suffix)[0]
    fly_suffix = filename_suffix.split(acquisition_str)[0]
    condition = ''.join(e for e in fly_suffix if e.isalnum())

    return condition

def get_logs_for_fly_date(date, flynum, logdir):
    '''
    Returns full paths to .log files, given date and fly number.

    Args:
        date (str): YYYYMMDD
        flynum (int): fly number, e.g., 2 for fly2
    Returns:
        logfiles (list)
    '''
    flyid = 'fly{}'.format(flynum) 
    r = re.compile(r'{}[^.]*{}'.format(date, flyid), flags=re.I | re.X)

    logfiles = glob.glob(os.path.join(logdir, '*.log'))
    curr_fns = sorted([f for f in logfiles if r.findall(f)], key=util.natsort)

    return curr_fns


def get_fly_and_date_from_fname(fn):
    datestr = re.findall('\d{8}-\d{6}', fn)[0]
    date = datestr.split('-')[0]
    fly_id = re.findall('fly\d{1}', fn)[0]

    return date, fly_id

def h5_to_df(meta_fpath):
    '''
    Get meta data as dataframe
    Metadata saved in PIMAQ/devices.py (self.write_obj.write_metadata()) 
    Write func defined in PIMAQ/utils.py (write_metadata(self, serial, framecount, frameid, timestamp,  sestime, cputime))

    cputime: computer time, only rly relevant if running on same machine
    framecount: counted frames in software
    frameid: frame ID from basler
    serial: serial number of camera
    sestime: relative time in session
    timestamp: timestamps of current rame 

    '''
    d_list=[]
    with h5py.File(meta_fpath, "r") as f:
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        #print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        for a_group_key in list(f.keys()):
            # If a_group_key is a group name, 
            # this gets the object names in the group and returns as a list
            data = list(f[a_group_key])
            #a_group_key = list(f.keys())[0]
            # get the object type for a_group_key: usually group or dataset
            #print(type(f[a_group_key])) 
            if type(f[a_group_key])==h5py._hl.dataset.Dataset:
                #print('dataset')
                # If a_group_key is a dataset name, 
                # this gets the dataset values and returns as a list
                data = list(f[a_group_key])
            else:
                # If a_group_key is a group name, 
                # this gets the object names in the group and returns as a list
                data = list(f[a_group_key])
            # preferred methods to get dataset values:
            #ds_obj = f[a_group_key]      # returns as a h5py dataset object
            ds_arr = f[a_group_key][()]  # returns as a numpy array
            sr = pd.Series(ds_arr, name=a_group_key)
            d_list.append(sr)
    df = pd.concat(d_list, axis=1)

    return df

# processing
def load_dataframe(fn):
    '''
    Loads and processes dataframe for tethered tap experiments (no odor).
    For behavior.load_dataframe():
        is_odor = False, remove_invalid=False
    Uses acquisition frame rate found in config file.

    '''
    exp_config = butil.load_experiment_config(fn)
    fps = exp_config['experiment']['acquisition_rate']
    df_ = butil.load_dataframe(fn, is_odor=False, remove_invalid=False) 
    df_ = process_df_blocks(df_, fps=fps)

    return df_

def process_df_blocks(df_, fps=120):
    '''
    Process df within block (see ft_skips_to_blocks), only relevant if remove_invalid=False

    def process_df(df, xvar='ft_posx', yvar='ft_posy', fliplr=False,
                bout_thresh=0.5, filter_duration=True, switch_method='previous',
                smooth=False, fps=60, fc=7.5, verbose=False):

    '''
    d_list = []
    df_  = ft_skips_to_blocks(df_, acquisition_rate=fps)
    for bnum, currd in df_.groupby('blocknum'):
        currd = butil.process_df(currd, fps=fps, filter_duration=False)
        d_list.append(currd)
    df = pd.concat(d_list, axis=0)
    return df

def ft_skips_to_blocks(df_, acquisition_rate=120, bad_skips=None, use_first_pos=False):
    '''
    Assign block numbers to separate chunks where FT "jumps" or skips frames.
    
    Arguments:
        df_ (pd.DataFrame) : single dataframe, load_dataframe(remove_invalude=False)

    Keyword Arguments:
        bad_skips (dict, None): output of check_ft_skips(df_, return_skips=True)

    '''
    if bad_skips is None:
        bad_skips = butil.check_ft_skips(df_, acquisition_rate=acquisition_rate, return_skips=True) 
    if 'rel_time' in bad_skips.keys():
        print("... WARNING: rel_time has skips, only taking up to 1st time point")
        i0 = df_.iloc[0].name
        df = df_.loc[i0:bad_skips['rel_time'][0]]
        df['blocknum'] = 0
        return df

    start_frame = df_.iloc[0].name
    end_frame = df_.iloc[-1].name
    if len(bad_skips)>0:
        zero_pos = []
        if 'ft_posx' in bad_skips.keys():
            zero_pos.extend(bad_skips['ft_posx'])
        if 'ft_posy' in bad_skips.keys():
            zero_pos.extend(bad_skips['ft_posy'])
        zero_pos = np.unique(zero_pos)

        # check with actual 0-pos
        if use_first_pos:
            x0 = round(df_['ft_posx'].iloc[0], 3)
            y0 = round(df_['ft_posy'].iloc[0], 3)
            df_[(df_['ft_posx'].round(3)==x0) & (df_['ft_posy'].round(3)==y0)]
        else:
            found_zeros = df_[(df_['ft_posx']==0) & (df_['ft_posy']==0)]

        bad_skip_start_ixs=[]
        if found_zeros.shape[0] != len(zero_pos):
            print("*Warning: N zero points ({}) don't match skips ({}) -- using N zero points.".format(found_zeros.shape[0], len(zero_pos)))
        zero_pos = found_zeros.index.tolist()
        grouped_by_consec = util.group_consecutives(zero_pos)
        bad_skip_start_ixs = [i[0] for i in grouped_by_consec]
        chunks=[]
        for ji, jump_index in enumerate(bad_skip_start_ixs): #bad_skips['ft_posx']):
            curr_chunk = (start_frame, jump_index)
            chunks.append(curr_chunk)
            start_frame = jump_index
            if ji==len(bad_skip_start_ixs)-1:
                chunks.append( (jump_index, end_frame+1) )
    else:
        chunks = [ (start_frame, end_frame+1) ]

    # include block num
    df_['blocknum'] = 0
    for bi, (start_ix, end_ix) in enumerate(chunks):
        df_.loc[start_ix:end_ix, 'blocknum'] = bi

    return df_       


# video processing
# ----------------------------------------------------
def find_synced_video_name(fn, vidsrc):
    # find 
    date, fly_id = get_fly_and_date_from_fname(fn)

    video_dirs = os.listdir(vidsrc)
    r = re.compile(r'{}[^.]*{}'.format(date, fly_id), flags=re.I | re.X)
    curr_vids = [f for f in video_dirs if r.findall(f)]

    curr_tstamp = fn.split('_')[0].split('-')[1][1] # assumes YYYYMMdd-HHmmSS
    found_tstamps = sorted([re.findall('\d{8}-\d{6}', f)[0]for f in curr_vids])
    closest_match = np.array([abs(int(curr_tstamp) - int(f.split('_')[0].split('-')[1])) for f in found_tstamps]).argmin()
    match_tstamp = found_tstamps[closest_match]
    curr_vid = [v for v in curr_vids if v.startswith(match_tstamp)][0]
    return curr_vid

