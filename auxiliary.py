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
import re

import pandas as pd
import numpy as np

import behavior as butil
import utils as util

# Loading
import yaml
def load_cam_config(camdir):
    '''Load yaml config file from PIMAQ acquisition. 

        Uses dir from get_videodir_from_tstamp()'''
    cfg_cam_fp = glob.glob(os.path.join(camdir, '*.yaml'))[0]
    with open(cfg_cam_fp, "r") as stream:
        try:
            cfg_cam = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return cfg_cam


# 2p
def load_neural_traces_from_csv(tifname, rootdir='/mnt/minerva/2p-data'):
    sess = re.findall('\d{8}', tifname)[0]
    traces_fns = glob.glob(os.path.join(rootdir, str(sess), 'processed', '{}*.csv'.format(tifname)))  # manually extracted
    d_list=[]
    for fn in traces_fns:
        df_ = pd.read_csv(fn)
        if 'LH' in fn:
            df_['hemi'] = 'left'
        else:
            df_['hemi'] = 'right'
        d_list.append(df_)
    traces = pd.concat(d_list, axis=0)

    return traces


# File sorting
def get_videodir_from_tstamp(fn, all_vid_fpaths, verbose=False):
    '''
    For a given log file, with YYYYMMDD-HHmmss format, find corresponding video directory for camera.
    '''
    behav_tstamp = int(re.match('\d{8}-\d{6}', fn)[0].split('-')[-1])

    cam0_tstamp_matched_ix = np.array([abs(behav_tstamp - int(re.match('\d{8}-\d{6}', os.path.split(vp)[-1])[0].split('-')[-1])) for vp in all_vid_fpaths]).argmin()
    camdir = all_vid_fpaths[cam0_tstamp_matched_ix]
    if verbose:
        print("Found cam0 video paths:")
        for fi, vp in enumerate(all_vid_fpaths):
            print(fi, os.path.split(vp)[-1])
        print("Found corresponding cam vid: {} (log={})".format(os.path.split(camdir)[-1], fn))

    return camdir


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

    return df_, exp_config

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
        currd = process_df(currd) #, fps=fps, filter_duration=False)
        d_list.append(currd)
    df = pd.concat(d_list, axis=0)
    return df


def process_df(df_, xvar='ft_posx', yvar='ft_posy', smooth=False):
    # add some calculations
    df_ = butil.calculate_speed(df_, xvar=xvar, yvar=yvar)
    df_ = butil.calculate_distance(df_, xvar=xvar, yvar=yvar)
    # smooth?
    if smooth:
        df_ = butil.smooth_traces(df_, fs=fs, fc=fc, return_same=True)

    return df_

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
#    if 'rel_time' in bad_skips.keys():
#        print("... WARNING: rel_time has skips, only taking up to 1st time point")
#        i0 = df_.iloc[0].name
#        df = df_.loc[i0:bad_skips['rel_time'][0]]
#        df['blocknum'] = 0
#        return df

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


def logfiles_to_dataframe(logfiles, flyid, xvar='ft_posx', yvar='ft_posy', default_cond='tap'):
    '''
    combine data from logfiles into processed df.

    Args
    ----
    logfiles: list of full paths to saved logs
    flyid: fly3, for ex.
    
    '''
    d_list = []
    for fn in logfiles: 
        print(fn)
        try:
            #fpath = os.path.join(logdir, '{}.log'.format(fn))
            curr_cond = extract_fly_condition_from_filename(flyid, fn)
            if curr_cond in ('', None):
                curr_cond = default_cond
            df_, cfg = load_dataframe(fn)
            df_['condition'] = curr_cond
            df_['flyid'] = flyid
            #fly_id = os.path.splitext(os.path.split(fpath)[-1])[0]
            d_list.append(df_)
        except Exception as e:
            print("ERROR: {}".format(fn))
            traceback.print_exc()
    df0 = pd.concat(d_list, axis=0)

    return df0

def merge_blocks(df0, fps=120.):
    '''
    Takes bad_skips in position when FT restarts to 0, and appends to prev trajectory chunk.

    Args:
    -----
    Processed df from logfiles_to_dataframe()
    
    Returns:
    --------
    df0: pd.DataFrame, with 'blocknum' as column

    '''
    df_list = []
    for fn, df in df0.groupby('filename'):
        #print(fi)
        if df['blocknum'].nunique()>1:
            # Get last points of 1st file
            last_x, last_y, last_t = df[df['blocknum']==0][['ft_posx', 'ft_posy', 'rel_time']].iloc[-1]
            for bnum, block_ in df[df['blocknum']>0].groupby('blocknum'):
                #print(bnum, last_x, last_y, last_t)
                curr_xvs = df[df['blocknum']==bnum]['ft_posx'].values
                curr_yvs = df[df['blocknum']==bnum]['ft_posy'].values
                #curr_ts = df[df['blocknum']==bnum]['rel_time'].values
                # add offsets
                df.loc[df['blocknum']==bnum, 'ft_posx'] = curr_xvs + last_x
                df.loc[df['blocknum']==bnum, 'ft_posy'] = curr_yvs + last_y
                #df.loc[df['blocknum']==bnum, 'rel_time'] = curr_ts + last_t
                # update last
                last_x, last_y, last_t = df[df['blocknum']==bnum][['ft_posx', 'ft_posy', 'rel_time']].iloc[-1]
            # reprocess with updated position info
            df_p = process_df(df) # fps=fps, filter_duration=False)
            df_list.append(df_p)
        else:
            df_list.append(df)
    merged = pd.concat(df_list, axis=0)
    #merged.loc[merged['speed']>100] = None

    return merged

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





