#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File           : behavior.py
Created        : 2022/09/21 14:24:37
Project        : /Users/julianarhee/Repositories/plume-tracking
Author         : Juliana Rhee
Last Modified  : 
Copyright (c) 2022 Your Company
'''

import os
import time
import glob
import re
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
import scipy as sp

import utils as util
import rdp
import _pickle as pkl
import scipy.stats as sts

# plotting
import matplotlib as mpl
import plotly.express as px
import pylab as pl
import seaborn as sns

# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def parse_info_from_file(fpath, experiment=None, 
            rootdir='/Users/julianarhee/Library/CloudStorage/GoogleDrive-edge.tracking.ru@gmail.com/My Drive/Edge_Tracking/Data'):
    '''
    _summary_

    Arguments:
        fpath -- _description_

    Keyword Arguments:
        experiment -- _description_ (default: {None})
        rootdir -- _description_ (default: {'/Users/julianarhee/Library/CloudStorage/GoogleDrive-edge.tracking.ru@gmail.com/My Drive/Edge_Tracking/Data'})

    Returns:
        experiment, datestr, fly_id, condition
    '''

    info_str = fpath.split('{}/'.format(rootdir))[-1]
    exp_cond_str, log_fname = os.path.split(info_str)
    fly_id=None
    cond=None

    if "Fly" in info_str or 'fly' in info_str:
        # assumes: nameofexperiment/maybestuff/FlyID
        experiment = exp_cond_str.split('/{}'.format('Fly'))[0] \
                    if "Fly" in exp_cond_str else exp_cond_str.split('/{}'.format('fly'))[0]
    else:
        experiment = exp_cond_str # fly ID likely in LOG filename
    # get fly_id
    fly_id = re.search('fly\d{1,3}[a-zA-Z]?', info_str, re.IGNORECASE)[0] # exp_cond_str
    # remove datestr
    date_str = re.search('[0-9]{8}-[0-9]{6}', log_fname)[0]
    cond_str = os.path.splitext(log_fname)[0].split('{}_'.format(date_str))[-1]
    # 
#    if re.search('fly\d{1,3}[a-zA-Z]?', cond_str, re.IGNORECASE):
#        #fly_id = re.search('fly\d{1,3}[a-zA-Z]?', log_fname, re.IGNORECASE)
#        #condition = cond_str.split('{}_'.format(fly_id))[-1]
#        condition = [c for c in cond_str.split('{}'.format(fly_id)) \
#                        if c!=fly_id and len(c)>1]
#        for ci, c in enumerate(condition):
#            if c.endswith('_'): 
#                condition[ci] = c[:-1]
#            elif c.startswith('_'):
#                condition[ci] = c[1:]
#        condition = condition[0]
#    else:
#        condition = cond_str
    #condition = '_'.join([c for c in cond_str.split('_') if fly_id not in c ])
    condition = '_'.join([c for c in cond_str.split('_') if fly_id not in c and not re.search('\d{3}', c)])

    #print(exp_cond, fly_id, condition)

    if fly_id is not None:
        fly_id = fly_id.lower()
    if condition is not None:
        condition = condition.lower()

    return experiment, date_str, fly_id, condition

def load_dataframe_test(fpath, mfc_id=None, led_id=None, verbose=False, cond='odor',
                    parse_info=True):
    '''
    Read raw .log file from behavior and return formatted dataframe.
    Assumes MFC for odor is either 'mfc2_stpt' or 'mfc3_stpt'.
    Assumes LED for on is 'led1_stpt'.

    Arguments:
        fpath -- (str) Full path to .log file
        mfc_id -- None, will find mfc var for odor automatically, otherwise "mfc2"
    '''
    # read .log as dataframe 

    df0 = pd.read_csv(fpath) #, encoding='latin' )#, sep=",", skiprows=[1], header=0, 
              #parse_dates=[1]).rename(columns=lambda x: x.strip())


    return df0


def load_dataframe(fpath, mfc_id=None, led_id=None, verbose=False, cond='odor',
                    parse_info=True, savedir=None):
    '''
    Read raw .log file from behavior and return formatted dataframe.
    Assumes MFC for odor is either 'mfc2_stpt' or 'mfc3_stpt'.
    Assumes LED for on is 'led1_stpt'.

    Arguments:
        fpath -- (str) Full path to .log file
        mfc_id -- None, will find mfc var for odor automatically, otherwise "mfc2"
    '''
    # read .log as dataframe 

    df0 = pd.read_csv(fpath, encoding='latin' )#, sep=",", skiprows=[1], header=0, 
              #parse_dates=[1]).rename(columns=lambda x: x.strip())

    # split up the timstampe str
    df0['timestamp'] = df0['timestamp -- motor_step_command']\
                            .apply(lambda x: x.split(' -- ')[0])
    df0['motor_step_command'] = df0['timestamp -- motor_step_command']\
                            .apply(lambda x: x.split(' -- ')[1]).astype('int')
    # convert timestamp str into datetime obj, convert to sec
    datefmt  = '%m/%d/%Y-%H:%M:%S.%f'
    df0['time'] = df0['timestamp'].apply(lambda x: \
                            time.mktime(datetime.strptime(x, datefmt).timetuple()) \
                            + datetime.strptime(x, datefmt).microsecond / 1E6 ).astype('float')
    # convert datestr
    df0['date'] = df0['timestamp'].apply(lambda s: \
            int(datetime.strptime(s.split('-')[0], "%m/%d/%Y").strftime("%Y%m%d")))

    if 'instrip' not in df0.columns:
        df0['instrip'] = False
        if mfc_id is not None:
            mfc_varname = '{}_stpt'.format(mfc_id)
            df0.loc[df0[mfc_varname]>0, 'instrip'] = True
        else: 
            mfc_vars = [c for c in df0.columns if 'mfc' in c \
                            and len(df0[c].unique())>1 and c in ['mfc2_stpt', 'mfc3_stpt'] \
                            and df0[c].dtype=='float64' ] #'float64'))]
            if len(mfc_vars)>0:
                assert len(mfc_vars)==1, "More than 1 MFC var found ({})".format(str(mfc_vars))
                mfc_varname = mfc_vars[0]
                df0.loc[df0[mfc_varname]>0, 'instrip'] = True
            # otherwise, only air (no odor)

    # check LEDs
    # if air_only==True, that means that we can ignore LEDs (not powered on)
    if 'led1_stpt' in df0.columns and 'led_on' not in df0.columns:
        df0['led_on'] = False

    if 'led1_stpt' in df0.columns:
#        if cond=='reinforced':
#            # for newer exp, weird thing where LED signal is 1 for "off" 
#            led1_vals = df0[~df0['instrip']]['led1_stpt'].unique() 
#            assert len(led1_vals)==1, "Too many out of strip values for LED: {}".format(str(led1_vals))
#            if led1_vals[0]==1: # when out of strip, no reinforcement. if has 1 and 0, likely, 1=off
#                df0['led_on'] = df0['led1_stpt']==0 # 1 is OFF, and 0 is ON (led2 is always 0)
#            elif led1_vals[0]==0:
#                df0['led_on'] = df0['led1_stpt']==1 # 1 is ON, and 0 is OFF (led2 is always 0)
        datestr = int(df0['date'].unique())
        if int(datestr) <= 20200720:
            df0['led_on'] = df0['led1_stpt']==1 
        else:
            df0['led_on'] = df0['led1_stpt']==0

        if cond in ['odor', 'air']:
            df0['led_on'] = False

#        else: #if cond=='light' or cond=='lightonly':   
#            # TODO: add check for datestr
#            df0['led_on'] = df0['led1_stpt']==0 # 20221018, quick fix for now bec dont know when things changed

    # check for wonky skips
    df0, ft_flag = check_ft_skips(df0, plot=True)
    if ft_flag:
        print("--> found wonky FTs, check: {}".format(fpath))
        if savedir is not None:
            fname = os.path.splitext(os.path.split(fpath)[-1])[0]
            pl.savefig(os.path.join(savedir, 'wonkyft_{}.png'.format(fname)))

    if parse_info:
        # get experiment info
        exp, datestr, fly_id, cond = parse_info_from_file(fpath)
        df0['experiment'] = exp
        df0['fly_name'] = fly_id
        df0['condition'] = cond
        df0['trial'] = datestr
        if verbose:
            print("Exp: {}, fly ID: {}, cond={}".format(exp, fly_id, cond))

        # make fly_id combo of date, fly_id since fly_id is reused across days
        df0['fly_id'] = ['{}-{}'.format(dat, fid) for (dat, fid) in df0[['date', 'fly_name']].values]
        df0['trial_id'] = ['{}_{}'.format(fly_id, trial) for (fly_id, trial) in \
                  df0[['fly_id', 'trial']].values]

    return df0




def check_ft_skips(df, plot=False):
    bad_skips={}
    max_step_size={'ft_posx': 10, 'ft_posy': 10, 'ft_frame': 100}
    for pvar, stepsize in max_step_size.items():
        if pvar=='ft_frame':
            wonky_skips = np.where(df[pvar]==1)[0]
            if len(wonky_skips)>1:
                wonky_skips = wonky_skips[1:]
            else:
                wonky_skips = []
        wonky_skips = np.where(df[pvar].diff().abs()>=stepsize)[0]
        if len(wonky_skips)>0:
            first_step = df[pvar].diff().abs().max()
            #time_step = df.iloc[wonky_skips[0]]['time'] - df.iloc[wonky_skips[0]-1]['time']
            bad_skips.update({pvar: wonky_skips})
            print("WARNING: found wonky ft skip ({} jumped {:.2f}).".format(pvar, first_step))
    if len(bad_skips.keys())>0:
        if plot==True:
            fig, ax = pl.subplots() 
            ax.plot(df['ft_frame'].diff().abs())
            cols = ['r', 'b', 'g']
            for pi, ((pvar, off_ixs), col) in enumerate(zip(bad_skips.items(), cols)):
                for i in off_ixs:
                    ax.plot(df.iloc[i].name, pi*100, '*', c=col, label=pvar)
            ax.legend()
            pl.show()

    if len(bad_skips)>0:
        flag=True
        wonky_skips = bad_skips['ft_frame']
        valid_df = df.iloc[0:wonky_skips[0]].copy()
        sz_removed = df.shape[0] - valid_df.shape[0]
        print("Removing {} of {} samples.".format(sz_removed, df.shape[0]))
    else:
        flag=False
        valid_df = df.copy()

    return valid_df, flag

def load_dataframe_resampled_csv(fpath):

    df_full = pd.read_table(fpath, sep=",", skiprows=[1], header=0, 
                parse_dates=[1]).rename(columns=lambda x: x.strip())
    df0 = df_full[['x', 'y', 'seconds', 'instrip']].copy()
    df0.loc[df0['instrip']=='False', 'instrip'] = 0
    df0 = df0.rename(columns={'x': 'ft_posx', 'y': 'ft_posy', 'seconds': 'time'}).astype('float')
    df0['instrip'] = df0['instrip'].astype(bool)

    other_cols = [c for c in df_full.columns if c not in df0.columns]
    for c in other_cols:
        df0[c] = df_full[c]

    return df0

def save_df(df, fpath):
    with open(fpath, 'wb') as f:
        pkl.dump(df, f)

def load_df(fpath):
    with open(fpath, 'rb') as f:
        df = pkl.load(f)
    return df

def load_combined_df(src_dir, create_new=False, verbose=False, save_errors=True):

    # first, check if combined df exists
    if 'raw' in src_dir:
        src_dir_temp = os.path.split(src_dir)[0]
    else:
        src_dir_temp = src_dir

    df_fpath = os.path.join(src_dir_temp, 'combined_df.pkl')
    if os.path.exists(df_fpath) and create_new is False:
        print("loading existing combined df")
        try:
            df = load_df(df_fpath)
            return df
        except Exception as e:
            create_new=True

    if save_errors:
        savedir = os.path.join(src_dir_temp, 'errors')
        if not os.path.exists(savedir):
            os.makedirs(savedir)
    else:
        savedir=None

    if create_new:
        print("Creating new combined df from raw files...")
        log_files = sorted([k for k in glob.glob(os.path.join(src_dir, '*.log'))\
                if 'lossed tracking' not in k], key=util.natsort)
        print("Found {} tracking files.".format(len(log_files)))

        dlist = []
        for fn in log_files:
            #air_only = '_Air' in fpath or '_air' in fpath
            #print(fn, air_only)
            exp, datestr, fly_id, cond = parse_info_from_file(fn)
            if verbose:
                print(exp, datestr, fly_id, cond)
            df_ = load_dataframe(fn, mfc_id=None, verbose=False, cond=cond, savedir=savedir)
            dlist.append(df_)
        df = pd.concat(dlist, axis=0)

        if 'vertical_strip/paired_experiments' in src_dir_temp:
            # update condition names
            df0.loc[df0['condition']=='light', 'condition'] = 'lightonly'

        # save
        print("Saving combined df to: {}".format(src_dir_temp))
        save_df(df, df_fpath)

    return df

def check_entry_left(df, entry_ix=0):
    '''
    Check whether fly enters from left/right of corridor based on prev tsteps.

    Arguments:
        df (pd.DataFrame) : dataframe with true indices
        entry_ix (int) : index of entry point

    Returns:
        entry_left (bool) : entered left True, otherwise False
    '''

    cumsum = df.loc[entry_ix-20:entry_ix]['ft_posx'].diff().cumsum().iloc[-1]
    if cumsum > 0: # entry is from left of strip (values get larger)
        entry_left=True
    elif cumsum < 0:
        entry_left=False
    else:
        entry_left=None

    return entry_left
    

def get_odor_params(df, odor_width=50, grid_sep=200, get_all_borders=True,
                    entry_ix=None, is_grid=False, check_odor=False, 
                    mfc_var='mfc2_stpt'):
    '''
    Get odor start times, boundary coords, etc.

    Arguments:
        df (pd.DataFrame) : full dataframe (from load_dataframe())
        
    Keyword Arguments:
        odor_width (float) :  Width of odor corridor in mm (default: 50)
        grid_sep (float) : Separation between odor strips (default: 200)
        entry_ix (int) : index of 1st entry (can be 1st entry of Nth corridor if is_grid)
        is_grid (bool) : 
            Is grid of odor strips (odor start is at odor edge if not 1st odor encounter)
        check_odor (bool) :
            Check if time of first instrip is also time of first mfc_on. Is redundant. 
    Returns:
        dict of odor params:
        {'trial_start_time': float
         'odor_start_time': float 
         'odor_boundary': (float, float) # x boundaries of odor corridor
         'odor_start_pos': (float, float) # animal's position at odor onset
    '''
    if df[df['instrip']].shape[0]==0:
        #print(df.shape)
        entry_left=False
        odor_xmin = -odor_width/2.
        odor_xmax = odor_width/2.
        odor_start_time = df.iloc[0]['time']
        odor_start_posx, odor_start_posy = (0, 0)
        currdf = df.copy()
    else:
        if entry_ix is None:
            entry_ix = df[df['instrip']].iloc[0].name
        entry_left = check_entry_left(df, entry_ix=entry_ix)
    
        currdf = df.loc[entry_ix:].copy()
        if is_grid and entry_left is not None: # entry_left must be true or false
            if get_all_borders:
                ogrid, in_odor = get_odor_grid(currdf, odor_width=odor_width, grid_sep=grid_sep,
                                    use_crossings=True, verbose=False)
                odor_borders = list(ogrid.values())
            else:
                if entry_left:
                    odor_xmin = currdf[currdf['instrip']].iloc[0]['ft_posx'] 
                    odor_xmax = currdf[currdf['instrip']].iloc[0]['ft_posx'] + odor_width
                else: # entered right, so entry point is largest val
                    odor_xmin = currdf[currdf['instrip']].iloc[0]['ft_posx'] - odor_width
                    odor_xmax = currdf[currdf['instrip']].iloc[0]['ft_posx'] 
                odor_borders = (odor_xmin, odor_xmax)
        else:
            odor_xmin = currdf[currdf['instrip']].iloc[0]['ft_posx'] - (odor_width/2.)
            odor_xmax = currdf[currdf['instrip']].iloc[0]['ft_posx'] + (odor_width/2.)
            odor_borders = (odor_xmin, odor_xmax)

        odor_start_time = currdf[currdf['instrip']].iloc[0]['time']
        odor_start_posx = currdf[currdf['instrip']].iloc[0]['ft_posx']
        odor_start_posy = currdf[currdf['instrip']].iloc[0]['ft_posy']

    trial_start_time = currdf.iloc[0]['time']

    if check_odor:
        assert odor_start_time == currdf.iloc[df[mfc_var].argmax()]['time'],\
            "ERR: odor start time does not match MFC switch time!"

    odor_params = {
                    'trial_start_time': trial_start_time,
                    'odor_start_time': odor_start_time,
                    'odor_boundary': odor_borders, #(odor_xmin, odor_xmax),
                    'odor_start_pos': (odor_start_posx, odor_start_posy),
                    'entry_left': entry_left
                    } 

    return odor_params

# ---------------------------------------------------------------------- 
# Data processing
# ----------------------------------------------------------------------
def process_df(df0, xvar='ft_posx', yvar='ft_posy', 
                conditions=None, bout_thresh=0.5, 
                smooth=False, window_size=11):
    if conditions is not None:
        df = df0[df0['condition'].isin(conditions)].copy()
    else:
        df = df0.copy()
    dlist=[]
    for (fly_id, cond), df_ in df.groupby(['fly_id', 'condition']):
        # parse in and out bouts
        df_ = parse_bouts(df_, count_varname='instrip', bout_varname='boutnum') # 1-count
        # filter in and out bouts by min. duration 
        df_ = filter_bouts_by_dur(df_, bout_thresh=bout_thresh, 
                            bout_varname='boutnum', count_varname='instrip', verbose=False)
        df_ = calculate_speed(df_, xvar=xvar, yvar=yvar) # smooth=False, window_size=11, return_same=True)
        df_ = calculate_distance(df_, xvar=xvar, yvar=yvar)
        if smooth:
            #for varname in ['ft_posx', 'ft_posy']:
            df_ = smooth_traces(df_, window_size=window_size, return_same=True)
        dlist.append(df_)

    DF=pd.concat(dlist, axis=0).reset_index(drop=True)

    return DF

def calculate_turn_angle(df, xvar='ft_posx', yvar='ft_posy'):
    '''
    Calculate angle bw positions. 

    Arguments:
        df -- _description_

    Keyword Arguments:
        xvar -- _description_ (default: {'ft_posx'})
        yvar -- _description_ (default: {'ft_posy'})

    Returns:
        _description_
    '''
    #ang_ = np.arctan2(np.gradient(df[yvar].values), np.gradient(df[xvar].values))
    df['turn_angle'] = np.arctan2(np.gradient(df[yvar]), np.gradient(df[xvar]))
    #ang_ = np.arctan2(df[yvar].diff(), df[xvar].diff())
    #df['turn_angle'] = ang_
    
    return df

def calculate_speed(df0, xvar='ft_posx', yvar='ft_posy'):
    xv = np.gradient(df0[xvar]) #/df0['time'].diff().mean()
    yv = np.gradient(df0[yvar]) #/df0['time'].diff().mean()
    tv = np.gradient(df0['time'])
    avg_tdiff = df0['time'].diff().mean()

    speed = np.linalg.norm(np.array([xv, yv]), axis=0)/tv #avg_tdiff #np.sqrt(xv**2+yv**2)
    df0['rel_time'] = df0['time'] - df0['time'].iloc[0]
    df0['cum_time'] = df0['rel_time'].cumsum()
    df0['speed'] = speed
    df0['upwind_speed'] = yv/tv #avg_tdiff
    df0['crosswind_speed'] = xv/tv #avg_tdiff

    return df0

def calculate_speed2(df0, smooth=True, window_size=11, return_same=True):
    '''
    Calculate instantaneous speed from pos1 to pos2 using linalg.norm().

    Arguments:
        df0 (pd.DataFrame) : behavior .log output, must have 'ft_posx', 'ft_posy', 'time'

    Keyword Arguments:
        return_same (bool) : add speed column to original df or as separate df (default: {True})

    Returns:
        Either the same df0 with column added, or diff df.
    '''

    diff_df = df0[['ft_posx', 'ft_posy', 'time']].diff()
    diff_df['speed'] = np.linalg.norm(diff_df[['ft_posx', 'ft_posy']], axis=1)\
                        /diff_df['time'] # inst. vel.
    diff_df['cum_time'] = diff_df['time'].cumsum() # Get relative time  

    if smooth:
        diff_df = smooth_traces_each(diff_df, varname='speed', window_size=window_size)

    if return_same:
        df0['speed'] = diff_df['speed']
        df0['cum_time'] = diff_df['cum_time']
        if smooth:
            df0['smoothed_speed'] = diff_df['smoothed_speed'] 
            
        return df0
    else:
        return diff_df

def calculate_distance(df, xvar='ft_posx', yvar='ft_posy'):
    df['euclid_dist'] = np.linalg.norm(df[[xvar, yvar]].diff(axis=0), axis=1)
    df['upwind_dist'] = df[yvar].diff()
    df['crosswind_dist'] = df[xvar].diff().abs()
    return df

def calculate_stops(df, stop_thresh=1.0, speed_varname='smoothed_speed'):
    '''
    Find bouts where animal is stopped (speed < stop_thresh, in mm/s).
    Assumes speed has been calculated.

    Arguments:
        df (pd.DataFrame) -- processed behavior (must have 'speed' or 'smoothed_speed')

    Keyword Arguments:
        stop_thresh -- speed in mm/s below which is considered a stop (default: {1.0})
    
    Returns:
        df with columns:
        'stopped' (bool) -- is speed < stop_thresh
    '''
    if 'boutnum' not in df.columns:
        df = filter_bouts_by_dur(df, bout_thresh=0.5, bout_varname='boutnum', 
                        count_varname='instrip', verbose=False)
    df['stopped'] = False
    for boutnum, df_ in df.groupby('boutnum'):
        df.loc[df_.index, 'stopped'] = df_[speed_varname] < stop_thresh
        #df['stopped'] = df[speed_varname] < stop_thresh

    return df

def parse_bouts(df, count_varname='instrip', bout_varname='boutnum', verbose=False):
    '''
    Group consecutive values of a boolean state to parse bouts.

    Arguments:
        df -- .log behavior dataframe, must have <count_varname> in columns.
        count_varname (str) -- column name (must be boolean) to count consecutive instances of
        bout_varname (str) -- desired count variable column name

    Returns:
        df[bout_varname] -- integers grouping each bout and non-bout (1-instrip, 2-outstrip, 3-instrip, etc.)
    '''
    if count_varname == 'instrip':
        bout_varname = 'boutnum'
    elif count_varname == 'stopped':
        bout_varname = 'stopboutnum'
    assert count_varname in df.columns, "ERR: Nothing to pares, var <{}> not found.".format(count_varname)

    if bout_varname in df.columns and verbose:
        print("WARNING: Column {} already exists. Overwriting...".format(bout_varname))

    #new_varname = '{}X'.format(count_varname)
    #df[new_varname] = df[count_varname].shift()
    #df[bout_varname] = (df[count_varname] != df[new_varname]).cumsum()

    df[bout_varname] = (df[count_varname] != df[count_varname].shift()).cumsum()

    return df


def get_bout_durs(df, bout_varname='boutnum'):
    '''
    Get duration of parsed bouts. 
    Parse with parse_bouts(count_varname='instrip', bout_varname='boutnum').

    Arguments:
        df -- behavior dataframe, must have 'boutnum' as column (run parse_inout_bouts())  

    Returns:
        dict, keys=boutnum, vals=boutdur (in sec)
    '''
    assert 'boutnum' in df.columns, "Bouts not parse. Run:  df=parse_inout_bouts(df)"

    boutdurs={}
    grouper = ['boutnum']
    for boutnum, df_ in df.groupby(bout_varname):
        boutdur = df_.iloc[-1]['time'] - df_.iloc[0]['time']
        boutdurs.update({boutnum: boutdur})

    return boutdurs

def filter_bouts_by_dur(df, bout_thresh=0.5, bout_varname='boutnum', 
                        count_varname='instrip', speed_varname='smoothed_speed', 
                        verbose=False):
    '''
    Calculate bout durs, and ignore bouts that are too short (0.5 sec default).
    Overwrites too-short bouts with previous bout (assigns In/Out strip).

    Arguments:
        df -- _description_

    Keyword Arguments:
        bout_thresh -- _description_ (default: {0.5})
        bout_varname -- _description_ (default: {'boutnum'})
        count_varname -- _description_ (default: {'instrip'})
        speed_varname -- _description_ (default: {'smoothed_speed'})
        verbose -- _description_ (default: {False})

    Returns:
        _description_
    '''
    if count_varname == 'instrip':
        bout_varname = 'boutnum'
    elif count_varname == 'stopped':
        bout_varname = 'stopboutnum'

    if bout_varname not in df.columns:
        df = parse_bouts(df, count_varname=count_varname, bout_varname=bout_varname)
    # Calc bout durations
    boutdurs = get_bout_durs(df, bout_varname=bout_varname)
    too_short = [k for k, v in boutdurs.items() if v < bout_thresh]

    if verbose:
        print("Found {} bouts too short (thr={:.2f} sec)".format(len(too_short), bout_thresh))
    # Check for too short bouts
    assert df[count_varname].dtype == bool, "ERR: State <{}> is not bool.".format(count_varname)

    while len(too_short) > 0:
        for boutnum, df_ in df.groupby(bout_varname):
            if boutdurs[boutnum] < bout_thresh:
                # opposite of whatever it is
                df.loc[df_.index, count_varname] = ~df_[count_varname]
        # reparse bouts
        df = parse_bouts(df, count_varname=count_varname, bout_varname=bout_varname)
        # calc bout durations
        boutdurs = get_bout_durs(df, bout_varname=bout_varname)
        too_short = [k for k, v in boutdurs.items() if v < bout_thresh]

    return df


def smooth_traces_each(df, varname='speed', window_size=11, return_same=True):
    smooth_t = util.temporal_downsample(df[varname], window_size)

    #df[new_varname] = util.smooth_timecourse(df[varname], window_size)

    if return_same:
        new_varname = 'smoothed_{}'.format(varname)
        df[new_varname] = smooth_t
        return df
    else:
        return smooth_t

def smooth_traces(df, xvar='ft_posx', yvar='ft_posy', window_size=13, return_same=True):
    for v in [xvar, yvar]:
        df = smooth_traces_each(df, varname=v, window_size=window_size, return_same=True)

    return df

def smooth_traces_interp(df, xvar='ft_posx', yvar='ft_posy', window_size=13, return_same=True):

    arr = df[[xvar, yvar]].values
    x, y = zip(*arr)
    #create spline function
    x = np.arange(0, len(y))
    x = df['time'].values
    f, u = sp.interpolate.splprep([x, y], s=window_size, per=False)
    #create interpolated lists of points
    #npts = int(np.round(len(x)*0.25))
    xint, yint = sp.interpolate.splev(np.linspace(0, 1, len(x)), f)

    if return_same:
        df['smoothed_{}'.format(xvar)] = xint
        df['smoothed_{}'.format(yvar)] = yint
        return df
    else:
        return xint, yint

# checks
def get_odor_grid_all_flies(df0, odor_width=50, grid_sep=200):
    odor_borders={}
    for trial_id, currdf in df0.groupby(['trial_id']):
        ogrid, in_odor = get_odor_grid(currdf, odor_width=odor_width, grid_sep=grid_sep,
                                    use_crossings=True, verbose=False)
        if not in_odor:
            print(trial_id, "WARNING: Fly never in odor (cond={})".format(currdf['condition'].unique()))
        try:
            odor_borders.update({trial_id: ogrid})
            #(odor_xmin, odor_xmax), = ogrid.values()
        except Exception as e:
            #traceback.print_exc()
            print(e)
            print(ogrid)
        #odor_borders.update({trial_id: (odor_xmin, odor_xmax)})

    return odor_borders

 
def find_odor_grid(df, odor_width=10, grid_sep=200): #use_crossings=True,
#                   use_mean=True, verbose=True):
    '''
    Finds the odor boundaries based on odor width and grid separation

    Arguments:
        df -- _description_

    Keyword Arguments:
        odor_width -- _description_ (default: {10})
        grid_sep -- _description_ (default: {200})

    Returns:
        _description_
    '''
    # get first odor entry
    assert len(df['instrip'].unique())==2, "Fly not in odor."
    curr_odor_xmin = df[df['instrip']].iloc[0]['ft_posx'] - (odor_width/2.)
    curr_odor_xmax = df[df['instrip']].iloc[0]['ft_posx'] + (odor_width/2.)

    # identify other grid crossings
    indf = df[df['instrip']].copy()
    # initiate grid dict
    odor_grid = {'c{}'.format(indf.iloc[0].name): (curr_odor_xmin, curr_odor_xmax)}

    indf = df[df['instrip']].copy()
    # where is the fly outside of current odor boundary but still instrip:
    # nextgrid_df = indf[ (indf['ft_posx']>(curr_odor_xmax.round(2)+grid_sep*0.5) | ((indf['ft_posx']<curr_odor_xmin.round(2)-grid_sep*0.5))].copy()
    nextgrid_df = indf[ (indf['ft_posx']>np.ceil(curr_odor_xmax)+grid_sep*0.5) \
                   | ((indf['ft_posx']<np.floor(curr_odor_xmin)-grid_sep*0.5)) ].copy()

    # loop through the remainder of odor strips in experiment until all strips found
    while nextgrid_df.shape[0] > 0:
        # get odor params of next corridor
        last_ix = nextgrid_df[nextgrid_df['instrip']].iloc[0].name
        next_odorp = get_odor_params(df, odor_width=odor_width, 
                            entry_ix=last_ix, is_grid=True, get_all_borders=False)
        # update odor param dict
        odor_grid.update({'c{}'.format(last_ix): (next_odorp['odor_boundary'])})
        (curr_odor_xmin, curr_odor_xmax) = next_odorp['odor_boundary']
        # look for another odor corridor (outside of current odor boundary, but instrip)
        nextgrid_df = indf[ (indf['ft_posx'] >= (curr_odor_xmax+grid_sep)) \
                        | ((indf['ft_posx'] <= (curr_odor_xmin-grid_sep))) ]\
                        .loc[last_ix:].copy()

    return odor_grid

def get_odor_grid(df, odor_width=10, grid_sep=200, use_crossings=True,
                    use_mean=True, verbose=True):
    if df[df['instrip']].shape[0]==0:
        # fly never in odor
        if verbose:
            print("WARNING: Fly is never in odor, using default corridor.")
        curr_odor_xmin = -odor_width/2.
        curr_odor_xmax = odor_width/2.
        odor_grid = {'c0': (curr_odor_xmin, curr_odor_xmax)}
        odor_flag = False
    else:
        odor_grid = find_odor_grid(df, odor_width=odor_width, grid_sep=grid_sep)
        odor_grid = check_odor_grid(df, odor_grid, odor_width=odor_width, grid_sep=grid_sep, 
                        use_crossings=use_crossings, use_mean=use_mean, verbose=verbose)
        odor_flag = True

    return odor_grid, odor_flag

def check_odor_grid(df, odor_grid, odor_width=10, grid_sep=200, use_crossings=True,
                    use_mean=True, verbose=True):
    '''
    Use actual edge crossings to get odor boundary.
    Standard way is to do +/- half odor width at position of odor onset.

    Arguments:
        df (pd.DataFrame) : parsed .log file, must have instrip, ft_posx, ft_posy

    Keyword Arguments:
        odor_width -- odor corridor width in mm (default: {10, specific to imaging; typically, 50mm})
        grid_sep -- separation in mm of odor corridors (default: {200})
        use_crossings - Use actual crossings (mean or min/max) for odor boundaries, not just 1st time in odor
        use_mean -- Use mean values or min/max values of crossings (default: {True})
        verbose -- print stuff or no (default: {True})

    Returns:
        (odor_xmin, odor_xmax) -- tuple, empirically estimated odor boundary based on animal crossings
        odor_grid -- dict, all found odor corridors 
    '''

    if use_crossings:
        bad_corridors=[]
        # Identify true odor boundaries based on these estimated coords.
        for cnum, (curr_odor_xmin, curr_odor_xmax) in odor_grid.items():
            curr_grid_ix = int(cnum[1:])
            # check if there was any crossing..
            crossed_edge = are_there_crossings(df, curr_odor_xmin, curr_odor_xmax)
            #print("Crossings? {}".format(crossed_edge))
            # if crossings detected, find traveled max/min 
            if crossed_edge:
                traveled_xmin, traveled_xmax = get_boundary_from_crossings(df, 
                                                curr_odor_xmin, curr_odor_xmax,
                                                ix=curr_grid_ix, odor_width=odor_width, grid_sep=grid_sep, 
                                                use_mean=use_mean)
                if verbose:
                    print('... {}: min {:.2f} vs {:.2f}'.format(cnum, curr_odor_xmin, traveled_xmin))
                    print('... {}: max {:.2f} vs {:.2f}'.format(cnum, curr_odor_xmax, traveled_xmax))
                    print("... True diff: {:.2f}".format(traveled_xmax - traveled_xmin))
            else:
                traveled_xmin = curr_odor_xmin
                traveled_xmax = curr_odor_xmax
                if verbose:
                    print("... this fly never crossed the edge")
            ctr = curr_odor_xmin + (curr_odor_xmax - curr_odor_xmin)/2.

            if not crossed_edge:
                # at start of odor, animal doesnt move
                print("... Using default odor min/max, animal did not move in odor")
                traveled_xmin = curr_odor_xmin
                traveled_xmax = curr_odor_xmax
            else:
                if traveled_xmax < ctr+odor_width*0.5: # animal never crosses right side
                    traveled_xmax = curr_odor_xmax
                    if verbose:
                        print("... setting travel xmax: {:.2f}".format(curr_odor_xmax))
                if traveled_xmin > ctr-odor_width*0.5:
                    traveled_xmin = curr_odor_xmin # animal never crosses left side (?)
                    if verbose:
                        print("... setting travel xmin: {:.2f}".format(curr_odor_xmin))
            # check width
            if abs(odor_width - (traveled_xmax - traveled_xmin)) < odor_width*0.25:
                if verbose:
                    print("... {} updating: ({:.2f}, {:.2f})".format(cnum, traveled_xmin, traveled_xmax))
                odor_grid.update({cnum: (traveled_xmin, traveled_xmax)})
            else:
                bad_corridors.append(cnum)
                if verbose:
                    print("... Difference was: {:.2f}".format(abs(odor_width - (traveled_xmax - traveled_xmin))))
                    print("... Skipping current boundary crossing")

        for b in bad_corridors:
            odor_grid.pop(b)

        unique_max=None; unique_min=None;
        all_mins = [min(v) for k, v in odor_grid.items()]
        all_maxs = [max(v) for k, v in odor_grid.items()]
        if len(all_mins)>1 and max(abs(np.diff(all_mins))) < 1:
            unique_min = np.mean(all_mins)
        #if len(all_maxs)>1 and max(abs(np.diff(all_maxs))) < 1:
            unique_max = np.mean(all_maxs)
            
        if unique_max is not None or unique_min is not None:
            print(odor_grid.keys())
            odor_grid.update({'c0': (unique_min, unique_max)})
            remove_keys = [k for k, v in odor_grid.items() if k!='c0']
            for k in remove_keys:
                odor_grid.pop(k)
               
    return odor_grid

def are_there_crossings(currdf, curr_odor_xmin, curr_odor_xmax):

    start_ix = currdf[currdf['instrip']].iloc[0].name
    always_under_max = currdf.loc[start_ix:]['ft_posx'].max() < curr_odor_xmax
    always_above_min = currdf.loc[start_ix:]['ft_posx'].min() > curr_odor_xmin

    if always_under_max and always_above_min:
        return False
    else: 
        return True


def get_boundary_from_crossings(df, curr_odor_xmin, curr_odor_xmax, ix=0,
                    grid_sep=200, odor_width=10, use_mean=True,
                    verbose=False):
    # get left and right edge crossings
    right_xings = df[(df['ft_posx'] <= (curr_odor_xmax+odor_width*.5)) \
                & (df['ft_posx'] >= (curr_odor_xmax-odor_width*.5))].loc[ix:].copy()
    left_xings = df[(df['ft_posx'] <= (curr_odor_xmin+odor_width*.5)) \
                & (df['ft_posx'] >= (curr_odor_xmin-odor_width*.5))].loc[ix:].copy()

    # get in/out bouts
    right_xings = parse_bouts(right_xings, count_varname='instrip')
    left_xings = parse_bouts(left_xings, count_varname='instrip')

    # get index of each bout's 1st entry, make sure to start from outstrip
    # ... left side
    first_in_ixs_left=[]
    if left_xings.shape[0]>0:
        starts_instrip = left_xings.iloc[0]['instrip']
        for bi, (bnum, bdf) in enumerate(left_xings[left_xings['instrip']].groupby('boutnum')):
            if starts_instrip and bi==0: # special case of crossing, get last instri and first outstrip
                in_ix = bdf.sort_values(by='time').iloc[-1].name
                if in_ix<df.iloc[-1].name: # found index is smaller than last index 
                    first_in_ixs_left.extend([in_ix, in_ix+1])
                continue
            in_ix = bdf.iloc[0].name
            first_in_ixs_left.extend([in_ix-1, in_ix])
    # ... right side
    first_in_ixs_right=[]
    if right_xings.shape[0]>0:
        starts_instrip = right_xings.iloc[0]['instrip']
        for bi, (bnum, bdf) in enumerate(right_xings[right_xings['instrip']].groupby('boutnum')):
            if starts_instrip and bi==0: # special case of crossing, get last instri and first outstrip
                in_ix = bdf.sort_values(by='time').iloc[-1].name
                if in_ix<df.iloc[-1].name: # found index is smaller than last index 
                    first_in_ixs_right.extend([in_ix, in_ix+1])
                continue
            in_ix = bdf.iloc[0].name
            first_in_ixs_right.extend([in_ix-1, in_ix])

    # select xmin/xmax based on actual travel positions
    if len(first_in_ixs_right)>0 and len(first_in_ixs_left)==0: # animal always on larger edge
        traveled_xmax = df.loc[first_in_ixs_right]['ft_posx'].mean() \
            if use_mean else df.loc[first_in_ixs_right]['ft_posx'].max()
        traveled_xmin = traveled_xmax - odor_width
    elif len(first_in_ixs_right)>0 and len(first_in_ixs_left)>0: # animal crosses at least once
        #print("got both")
        traveled_xmax = df.loc[first_in_ixs_right]['ft_posx'].mean() \
            if use_mean else df.loc[first_in_ixs_right]['ft_posx'].max()
        traveled_xmin = df.loc[first_in_ixs_left]['ft_posx'].mean() \
            if use_mean else df.loc[first_in_ixs_left]['ft_posx'].min() #traveled_xmax - odor
    else:
        # len(onright)==0 and len(onleft)>0 # animal always on smaller/left edge
        traveled_xmin = df.loc[first_in_ixs_left]['ft_posx'].mean() \
            if use_mean else df.loc[first_in_ixs_left]['ft_posx'].min()
        traveled_xmax = traveled_xmin + odor_width
    #set_odor_min = traveled_xmin if traveled_xmin < curr_odor_xmin else curr_odor_xmin
    #set_odor_max = traveled_xmax if traveled_xmax > curr_odor_xmax else curr_odor_xmax
    #set_odor_min = traveled_xmin
    #set_odor_max = traveled_xmax

    return traveled_xmin, traveled_xmax 

# Ramer-Douglas-Peuker functions (RDP)
# --------------------------------------------------------------------
def dsquared_line_points(P1, P2, points):
    '''
    Calculate only squared distance, only needed for comparison
    http://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    '''
    xdiff = P2[0] - P1[0]
    ydiff = P2[1] - P1[1]
    nom  = (
        ydiff*points[:,0] - \
        xdiff*points[:,1] + \
        P2[0]*P1[1] - \
        P2[1]*P1[0]
    )**2
    denom = ydiff**2 + xdiff**2
    return np.divide(nom, denom)

def rdp_numpy(M, epsilon = 0):

    # initiate mask array
    # same amount of points
    mask = np.empty(M.shape, dtype = bool)

        # Assume all points are valid and falsify those which are found
    mask.fill(True)

    # The stack to select start and end index
    stack = [(0 , M.shape[0]-1)]

    while (len(stack) > 0):
        # Pop the last item
        (start, end) = stack.pop()

        # nothing to calculate if no points in between
        if end - start <= 1:
            continue

        # Calculate distance to points
        P1 = M[start]
        P2 = M[end]
        points = M[start + 1:end]
        dsq = dsquared_line_points(P1, P2, points)

        mask_eps = dsq > epsilon**2

        if mask_eps.any():
            # max point outside eps
            # Include index that was sliced out
            # Also include the start index to get absolute index
            # And not relative 
            mid = np.argmax(dsq) + 1 + start
            stack.append((start, mid))
            stack.append((mid, end))

        else:
            # Points in between are redundant
            mask[start + 1:end] = False

    return mask

def rdp_mask(df, epsilon=0.1, xvar='ft_posx', yvar='ft_posy'):
    M = df[[xvar, yvar]].values
    simp = rdp_numpy(M, epsilon = epsilon)

    return simp

def add_rdp_by_bout(df_, epsilon=0.1, xvar='ft_posx', yvar='ft_posy'):
    df_['rdp_{}'.format(xvar)] = None
    df_['rdp_{}'.format(yvar)] = None
    for b, b_ in df_.groupby(['condition', 'boutnum']):
        simp = rdp_mask(b_, epsilon=epsilon, xvar=xvar, yvar=yvar)
        df_.loc[b_.index, 'rdp_{}'.format(xvar)] = simp[:, 0]
        df_.loc[b_.index, 'rdp_{}'.format(yvar)] = simp[:, 1]
    return df_


def get_rdp_distances(df_, rdp_var='rdp_ft_posx'):
    dists_=[]
    for bi, (bnum, b_) in enumerate(df_.groupby('boutnum')):
        rdp_points = b_[b_[rdp_var]].shape[0]
        #if rdp_points <=2 :
        total_dist = b_['euclid_dist'].sum()
        r_ = pd.DataFrame({'boutnum': bnum,
                           'rdp_points': rdp_points,
                           'euclid_dist': b_['euclid_dist'].sum()-b_['euclid_dist'].iloc[0],
                           'upwind_dist': b_['upwind_dist'].sum()-b_['upwind_dist'].iloc[0],
                           'crosswind_dist': b_['crosswind_dist'].sum()-b_['crosswind_dist'].iloc[0],
                          },
                           index=[bi]
                          )
        dists_.append(r_)
    rdp_dists = pd.concat(dists_)

    return rdp_dists

def plot_overlay_rdp_v_smoothed(b_, ax, xvar='ft_posx', yvar='ft_posy', epsilon=1):

    if 'rdp_{}'.format(xvar) not in b_.columns:
        add_rdp_by_bout(b_, epsilon=epsilon, xvar=xvar, yvar=yvar)

    rdp_x = 'rdp_{}'.format(xvar)
    rdp_y = 'rdp_{}'.format(yvar)
    ax.plot(b_['ft_posx'], b_['ft_posy'], 'w', alpha=1, lw=0.5)
    ax.plot(b_[b_[rdp_x]][xvar], b_[b_[rdp_y]][yvar], 'r', alpha=1, lw=0.5)
    if 'smoothed_ft_posx' in b_.columns:
        ax.plot(b_['smoothed_ft_posx'], b_['smoothed_ft_posy'], 'cornflowerblue', alpha=0.7)
    ax.scatter(b_[b_[rdp_x]][xvar], b_[b_[rdp_y]][yvar], 
               c=b_[b_[rdp_x]]['speed'], alpha=1, s=3)


def plot_overlay_rdp_v_smoothed_multi(df_, boutlist=None, nr=4, nc=6, distvar=None,
                                rdp_epsilon=1.0, smooth_window=11, xvar='ft_posx', yvar='ft_posy'):
    if boutlist is None:
        #boutlist = list(np.arange(1, nr*nc))
        nbouts_plot = nr*nc
        boutlist = df_['boutnum'].unique()[0:nbouts_plot]
    fig, axes = pl.subplots(nr, nc, figsize=(nc*2, nr*1.5))
    for ax, bnum in zip(axes.flat, boutlist):
        b_ = df_[(df_['boutnum']==bnum)].copy()
        plot_overlay_rdp_v_smoothed(b_, ax, xvar=xvar, yvar=yvar)
        if distvar is not None:
            dist_traveled = b_[distvar].sum()-b_[distvar].iloc[0]
            ax.set_title('{}: {:.2f}'.format(bnum, dist_traveled))
        else:
            ax.set_title(bnum)
    for ax in axes.flat:
        ax.set_aspect('equal')
        ax.axis('off')
    legh = [mpl.lines.Line2D([0], [0], color='w', lw=2, label='orig'),
           mpl.lines.Line2D([0], [0], color='r', lw=2, label='RDP ({})'.format(rdp_epsilon))]
    if 'smoothed_ft_posx' in b_.columns: 
        legh.append(mpl.lines.Line2D([0], [0], color='b', lw=2, label='smoothed ({})'.format(smooth_window)))
   
    axes.flat[nc-1].legend(handles=legh, bbox_to_anchor=(1,1), loc='upper left')
    return fig


# TURN ANGLES AND HEADING
# --------------------------------------------------------------------

def convert_cw(v):
    vv=v.copy()
    vv[v<0] += 2*np.pi
#     if angle < 0:
#         angle += 2 * math.pi
    #vv = (180. / np.pi) * v
    return vv  #(180 / math.pi) * angle


def examine_heading_in_bout(b_, theta_range=(0, 2*np.pi), xvar='ft_posx', yvar='ft_posy'):
    fig, axn = pl.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)
    ax=axn[0, 0]
    sns.scatterplot(data=b_, x="ft_posx", y="ft_posy", ax=ax,
                    hue='time', s=4, edgecolor='none', palette='viridis')
    ax.legend(bbox_to_anchor=(-0.1, 1.4), ncols=2, loc='upper left', title='time')
    # ---------------------
    ax=axn[0, 1]
    sns.scatterplot(data=b_, x="ft_posx", y="ft_posy", ax=ax,
                    hue='ft_heading_deg', s=4, edgecolor='none', palette='hsv')
    ax.legend(bbox_to_anchor=(-0.1, 1.4), ncols=2, loc='upper left', title='ft_heading')
    #theta_range = (0, 2*np.pi)
    cax = util.add_colorwheel(fig, axes=[0.75, 0.5, 0.25, 0.25], theta_range=theta_range, cmap='hsv') 
    # ---------------------
    ax=axn[1, 0]; ax.set_title('arctan2')
    rdp_x ='rdp_{}'.format(xvar)
    rdp_y ='rdp_{}'.format(yvar)
    xv = b_[b_[rdp_x]][xvar]
    yv = b_[b_[rdp_y]][yvar]
    angles = convert_cw(np.arctan2(np.gradient(xv*3), np.gradient(yv*3)) )
    assert angles.min().round(1)==0, "Min ({:.2f}) is not 0".format(angles.min())
    assert angles.max().round(1)==round(2*np.pi, 1), "Min ({:.2f}) is not 2pi".format(angles.max())
    # print('cw arctan2: ({:.2f}, {:.2f})'.format(angles.min(), angles.max()))
    # -- 
    rdp_var = rdp_x
    ax.plot(b_[xvar], b_[yvar], 'w', lw=0.5)
    ax.scatter(b_[b_[rdp_x]][xvar], b_[b_[rdp_y]][yvar], 
            c=angles, cmap='hsv')
    xy = b_[b_[rdp_var]][[xvar, yvar]].values
    xy = xy.reshape(-1, 1, 2)
    segments = np.hstack([xy[:-1], xy[1:]])
    coll = mpl.collections.LineCollection(segments, cmap='hsv') #plt.cm.gist_ncar)
    coll.set_array(angles) #np.random.random(xy.shape[0]))
    ax.add_collection(coll)
    # ---------------------
    # mean angles
    ixs = b_[b_[rdp_var]].index.tolist()
    mean_angles=[] #sts.circmean(b_.loc[ix:ixs[i+1]]['ft_heading'], high=2*np.pi, low=0)]
    for i, ix in enumerate(ixs[0:-1]):
        ang = sts.circmean(b_.loc[ix:ixs[i+1]]['ft_heading'], high=2*np.pi, low=0)
        mean_angles.append(ang)
        if i==0:
            mean_angles.append(ang)
    mean_angles = np.array(mean_angles)
    assert mean_angles.min().round(1)==0, "Min ({:.2f}) is not 0".format(mean_angles.min())
    assert mean_angles.max().round(1)==round(2*np.pi, 1), "Min ({:.2f}) is not 2pi".format(mean_angles.max())
    # print('mean angles: ({:.2f}, {:.2f})'.format(min(mean_angles), max(mean_angles)))
    # ---
    ax=axn[1,1]; ax.set_title('mean angles')
    ax.plot(b_[xvar], b_[yvar], 'w', lw=0.5)
    ax.scatter(b_[b_[rdp_x]][xvar], b_[b_[rdp_y]][yvar], 
            c=mean_angles, cmap='hsv')
    xy = b_[b_[rdp_var]][[xvar, yvar]].values
    xy = xy.reshape(-1, 1, 2)
    segments = np.hstack([xy[:-1], xy[1:]])
    col2 = mpl.collections.LineCollection(segments, cmap='hsv') #plt.cm.gist_ncar)
    col2.set_array(mean_angles) #np.random.random(xy.shape[0]))
    ax.add_collection(col2)
    return fig


def examine_heading_at_stops(b_, xvar='ft_posx', yvar='ft_posy'):
    fig, axn = pl.subplots(1, 3, figsize=(9, 4), sharex=True, sharey=True)
    ax=axn[0]
    cmap = pl.get_cmap("viridis")
    b_['time'] -= b_['time'].iloc[0]
    norm = pl.Normalize(b_['time'].min(), b_['time'].max())
    sns.scatterplot(data=b_, x="ft_posx", y="ft_posy", ax=ax,
                    hue='time', s=3, edgecolor='none', palette=cmap, legend=False)
    #ax.legend(bbox_to_anchor=(-0.2, 1.), ncols=2, loc='lower left', title='time')
    sm =  mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.ax.set_title("time", fontsize=10)
    cbar.ax.tick_params(labelsize=10)
    # ---------------------
    ax=axn[1]
    if b_.shape[0]>10000:
        skip_every=20
        print("skipping some")
    else:
        skip_every=1
    sns.scatterplot(data=b_.iloc[0::skip_every], x="smoothed_ft_posx", y="smoothed_ft_posy", ax=ax,
                    hue='stopped', s=5, edgecolor='none', palette={True: 'r', False: 'w'}, alpha=0.5)
    n_stops_in_bout = len(b_[b_['stopped']]['stopboutnum'].unique())
    leg = ax.legend(bbox_to_anchor=(1, 1.), ncols=1, loc='upper left', \
              title='stopped (n={})'.format(n_stops_in_bout), fontsize=6)
    ax.get_legend()._legend_box.align = "left"
    pl.setp(leg.get_title(),fontsize='x-small')
    # ---------------------
    ax=axn[2]; #ax.set_title('rdp-heading')
    rdp_x ='rdp_{}'.format(xvar)
    rdp_y ='rdp_{}'.format(yvar)
    xv = b_[b_[rdp_x]][xvar]
    yv = b_[b_[rdp_y]][yvar]
    angles = convert_cw(np.arctan2(np.gradient(xv*3), np.gradient(yv*3)) )
    # -- 
    ax.plot(b_[xvar], b_[yvar], 'w', lw=0.5)
    ax.scatter(b_[b_[rdp_x]][xvar], b_[b_[rdp_y]][yvar], 
            c=angles, cmap='hsv', s=4) 
    xy = b_[b_[rdp_x]][[xvar, yvar]].values
    xy = xy.reshape(-1, 1, 2)
    segments = np.hstack([xy[:-1], xy[1:]])
    coll = mpl.collections.LineCollection(segments, cmap='hsv') #plt.cm.gist_ncar)
    coll.set_array(angles) #np.random.random(xy.shape[0]))
    ax.add_collection(coll)
    # legend
    theta_range = (0, 2*np.pi)
    cax = util.add_colorwheel(fig, axes=[0.8, 0.5, 0.2, 0.2], theta_range=theta_range, cmap='hsv') 
    cax.set_title('rdp-heading', fontsize=10)
    # -----
    pl.subplots_adjust(right=0.8, top=0.7, wspace=0.8, bottom=0.2, left=0.1)

    return fig


# data processing
def get_speed_and_stops(b_, speed_thresh=1.0, stopdur_thresh=0.5,
                        xvar='smoothed_ft_posx', yvar='smoothed_ft_posy'):
    b_ = calculate_speed(b_, xvar=xvar, yvar=yvar)
    b_ = calculate_stops(b_, stop_thresh=speed_thresh, speed_varname='speed')
    b_ = parse_bouts(b_, count_varname='stopped', bout_varname='stopboutnum')
    b_ = filter_bouts_by_dur(b_, bout_thresh=stopdur_thresh, \
                               count_varname='stopped', bout_varname='stopboutnum')
    return b_

def mean_dir_after_stop(df, speed_thresh=1.0, stopdur_thresh=0.5):
    d_list = []
    i=0
    for bnum, b_ in df.groupby('boutnum'):
        # b_ = get_speed_and_stops(b_, speed_thresh=speed_thresh, stopdur_thresh=stopdur_thresh)
        xwind_dist = b_['crosswind_dist'].sum() - b_['crosswind_dist'].iloc[0]
        stopbouts = b_[b_['stopped']]['stopboutnum'].unique()
        #print(bnum, len(stopbouts))
        for snum in stopbouts:
            if b_[b_['stopboutnum']==(snum+1)].shape[0]==0:
                continue
            d_ = pd.DataFrame({
                'fly_id': b_['fly_id'].unique()[0],
                'trial_id': b_['trial_id'].unique()[0],
                'condition': b_['condition'].unique()[0],
                'boutnum': bnum,
                'crosswind_dist': xwind_dist,
                'stopboutnum': snum,
                'meandir': np.rad2deg(sts.circmean(b_[b_['stopboutnum']==(snum+1)]['ft_heading']))},
                index=[i]
            )
            i+=1
            d_list.append(d_)
    meandirs = pd.concat(d_list)
    return meandirs


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------

def plot_trajectory_from_file(fpath, parse_info=False,
            odor_width=10, grid_sep=200, ax=None):
    # load and process the csv data  
    df0 = load_dataframe(fpath, mfc_id=None, verbose=False, cond=None, 
                parse_info=False)
    fly_id=None
    if parse_info:
        # try to parse experiment details from the filename
        exp, datestr, fid, cond = parse_info_from_file(fpath)
        print('Experiment: {}{}Fly ID: {}{}Condition: {}'.format(exp, '\n', fid, '\n', cond))
        fly_id = df0['fly_id'].unique()[0]

    # get experimentally determined odor boundaries:
    ogrid, in_odor = get_odor_grid(df0, odor_width=odor_width, grid_sep=grid_sep,
                            use_crossings=True, verbose=False )
    #(odor_xmin, odor_xmax), = ogrid.values()
    odor_bounds = list(ogrid.values())

    title = os.path.splitext(os.path.split(fpath)[-1])[0]
    print(odor_bounds) 
    plot_trajectory(df0, odor_bounds=odor_bounds, title=title, ax=ax)

    return ax

def plot_trajectory(df0, odor_bounds=[], ax=None,
        hue_varname='instrip', palette={True: 'r', False: 'w'},
        start_at_odor = True, odor_lc='lightgray', odor_lw=0.5, title='',
        markersize=0.5, center=True):

    # ---------------------------------------------------------------------
    if ax is None: 
        fig, ax = pl.subplots()
    if not isinstance(odor_bounds, list):
        odor_bounds = [odor_bounds]
    sns.scatterplot(data=df0, x="ft_posx", y="ft_posy", ax=ax, 
                    hue=hue_varname, s=markersize, edgecolor='none', palette=palette)
    for (odor_xmin, odor_xmax) in odor_bounds:
        plot_odor_corridor(ax, odor_xmin=odor_xmin, odor_xmax=odor_xmax)

    if df0[df0['instrip']].shape[0]>0:
        odor_start_ix = df0[df0['instrip']].iloc[0]['ft_posy']
        ax.axhline(y=odor_start_ix, color='w', lw=0.5, linestyle=':')

    ax.legend(bbox_to_anchor=(1,1), loc='upper left', title=hue_varname)
    ax.set_title(title)
    xmax=500
    if center:
        try:
            # Center corridor
            xmax = np.ceil(df0['ft_posx'].dropna().abs().max())
            ax.set_xlim([-xmax-10, xmax+10])
        except ValueError as e:
            xmax = 500
    pl.subplots_adjust(left=0.2, right=0.8)

    return ax

def plot_odor_corridor(ax, odor_xmin=-100, odor_xmax=100, \
                    odor_linecolor='gray', odor_linewidth=0.5,
                    offset=10):
    ax.axvline(odor_xmin, color=odor_linecolor, lw=odor_linewidth)
    ax.axvline(odor_xmax, color=odor_linecolor, lw=odor_linewidth)
    xmin = min([odor_xmin-offset, min(ax.get_xlim())])
    xmax = max([odor_xmax + offset, min(ax.get_xlim())])
    ax.set_xlim([xmin, xmax])
    #ax.axhline(odor_start_posy, color=startpos_linecolor, 
    #            lw=startpos_linewidth, linestyle=':')


def plot_45deg_corridor(ax, odor_lc='gray', odor_lw=0.5):
    ax.plot([-25, 975], [0,1000], color=odor_lc, lw=odor_lw)
    ax.plot([25, 1025], [0,1000], color=odor_lc, lw=odor_lw)

def center_odor_path(ax, xmin=-500, xmax=500, ymin=-100, ymax=1000):
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

# interactive plotting 

def iplot_odor_corridor(fig, odor_xmin=-100, odor_xmax=100, \
                            odor_linecolor='gray', odor_linewidth=0.5,
                            odor_start_posy=0, startpos_linecolor='gray', 
                            startpos_linewidth=0.5):
    # Set data range
    fig.add_vline(x=odor_xmin, line_color=odor_linecolor, line_width=odor_linewidth)
    fig.add_vline(x=odor_xmax, line_color=odor_linecolor, line_width=odor_linewidth)

    fig.add_hline(y=odor_start_posy, line_color=startpos_linecolor, 
                line_width=startpos_linewidth, line={'dash': 'dot'})

    return fig

def icenter_odor_path(fig, xmin=-500, xmax=500, ymin=-100, ymax=1000):
    # Set data range
    fig.update_layout(yaxis=dict(range=[ymin, ymax], scaleratio=1))
    fig.update_layout(xaxis=dict(range=[xmin, xmax], scaleratio=1))

    return fig

def iplot_trajectory(df0, xvar='ft_posx', yvar='ft_posy', plot_odor=True,\
                        xmin=-500, xmax=500, ymin=-100, ymax=1000,
                        title='Plume trajectory', fpath=None, 
                        path_linecolor='red', path_linewidth=0.5, \
                        odor_xmin=-100, odor_xmax=100,
                        odor_linecolor='gray', odor_linewidth=0.5):
    '''
    Interactive plot (in jupyter, assumes defaults) of animal's trajectory.

    Arguments:
        df0 -- _description_

    Keyword Arguments:
        xvar -- _description_ (default: {'ft_posx'})
        yvar -- _description_ (default: {'ft_posy'})
        fpath -- _description_ (default: {None})
        path_linecolor -- _description_ (default: {'red'})
        path_linewidth -- _description_ (default: {0.5})
        odor_linewidth -- _description_ (default: {0.5})

    Returns:
        _description_
    '''
    fig = px.line(df0, x="ft_posx", y="ft_posy", title=title)
    # This styles the line
    fig.update_traces(line=dict(color=path_linecolor, width=path_linewidth))
    # Set data range
    icenter_odor_path(fig, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    # Plot odor 
    if plot_odor:
        iplot_odor_corridor(fig, odor_xmin=odor_xmin, odor_xmax=odor_xmax, \
                            odor_linecolor=odor_linecolor, odor_linewidth=odor_linewidth)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig


def get_quiverplot_inputs(df_, xvar='ft_posx', yvar='ft_posy'):

    x = df_[xvar].values
    y = df_[yvar].values
    uu = df_[xvar].shift(periods=-1) - df_[xvar]
    vv = df_[yvar].shift(periods=-1) - df_[yvar]

    return x, y, uu, vv
