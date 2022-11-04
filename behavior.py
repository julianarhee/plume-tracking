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

import re
from datetime import datetime
import numpy as np
import pandas as pd

import utils as util
import rdp

# plotting
import plotly.express as px

# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def parse_info_from_file(fpath, experiment=None, 
            rootdir='/Users/julianarhee/Library/CloudStorage/GoogleDrive-edge.tracking.ru@gmail.com/My Drive/Edge_Tracking/Data'):

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
    if re.search('fly\d{1,3}[a-zA-Z]?', cond_str, re.IGNORECASE):
        #fly_id = re.search('fly\d{1,3}[a-zA-Z]?', log_fname, re.IGNORECASE)
        #condition = cond_str.split('{}_'.format(fly_id))[-1]
        condition = [c for c in cond_str.split('{}'.format(fly_id)) \
                        if c!=fly_id and len(c)>1]
        for ci, c in enumerate(condition):
            if c.endswith('_'): 
                condition[ci] = c[:-1]
            elif c.startswith('_'):
                condition[ci] = c[1:]
        condition = condition[0]
    else:
        condition = cond_str
    #print(exp_cond, fly_id, condition)

    if fly_id is not None:
        fly_id = fly_id.lower()
    if condition is not None:
        condition = condition.lower()

    return experiment, fly_id, condition

def load_dataframe(fpath, mfc_id=None, led_id=None, verbose=False, cond='odor',
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

    df0 = pd.read_table(fpath, sep=",", skiprows=[1], header=0, 
              parse_dates=[1]).rename(columns=lambda x: x.strip())
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
    if cond=='reinforced':
        # for newer exp, weird thing where LED signal is 1 for "off" 
        led1_vals = df0[~df0['instrip']]['led1_stpt'].unique() 
        assert len(led1_vals)==1, "Too many out of strip values for LED: {}".format(str(led1_vals))
        if led1_vals[0]==1: # when out of strip, no reinforcement. if has 1 and 0, likely, 1=off
            df0['led_on'] = df0['led1_stpt']==0 # 1 is OFF, and 0 is ON (led2 is always 0)
        elif led1_vals[0]==0:
            df0['led_on'] = df0['led1_stpt']==1 # 1 is ON, and 0 is OFF (led2 is always 0)
    elif cond in ['odor', 'air']:
        df0['led_on'] = False
    elif cond=='light' or cond=='lightonly':
        df0['led_on'] = df0['led1_stpt']==1 # 20221018, quick fix for now bec dont know when things changed

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

    if parse_info:
        # get experiment info
        exp, fly_id, cond = parse_info_from_file(fpath)
        df0['experiment'] = exp
        df0['fly_name'] = fly_id
        df0['condition'] = cond
        if verbose:
            print("Exp: {}, fly ID: {}, cond={}".format(exp, fly_id, cond))

        # make fly_id combo of date, fly_id since fly_id is reused across days
        df0['fly_id'] = ['{}-{}'.format(dat, fid) for (dat, fid) in df0[['date', 'fly_name']].values]

    return df0

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


def get_odor_params(df0, odor_width=50, check_odor=False):
    '''
    Get odor start times, boundary coords, etc.

    Arguments:
        df0 -- formatted dataframe (from load_dataframe())

    Keyword Arguments:
        odor_width -- width of odor corridor in mm (default: {50})

    Returns:
        dict of odor params:
        {'trial_start_time': float
         'odor_start_time': float 
         'odor_boundary': (float, float) # x boundaries of odor corridor
         'odor_start_pos': (float, float) # animal's position at odor onset
    '''
    odor_xmin = df0[df0['instrip']].iloc[0]['ft_posx'] - (odor_width/2.)
    odor_xmax = df0[df0['instrip']].iloc[0]['ft_posx'] + (odor_width/2.)

    trial_start_time = df0.iloc[0]['time']
    odor_start_time = df0[df0['instrip']].iloc[0]['time']

    odor_start_posx = df0[df0['instrip']].iloc[0]['ft_posx']
    odor_start_posy = df0[df0['instrip']].iloc[0]['ft_posy']

    if check_odor:
        assert odor_start_time == df0.iloc[df0['mfc2_stpt'].argmax()]['time'],\
            "ERR: odor start time does not match MFC switch time!"

    odor_params = {
                    'trial_start_time': trial_start_time,
                    'odor_start_time': odor_start_time,
                    'odor_boundary': (odor_xmin, odor_xmax),
                    'odor_start_pos': (odor_start_posx, odor_start_posy)
                    } 

    return odor_params

# ---------------------------------------------------------------------- 
# Data processing
# ----------------------------------------------------------------------
def process_df(df0, xvar='ft_posx', yvar='ft_posy', 
                conditions=['odor', 'reinforced'], bout_thresh=0.5):
    df = df0[df0['condition'].isin(conditions)].copy()
    dlist=[]
    for (fly_id, cond), df_ in df.groupby(['fly_id', 'condition']):
        for varname in ['ft_posx', 'ft_posy']:
            df_ = smooth_traces(df_, varname=varname, window_size=101)
        # parse in and out bouts
        df_ = parse_bouts(df_, count_varname='instrip', bout_varname='boutnum') # 1-count
        # filter in and out bouts by min. duration 
        df_ = filter_bouts_by_dur(df_, bout_thresh=bout_thresh, 
                            bout_varname='boutnum', count_varname='instrip', verbose=False)
        df_ = calculate_speed(df_, xvar=xvar, yvar=yvar) # smooth=False, window_size=11, return_same=True)
        df_ = calculate_distance(df_, xvar=xvar, yvar=yvar)
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

def calculate_speed2(df0, smooth=True, window_size=101, return_same=True):
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
        diff_df = smooth_traces(diff_df, varname='speed', window_size=window_size)

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
        bout_varname = 'stopbout'
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
        bout_varname = 'stopbout'

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


def smooth_traces(df, varname='speed', window_size=101):
    new_varname = 'smoothed_{}'.format(varname)
    #df[new_varname] = util.smooth_timecourse(df[varname], window_size)
    df[new_varname] = util.temporal_downsample(df[varname], window_size)
    return df



# checks 
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
    assert len(df['instrip'].unique())==2
    curr_odor_xmin = df[df['instrip']].iloc[0]['ft_posx'] - (odor_width/2.)
    curr_odor_xmax = df[df['instrip']].iloc[0]['ft_posx'] + (odor_width/2.)
    # identify other grid crossings
    indf = df[df['instrip']].copy()
    # initiate grid dict
    odor_grid = {'c{}'.format(indf.iloc[0].name): (curr_odor_xmin, curr_odor_xmax)}
    # where is the fly outside of current odor boundary but still instrip:
    # nextgrid_df = indf[ (indf['ft_posx']>curr_odor_xmax.round(2)) | ((indf['ft_posx']<curr_odor_xmin.round(2)))].copy()
    nextgrid_df = indf[ (indf['ft_posx']>np.ceil(curr_odor_xmax)) \
                   | ((indf['ft_posx']<np.floor(curr_odor_xmin))) ].copy()

    # loop through the remainder of odor strips in experiment until all strips found
    while nextgrid_df.shape[0] > 0:
        # get odor params of next corridor
        next_odorp = get_odor_params(nextgrid_df, odor_width=odor_width)
        # update odor param dict
        last_ix = nextgrid_df[nextgrid_df['instrip']].iloc[0].name
        odor_grid.update({'c{}'.format(last_ix): (next_odorp['odor_boundary'])})
        curr_odor_xmin, curr_odor_xmax = next_odorp['odor_boundary']
        # look for another odor corridor (outside of current odor boundary, but instrip)
        nextgrid_df = indf[ (indf['ft_posx'] >= (curr_odor_xmax+grid_sep)) \
                        | ((indf['ft_posx'] <= (curr_odor_xmin-grid_sep))) ]\
                        .loc[last_ix:].copy()

    return odor_grid

def get_odor_grid(df, odor_width=10, grid_sep=200, use_crossings=True,
                    use_mean=True, verbose=True):
    odor_grid = find_odor_grid(df, odor_width=odor_width, grid_sep=grid_sep)

    odor_grid = check_odor_grid(df, odor_grid, odor_width=odor_width, grid_sep=grid_sep, 
                        use_crossings=use_crossings, use_mean=use_mean, verbose=verbose)

    return odor_grid

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
            traveled_xmin, traveled_xmax = get_boundary_from_crossings(df, curr_odor_xmin, curr_odor_xmax,
                                                ix=curr_grid_ix, odor_width=odor_width, grid_sep=grid_sep, 
                                                use_mean=use_mean)
            if verbose:
                print('{}: min {:.2f} vs {:.2f}'.format(cnum, curr_odor_xmin, traveled_xmin))
                print('{}: max {:.2f} vs {:.2f}'.format(cnum, curr_odor_xmax, traveled_xmax))
                print("True diff: {:.2f}".format(traveled_xmax - traveled_xmin))
            ctr = curr_odor_xmin + (curr_odor_xmax - curr_odor_xmin)/2.
            if traveled_xmax < ctr+odor_width*0.5: # animal never crosses to right side
                traveled_xmax = curr_odor_xmax
                if verbose:
                    print("setting travel xmax: {:.2f}".format(curr_odor_xmax))
            if traveled_xmin > ctr-odor_width*0.5:
                traveled_xmin = curr_odor_xmin # animal never goes to left side (?)
                if verbose:
                    print("setting travel xmin: {:.2f}".format(curr_odor_xmin))

            if abs(odor_width - (traveled_xmax - traveled_xmin)) < odor_width*0.25:
                if verbose:
                    print("{} updating".format(cnum))
                odor_grid.update({cnum: (traveled_xmin, traveled_xmax)})
            else:
                bad_corridors.append(cnum)
                if verbose:
                    print("Difference was: {:.2f}".format(abs(odor_width - (traveled_xmax - traveled_xmin))))
                    print("Skipping current boundary crossing")

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

# 
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

def rdp_mask(df, epsilon=0.1):
    M = df[['ft_posx', 'ft_posy']].values
    simp = rdp_numpy(M, epsilon = epsilon)

    return simp

def add_rdp_by_bout(df_, epsilon=0.1):
    df_['rdp_posx'] = None
    df_['rdp_posy'] = None
    for b, b_ in df_.groupby(['condition', 'boutnum']):
        simp = rdp_mask(b_, epsilon=epsilon)
        df_.loc[b_.index, 'rdp_posx'] = simp[:, 0]
        df_.loc[b_.index, 'rdp_posy'] = simp[:, 1]
    return df_

# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------
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

def plot_odor_corridor(ax, odor_xmin=-100, odor_xmax=100, \
                            odor_linecolor='gray', odor_linewidth=0.5,\
                            odor_start_posy=0, startpos_linecolor='gray',
                            startpos_linewidth=0.5):
    ax.axvline(odor_xmin, color=odor_linecolor, lw=odor_linewidth)
    ax.axvline(odor_xmax, color=odor_linecolor, lw=odor_linewidth)
    ax.axhline(odor_start_posy, color=startpos_linecolor, 
                lw=startpos_linewidth, linestyle=':')


def plot_45deg_corridor(ax, odor_lc='gray', odor_lw=0.5):
    ax.plot([-25, 975], [0,1000], color=odor_lc, lw=odor_lw)
    ax.plot([25, 1025], [0,1000], color=odor_lc, lw=odor_lw)



def icenter_odor_path(fig, xmin=-500, xmax=500, ymin=-100, ymax=1000):
    # Set data range
    fig.update_layout(yaxis=dict(range=[ymin, ymax], scaleratio=1))
    fig.update_layout(xaxis=dict(range=[xmin, xmax], scaleratio=1))

    return fig

def center_odor_path(ax, xmin=-500, xmax=500, ymin=-100, ymax=1000):
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))


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
