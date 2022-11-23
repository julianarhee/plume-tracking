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

import rdp
import _pickle as pkl
import scipy.stats as sts

# plotting
import matplotlib as mpl
import plotly.express as px
import pylab as pl
import seaborn as sns

# custom
import utils as util
import google_drive as gdrive

# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def get_log_files(src_dir=None, experiment=None, verbose=False, is_gdrive=False,
        return_loginfo=False,
        rootdir='/Users/julianarhee/Library/CloudStorage/GoogleDrive-edge.tracking.ru@gmail.com/My Drive/Edge_Tracking/Data'):
    '''
    Get a list of full paths of all .log files of interest. 
    Can pull from google sheet, or from directory mounted on local machine.
    MUST provide either src_dir (local) or experiment (is_gdrive=True).

    Keyword Arguments:
        src_dir (str) : full path to parent dir of .log files (default: {None})
        experiment (str) : subfolder(s) from rootdir that leads to .log (default: {None})
        verbose (bool) : print a bunch of stuff or no (default: {False})
        is_gdrive (bool) : pull log filenames from google sheet (default: {False})
        rootdir (str) : base dir of all exp folders (default: {'/Users/julianarhee/Library/CloudStorage/GoogleDrive-edge.tracking.ru@gmail.com/My Drive/Edge_Tracking/Data'})

    Returns:
        _description_
    '''
    if is_gdrive:
        # connect to google drive and get info from sheet
        assert os.path.split(rootdir)[-1]=='Analysis', 'For G-drive, rootdir should be /Edge_Tracking/Aanalysis. Current rootdir is: {}{{}'.format('\n', rootdir)
        logdf = gdrive.get_info_from_gsheet(experiment)
#        if 'degree' in experiment: # specific to 'degree' experiments
#            exp_key = experiment.split('-degree')[0]
#        else:
#            exp_key = experiment
#        #curr_logs = logdf[logdf['experiment']==exp_key].copy()
        if os.path.exists(os.path.join(rootdir, experiment, 'logs')):
            logdir = os.path.join(rootdir, experiment, 'logs')
        else:
            logdir = os.path.join(rootdir, experiment)
        log_files = [os.path.join(logdir, '{}'.format(f)) for f in logdf['log'].values \
                        if os.path.exists(os.path.join(logdir, '{}'.format(f)))]
        # check rootdir 
        if len(log_files)==0 and os.path.split(rootdir)[-1]=='Analysis':
            print("Checking Data dir instead of Analysis")
            rootdir_tmp = rootdir.replace('Analysis', 'Data')
            logdir = os.path.join(rootdir_tmp, experiment)
            log_files = [os.path.join(logdir, '{}'.format(f)) for f in logdf['log'].values \
                        if os.path.exists(os.path.join(logdir, '{}'.format(f)))] 
        print("{} of {} files found in: {}".format(len(log_files),
                                                    len(logdf['log'].unique()), logdir))
        if len(log_files)==0:
            print("Check experiment name, only the following have sheet IDs: ")
            gsheet_dict = gdrive.get_sheet_keys()
            for i, k in enumerate(gsheet_dict.keys()):
                print('    {}'.format(k))
            print("Update google_drive.get_sheet_keys() for new sheets.")
    else:
        try:
            log_files = sorted([k for k in glob.glob(os.path.join(src_dir, 'raw', '*.log'))], \
                            key=util.natsort)
            assert len(log_files)>0, "No log files found in src_dir raw: {}".format(src_dir)
        except AssertionError:
            print("Checking parent dir.")
            log_files = sorted([k for k in glob.glob(os.path.join(src_dir, '*.log'))], \
                            key=util.natsort)
        print("Found {} tracking files.".format(len(log_files)))

    if verbose:
        for fi, fn in enumerate(log_files):
            print(fi, os.path.split(fn)[-1])

    log_files = sorted(log_files, key=util.natsort)
    if return_loginfo:
        return log_files, logdf
    else:
        return log_files

def parse_info_from_filename(fpath, experiment=None, 
            rootdir='/Users/julianarhee/Library/CloudStorage/GoogleDrive-edge.tracking.ru@gmail.com/My Drive/Edge_Tracking/Data'):
    '''
    Attempts to find experiment info from filename.

    Arguments:
        fpath -- _description_

    Keyword Arguments:
        experiment -- _description_ (default: {None})
        rootdir -- _description_ (default: {'/Users/julianarhee/Library/CloudStorage/GoogleDrive-edge.tracking.ru@gmail.com/My Drive/Edge_Tracking/Data'})

    Returns:
        experiment, datestr, fly_id, condition
    '''

    #info_str = fpath.split('{}/'.format(rootdir))[-1]
    info_str = fpath.split('{}/'.format('/Edge_Tracking'))[-1]
    exp_cond_str, log_fname = os.path.split(info_str)
    fly_id=None
    cond=None

    # remove datestr
    date_str = re.search('[0-9]{8}-[0-9]{6}', log_fname)[0]
    cond_str = os.path.splitext(log_fname)[0].split('{}_'.format(date_str))[-1]
    if 'fly' in info_str.lower():
        # assumes: nameofexperiment/maybestuff/FlyID
        if experiment is None:
            experiment = exp_cond_str.lower().split('/{}'.format('fly'))[0] #\
                    #if "Fly" in exp_cond_str else exp_cond_str.split('/{}'.format('fly'))[0]
        # get fly_id
        fly_id = re.search('fly\d{1,3}[a-zA-Z]?', info_str, re.IGNORECASE)[0] # exp_cond_str
    else:
        if experiment is None:
            experiment = exp_cond_str # fly ID likely in LOG filename

    if fly_id is None:
        fly_id = 'fly{}'.format(date_str.split('-')[-1]) # use timestamp for unique fly id for current date

    condition = '_'.join([c for c in cond_str.split('_') if fly_id not in c \
                    and not re.search('\d{3}', c)])
    if condition is not None:
        condition = condition.lower()
    if fly_id is not None:
        fly_id = fly_id.lower()


    return experiment, date_str, fly_id, condition

def load_dataframe_test(fpath, verbose=False, 
                    parse_filename=True):
    '''
    Read raw .log file from behavior and return formatted dataframe.
    Assumes MFC for odor is either 'mfc2_stpt' or 'mfc3_stpt'.
    Assumes LED for on is 'led1_stpt'.

    Arguments:
        fpath -- (str) Full path to .log file
    '''
    # read .log as dataframe 

    df0 = pd.read_csv(fpath) #, encoding='latin' )#, sep=",", skiprows=[1], header=0, 
              #parse_dates=[1]).rename(columns=lambda x: x.strip())

    return df0


def load_dataframe(fpath, verbose=False, experiment=None, 
                    parse_filename=True, savedir=None, remove_invalid=True, plot_errors=False):
    '''
    Read raw .log file from behavior and return formatted dataframe.
    Assumes MFC for odor is either 'mfc2_stpt' or 'mfc3_stpt'.
    Assumes LED for on is 'led1_stpt'.

    Arguments:
        fpath -- (str) Full path to .log file
    '''
    # read .log as dataframe 
    df0 = pd.read_csv(fpath, encoding='latin' )#, sep=",", skiprows=[1], header=0, 
    # get file info
    fname = os.path.splitext(os.path.split(fpath)[-1])[0]
    df0['filename'] = fname
    df0['fpath'] = fpath
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
    df0['rel_time'] = df0['time'] - df0['time'].iloc[0]
    # convert datestr
    df0['date'] = df0['timestamp'].apply(lambda s: \
            int(datetime.strptime(s.split('-')[0], "%m/%d/%Y").strftime("%Y%m%d")))

    # convert ft_heading to make it continuous and in range (-pi, pi)
    if 'ft_heading' in df0.columns:
        p = util.unwrap_and_constrain_angles(df0['ft_heading'].values)
        df0['ft_heading'] = -p 

    # Calculate MDF odor on or off 
    mfc_vars = [c for c in df0.columns if 'mfc' in c]
    mfc_vars0 = [c for c in mfc_vars if len(df0[c].unique())>1] #== 2
    df0['odor_on'] = False
    mfc_vars0 = [c for c in mfc_vars if len(df0[c].unique())>1] #== 2
    if len(mfc_vars0)>0:
        mfc_vars = [c for c in mfc_vars0 if c!='mfc1_stpt']
        mfc_varname = mfc_vars[0]
        df0.loc[df0[mfc_varname]>0, 'odor_on'] = True
    # otherwise, only air (no odor)
    else:
        if verbose:
            print("... no odor changes detected in MFCs.")

    # check strip type (gradient or constant)
    is_gradient = len([c for c in mfc_vars0 if len(df0[c].unique())>2]) == 2    
    df0['strip_type'] = 'gradient' if is_gradient else 'constant'

    # check LEDs
    df0['led_on'] = False
    if 'led1_stpt' in df0.columns: #and 'led_on' not in df0.columns:
        datestr = int(df0['date'].unique())
        if int(datestr) <= 20200720:
            df0['led_on'] = df0['led1_stpt']==1 
        else:
            df0['led_on'] = df0['led1_stpt']==0

    # assign "instrip" -- can be odor or led (odor takes priority)
    df0['instrip'] = False
    if True in df0['odor_on'].unique():
        df0['instrip'] = df0['odor_on']
    elif True in df0['led_on'].unique():
        df0['instrip'] = df0['led_on']

    # check for wonky skips
    figpath = os.path.join(savedir, 'errors', 'wonkyft_{}.png'.format(fname)) if savedir is not None else None
    if savedir is not None:
        if not os.path.exists(os.path.join(savedir, 'errors')):
            os.makedirs(os.path.join(savedir, 'errors'))

    if figpath is None and plot_errors is True:
        print("[warning]: Provide savedir to save errors fig")
    df0, ft_flag = check_ft_skips(df0, plot=plot_errors, remove_invalid=remove_invalid,
                    figpath=figpath, verbose=verbose)
    if ft_flag:
        print("--> found bad skips in FTs, check: {}".format(fname))

    # get experiment info
    if parse_filename:
        #print("... parsing info from filename")
        exp, datestr, fly_id, cond = parse_info_from_filename(fpath, experiment)
        df0['experiment'] = experiment
        df0['fly_name'] = fly_id
        df0['condition'] = cond
        df0['trial'] = datestr
        if verbose:
            print("Exp: {}, fly ID: {}, cond={}".format(exp, fly_id, cond))
        # make fly_id combo of date, fly_id since fly_id is reused across days
        df0['fly_id'] = ['{}-{}'.format(dat, fid) for (dat, fid) in df0[['date', 'fly_name']].values]
        df0['trial_id'] = ['{}_{}'.format(fly_id, trial) for (fly_id, trial) in \
                  df0[['fly_id', 'trial']].values]
    else:
        df0['fly_id'] = fname
        df0['trial_id'] = fname
        df0['condition'] = 'none'

    return df0

def check_ft_skips(df, plot=False, remove_invalid=True, figpath=None, verbose=False):
    '''
    Check dataframe of current logfile and find large skips.

    Arguments:
        df (pd.DataFrame) : loaded/processed from load_dataframe()

    Keyword Arguments:
        plot (bool) : plot errors (default: {False})
        remove_invalid (bool) : only take first set of valid points (default: {True})

    Returns:
       df (pd.DataFrame) : either just itself or valid only
       valid_flag (bool) : True if bad skips detected
    '''
    fname = df['filename'].unique()
    bad_skips={}
    max_step_size={'ft_posx': 10, 'ft_posy': 10, 'ft_frame': 100}
    for pvar, stepsize in max_step_size.items():
        if pvar=='ft_frame':
            first_frame = df['ft_frame'].min() 
            wonky_skips = np.where(df[pvar]==first_frame+1)[0]
            if len(wonky_skips)>1:
                wonky_skips = wonky_skips[1:]
            else:
                wonky_skips = []
        wonky_skips = np.where(df[pvar].diff().abs()>=stepsize)[0]
        if len(wonky_skips)>0:
            first_step = df[pvar].diff().abs().max()
            #time_step = df.iloc[wonky_skips[0]]['time'] - df.iloc[wonky_skips[0]-1]['time']
            bad_skips.update({pvar: wonky_skips})
            if verbose:
                print("WARNING: found wonky ft skip ({} jumped {:.2f}).".format(pvar, first_step))
    if plot==True and len(bad_skips.keys())>0:
        fig, ax = pl.subplots(figsize=(3,3)) 
        fname = df['filename'].unique()[0]
        ax.set_title(fname)
        ax.plot(df['ft_frame'].diff().abs())
        cols = ['r', 'b', 'g']
        for pi, ((pvar, off_ixs), col) in enumerate(zip(bad_skips.items(), cols)):
            for i in off_ixs:
                ax.plot(df.iloc[i].name, pi*100, '*', c=col, label=pvar)
        ax.legend()
        #pl.show()
        if figpath is not None:
            pl.savefig(figpath)
        pl.close()

    flag = len(bad_skips)>0
    valid_df = df.copy()
    if flag:
        if 'ft_frame' in bad_skips.keys():
            wonky_skips = bad_skips['ft_frame']
        else:
            key = list(bad_skips.keys())[0]
            wonky_skips = bad_skips[key]
        if remove_invalid:
            valid_df = df.iloc[0:wonky_skips[0]].copy()
            sz_removed = df.shape[0] - valid_df.shape[0]
            print("[WARNING] {}: Found bad skips, removing {} of {} samples.".format(fname, sz_removed, df.shape[0]))
        else:
            valid_df = df.copy()

    return valid_df, flag

def load_dataframe_resampled_csv(fpath):
    '''
    Temp loading func for processed .csv files. 

    Arguments:
        fpath (str) : full path to .csv file. 

    Returns:
        df (pd.DataFrame) : DF with dtypes formatted to expected. Currently, does not add the other additional vars. See load_dataframe().
    '''
    df_full = pd.read_table(fpath, sep=",", skiprows=[1], header=0, 
                parse_dates=[1]).rename(columns=lambda x: x.strip())
    df0 = df_full[['x', 'y', 'seconds', 'instrip']].copy()
    df0.loc[df0['instrip']=='False', 'instrip'] = 0
    df0 = df0.rename(columns={'x': 'ft_posx', 'y': 'ft_posy', 'seconds': 'time'}).astype('float')
    df0['instrip'] = df0['instrip'].astype(bool)

    other_cols = [c for c in df_full.columns if c not in df0.columns]
    for c in other_cols:
        df0[c] = df_full[c]

    file_id = [f for f in fpath.split('/') if re.findall('[0-9]{8}', f)][0]
    condition = os.path.split(os.path.split(fpath)[0])[-1] # assumes `et` and `replay` are parents dirs of .csv for hdeltac
    df0['condition'] = condition
    df0['fly_id'] = file_id
    df0['trial_id'] = ['_'.join([fly_id, cond]) for (fly_id, cond) \
                   in df0[['fly_id', 'condition']].values]

    return df0

def save_df(df, fpath):
    with open(fpath, 'wb') as f:
        pkl.dump(df, f)

def load_df(fpath):
    with open(fpath, 'rb') as f:
        df = pkl.load(f)
    return df

def correct_manual_conditions(df, experiment, logdf=None):
    '''
    Tries to correct manually-renamed files so that "condition" is accurate.

    Arguments:
        df -- _description_
        experiment -- _description_
        logdf (pd.DataFrame) : loaded from google sheets, has correct experiment names
    Returns:
        _description_
    '''
    print("Correcting experiment conditions: {}".format(experiment))
    if logdf is not None:
        for logfn, df_ in df.groupby('filename'):
            # specific to old data log format from google sheets...
            experiment = logdf[logdf['log']=='{}.log'.format(logfn)]['experiment'].values[0]
            genotype = logdf[logdf['log']=='{}.log'.format(logfn)]['genotype'].values[0]
            df.loc[df['filename']==logfn, 'condition'] = experiment
            df.loc[df['filename']==logfn, 'genotype'] = genotype
    else:
        if experiment=='vertical_strip/paired_experiments':
            # update condition names
            df.loc[df['condition']=='light', 'condition'] = 'lightonly'

        elif experiment=='reverse gradient':
            df.loc[df['condition']=='cantons_constantodor', 'condition'] = 'constantodor'
            df.loc[df['condition']=='cantons_reversegradient', 'condition'] = 'reversegradient'
            df.loc[df['condition']=='cantons_contantodor', 'condition'] = 'constantodor'
        
        elif 'degree' in experiment:
            df['condition'] = experiment 
            #df.loc[df['condition']!='cantons_constantodor', 'condition'] = 'cantons_constantodor'  
        df['genotype'] = ''
    return df

def load_combined_df(src_dir=None, log_files=None, logdf=None, is_csv=False, 
                experiment=None, savedir=None, 
                create_new=False, verbose=False, save_errors=True, remove_invalid=True, 
                process=True, save=True, parse_filename=True, 
                rootdir='/Users/julianarhee/Library/CloudStorage/GoogleDrive-edge.tracking.ru@gmail.com/My Drive/Edge_Tracking/Data'):
    '''
    _summary_

    Keyword Arguments:
        src_dir (str, None) : parent dir of .log files (default: {None})
        log_files (list) : list of full paths to .log files (default: {None})
        savedir (str, None) : where to save combined DF (default: {parent dir of 1st log file})
        create_new (bool) : load everything anew (default: {False})
        verbose (bool): print a bunch of stuff (default: {False})
        save_errors (bool) : save plots for bad skips, etc. (default: {True})
        remove_invalid (bool) : only include data that is valid, no big skips (default: {True})
        process (bool) : do some additional calculations (default: {True})
        save (bool) : save combined df (default: {True})
        rootdir (str) : path to base dir of all experiments (default: {'/Users/julianarhee/Library/CloudStorage/GoogleDrive-edge.tracking.ru@gmail.com/My Drive/Edge_Tracking/Data'})

    Returns:
        df (pd.DatFrame) : all (processed) dfs across found log files. 
    '''
    if src_dir is None:
        assert log_files is not None, "Must provide src_dir or log_files"
        src_dir = os.path.split(log_files[0])[0]
        src_dir = src_dir.split('/log')[0]
    elif log_files is None:
        assert src_dir is not None, "Must provide src_dir or log_files"

    # get savedir for saving final .pkl file
    if savedir is None:
        if src_dir is None:
            src_dir = os.path.split(log_files[0])[0]
        savedir = src_dir.split('/raw')[0]
    # first, check if combined df exists
    df_fpath = os.path.join(savedir, 'combined_df.pkl')
    if create_new is False:
        if os.path.exists(df_fpath):
            print("loading existing combined df")
            try:
                df = load_df(df_fpath)
                return df
            except Exception as e:
                create_new=True
        else:
            create_new=True

    if save_errors:
        if not os.path.exists(os.path.join(savedir, 'errors')):
            os.makedirs(os.path.join(savedir, 'errors'))

    if log_files is None:
        print("Creating new combined df from raw files...")
        log_fmt = 'csv' if is_csv else 'log'
        log_files = sorted([k for k in glob.glob(os.path.join(src_dir, '*.{}'.format(log_fmt)))\
                if 'lossed tracking' not in k], key=util.natsort)
    print("Processing {} tracking files.".format(len(log_files)))

    if create_new:
        # cycle thru log files and combine into 1 mega df
        dlist = []
        for fi, fn in enumerate(log_files):
            fname = os.path.split(fn)[-1]
            if verbose is True:
                try:
                    exp, datestr, fly_id, cond = parse_info_from_filename(fn) 
                    print(fi, datestr, fly_id, cond)
                except Exception as e:
                    print(fname)
                    parse_filename=False
            if is_csv:
                df_  = load_dataframe_resampled_csv(fn)
            else:
                df_ = load_dataframe(fn, verbose=False, experiment=experiment, 
                                parse_filename=parse_filename,
                                savedir=savedir, remove_invalid=remove_invalid, plot_errors=save_errors)
            dlist.append(df_)
        df = pd.concat(dlist, axis=0)
        # get experiment name
        experiment = src_dir.split('{}/'.format(rootdir))[-1]
        df = correct_manual_conditions(df, experiment, logdf=logdf)
        # do some processing, like distance and speed calculations
        if process:
            if verbose:
                print("---> processing")
            df = process_df(df, verbose=verbose)
       # save
        if save:
            print("Saving combined df to: {}".format(savedir))
            save_df(df, df_fpath)

    return df

# strip-related calculations
# --------------------------------------------------------------------
def get_odor_params(df, strip_width=50, strip_sep=200, get_all_borders=True,
                    entry_ix=None, is_grid=False, check_odor=False, 
                    mfc_var='mfc2_stpt'):
    '''
    Get odor start times, boundary coords, etc.
    Note:  df should not start directly ON first entry index if calculate entry dir

    Arguments:
        df (pd.DataFrame) : full dataframe (from load_dataframe())
        
    Keyword Arguments:
        strip_width (float) :  Width of odor corridor in mm (default: 50)
        strip_sep (float) : Separation between odor strips (default: 200)
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
        entry_left_edge=None
        odor_xmin = -strip_width/2.
        odor_xmax = strip_width/2.
        odor_start_time = df.iloc[0]['time']
        odor_start_posx, odor_start_posy = (0, 0)
        #currdf = df.copy()
    else:
        odor_borders, entry_left_edge = find_strip_borders(df, 
                                strip_width=strip_width, strip_sep=strip_sep, 
                                get_all_borders=get_all_borders, entry_ix=None, is_grid=is_grid,
                                return_entry_sides=True)

        odor_start_time = df[df['instrip']].iloc[0]['time']
        odor_start_posx = df[df['instrip']].iloc[0]['ft_posx']
        odor_start_posy = df[df['instrip']].iloc[0]['ft_posy']

    trial_start_time = df.iloc[0]['time']

    if check_odor:
        assert odor_start_time == df.iloc[df[mfc_var].argmax()]['time'],\
            "ERR: odor start time does not match MFC switch time!"

    odor_params = {
                    'trial_start_time': trial_start_time,
                    'odor_start_time': odor_start_time,
                    'odor_boundary': odor_borders, #(odor_xmin, odor_xmax),
                    'odor_start_pos': (odor_start_posx, odor_start_posy),
                    'entry_left_edge': entry_left_edge
                    } 

    return odor_params



def check_entryside_and_flip(df_, strip_width=50, odor_dict=None, verbose=False):
    '''
    Check if animal enters on corridor's left or right edge. Flip so that animal 
    enters on corridor's RIGHT edge (animal's left side). 

    Arguments:
        df_ -- _description_

    Keyword Arguments:
        strip_width -- _description_ (default: {50})
        odor_dict -- _description_ (default: {{}})
        verbose -- _description_ (default: {False})

    Returns:
        df_fp (pd.DataFrame) : dataframe with coords flipped to have tracking on odor RIGHT edge (fly's left)
        new_borders (dict) : odor borders for flipped
    '''
    new_borders={}
    in_odor = odor_dict is not None
    if odor_dict is None:
        odor_dict, in_odor = get_odor_grid(df_, strip_width=strip_width)
    if not in_odor:
        return 
    entry_ixs = [int(k[1:]) for k, v in odor_dict.items()]
    df_copy = df_.copy()
    df_copy['flipped'] = False
    for si, entry_ix in enumerate(entry_ixs):
        #df.loc[start_ix]
        if entry_ix == 0:
            last_bout_ix = 0
        else:
            last_outbout_ix = df_.loc[entry_ix-1].name
        start_ = df_.iloc[0].name if si==0 else last_outbout_ix
        stop_ = df_.iloc[-1].name if entry_ix == entry_ixs[-1] else entry_ixs[si+1]-1
        tmpdf = df_.loc[start_:stop_]
        oparams = get_odor_params(tmpdf.loc[start_:stop_], strip_width=strip_width, 
                                        is_grid=True, get_all_borders=False, entry_ix=entry_ix)
        if verbose:
            print('... {}: {}'.format(si, oparams['entry_left_edge']))
        if oparams['entry_left_edge']:
            # flip it
            xp, yp = util.fliplr_coordinates(tmpdf['ft_posx'].values, tmpdf['ft_posy'].values)
                # util.rotate_coordinates(df1['ft_posx'], df1['ft_posy'], -np.pi)
            df_copy.loc[tmpdf.index, 'ft_posx'] = xp
            df_copy.loc[tmpdf.index, 'ft_posy'] = yp
            border_flip1, _ = util.fliplr_coordinates(oparams['odor_boundary'][0][0], 0) 
            border_flip2, _ = util.fliplr_coordinates(oparams['odor_boundary'][0][1], 0)
            df_copy.loc[tmpdf.index, 'flipped'] = True
        else:
            border_flip1, border_flip2 = oparams['odor_boundary'][0]
        
        new_borders.update({'c{}'.format(entry_ix): (border_flip1, border_flip2)})

    # flip headigs
    for heading_var in ['ft_heading', 'heading']:
        if heading_var in df_copy.columns:
            df_copy['{}_og'.format(heading_var)] = df_[heading_var].values
            df_copy[heading_var] = df_[heading_var].values
            tmpdf = df_copy[df_copy['flipped']]['{}_og'.format(heading_var)]

            #df_copy['{}_og'.format(heading_var)] = df_copy[heading_var].values
            #vals = -1*df_copy[df_copy['flipped']]['{}_og'.format(heading_var)].values
            #df_copy.loc[df_copy['flipped'], heading_var] = vals
            vals = -1*tmpdf.values
            df_copy.loc[tmpdf.index, heading_var] = vals

    return df_copy, new_borders

def check_entry_left_edge(df, entry_ix=None, nprev_steps=5, 
            return_bool=False, only_upwind_outbouts=False):
    '''
    Check whether fly enters from left/right of corridor based on prev tsteps.

    Arguments:
        df (pd.DataFrame) : dataframe with true indices
        entry_ix (int) : index of entry point

    Returns:
        entry_left (bool) : entered left True, otherwise False
        entry_df (pd.DataFrame) : if return_bool, dataframe (entry_index, prev_outbout_num, enterd_left_bool) for each instrip bout entry
    '''
    # if start of experiment, 1st odor start centered on animal
    # so look at 2nd odor bout
    if entry_ix is None:
        entry_ix = df[df['instrip']].iloc[0].name 
    # check first entry *after* odor start that is also UPWIND
    from_odor_onset = df.loc[entry_ix:].copy()
    outdfs_after_entry = from_odor_onset[~from_odor_onset['instrip']]
    #outbouts_after_entry = df[~df['instrip']].loc[entry_ix:] 
    # only if last N (=20) time steps are roughly upwind (i.e., not downwind) into the strip, 
    # otherwise left/right difficult to distinguish
    outbouts_after_entry = outdfs_after_entry['boutnum'].unique()
    upwind_before_entry = [b for b, b_ in outdfs_after_entry.groupby('boutnum') \
                            if b_.iloc[-nprev_steps:]['ft_posy'].diff().sum()>=0]
    bouts_to_check = upwind_before_entry if only_upwind_outbouts else outbouts_after_entry
    # for each outbout that is upwind just prior to an entry, determine if entering 
    # on LEFT or RIGHT edge of strip
    entry_lefts=[]
    for start_at_this_outbout in bouts_to_check: #upwind_before_entry:
        # get first INSTRIP frame that is after the 1st odor bout and also upwind
        try:
            # get the first outbout index and the following inbout index
            exit_ix = df[df['boutnum']>=start_at_this_outbout].iloc[0].name
            df_tmp = df.loc[exit_ix:]
            test_entry_ix = df_tmp[df_tmp['instrip']].iloc[0].name
        except IndexError as e:
            # this is a second (or later) entry
            continue
        # If values are increasing, then fly is entering on strip's left edge,
        # i.e., animal enters on its right side
        #s_ix = test_entry_ix - nprev_steps
        #e_ix = test_entry_ix + nprev_steps*2.
        min_steps = min([df_tmp.groupby(['boutnum']).count().min().min(), nprev_steps])
        max_steps = min([df_tmp.groupby(['boutnum']).count().min().min(), nprev_steps])
        # print(min_steps, max_steps)
        s_ix = test_entry_ix - min_steps
        e_ix = test_entry_ix + max_steps
        cumsum = df_tmp.loc[s_ix:e_ix]['ft_posx'].diff().sum()
        if cumsum > 0: # entry is from LEFT of strip (values get larger)
            entry_left_edge=True
        elif cumsum < 0: # entry is into RIGHT side of strip (values get smaller)
            entry_left_edge=False
        else:
            entry_left_edge=None
        entry_lefts.append((test_entry_ix, start_at_this_outbout, entry_left_edge))

    entry_df = pd.DataFrame(data=entry_lefts, \
                    columns=['entry_index', 'previous_outbout', 'entry_left_edge'])

    # if majority of entries (>50%) are on the left, left-enterer, otherwise not.
    entry_vals = entry_df['entry_left_edge'].values #[i[-1] for i in entry_lefts]
    try:
        if sum(entry_vals)/len(entry_vals) > 0.5:
            entry_left_edge=True
        elif sum(entry_vals)/len(entry_vals) < 0.5:
            entry_left_edge=False
        else:
            entry_left_edge=None
    except ZeroDivisionError:
        entry_left_edge=None

    if return_bool:
        # compare each val to previous val then apply cumsum to get counter value for each group
        # of consecutive vals
        entry_df['consecutive_bout'] = entry_df['entry_left_edge']\
                                        .ne(entry_df['entry_left_edge'].shift()).cumsum()
        # count N consecutive values for each bout or group of consecutive values
        entry_df['consecutive_count'] = (entry_df.groupby(entry_df['entry_left_edge']\
                                        .ne(entry_df['entry_left_edge'].shift()).cumsum())
                                        .cumcount())
        return entry_left_edge, entry_df
    else: 
        return entry_left_edge
 
def find_strip_borders(df, entry_ix=None, strip_width=50, return_entry_sides=False,
                        strip_sep=200, is_grid=True, get_all_borders=True):
    '''
    Get all strip borders using get_odor_grid() OR taking first values of instrip.

    Arguments:
        df (pd.DataFrame): DF for 1 log file. 

    Keyword Arguments:
        entry_ix (int): index of first odor onset (default: {None})
        strip_width (float, int): width of strip, mm (default: {50})
        strip_sep (float, int): separation between strips, mm (default: {20})
        is_grid (bool): True if multiple strips (default: {True})
        get_all_borders (bool): Get all found borders (default: {True})

    Returns:
        odor_borders (list) : List of tuples, each indicating (strip_min, strip_max)        
    '''
    if entry_ix is None:
        try:
            entry_ix = df[df['instrip']].iloc[0].name
        except IndexError:
            entry_ix=None
            odor_xmin = -(strip_width/2.)
            odor_xmax = strip_width/2.
            return [(odor_xmin, odor_xmax)]
     
    entry_left_edge, entry_lefts = check_entry_left_edge(df, entry_ix=entry_ix, return_bool=True)
    currdf = df.loc[entry_ix:].copy()
    if entry_left_edge is not None: # entry_left must be true or false
        if get_all_borders:
            ogrid, in_odor = get_odor_grid(currdf, 
                                strip_width=strip_width, strip_sep=strip_sep,
                                use_crossings=True, verbose=False)
            odor_borders = list(ogrid.values())
        else:
            if entry_left_edge:
                odor_xmin = currdf[currdf['instrip']].iloc[0]['ft_posx'] 
                odor_xmax = currdf[currdf['instrip']].iloc[0]['ft_posx'] + strip_width
            else: # entered right, so entry point is largest val
                odor_xmin = currdf[currdf['instrip']].iloc[0]['ft_posx'] - strip_width
                odor_xmax = currdf[currdf['instrip']].iloc[0]['ft_posx'] 
            odor_borders = [(odor_xmin, odor_xmax)]
    else:
        odor_xmin = currdf[currdf['instrip']].iloc[0]['ft_posx'] - (strip_width/2.)
        odor_xmax = currdf[currdf['instrip']].iloc[0]['ft_posx'] + (strip_width/2.)
        odor_borders = [(odor_xmin, odor_xmax)]

    if return_entry_sides:
        return odor_borders, entry_left_edge
    else:
        return odor_borders
 

def find_crossovers(df_, strip_width=50):
    crossover_bouts = [bnum for bnum, b_ in df_[df_['instrip']].groupby('boutnum') \
                         if np.ceil(b_['ft_posx'].max() - b_['ft_posx'].min()) >= strip_width]
    return crossover_bouts



# ---------------------------------------------------------------------- 
# Data processing
# ----------------------------------------------------------------------
def process_df(df, xvar='ft_posx', yvar='ft_posy', 
                bout_thresh=0.5, 
                smooth=False, window_size=11, verbose=False):
    dlist=[]
    for trial_id, df_ in df.groupby('trial_id'):
        if verbose:
            print("... processing {}".format(trial_id))
        # parse in and out bouts
        df_ = parse_bouts(df_, count_varname='instrip', bout_varname='boutnum') # 1-count
        # filter in and out bouts by min. duration 
        df_ = filter_bouts_by_dur(df_, bout_thresh=bout_thresh, 
                            bout_varname='boutnum', count_varname='instrip', verbose=False)
        # add some calculations
        df_ = calculate_speed(df_, xvar=xvar, yvar=yvar)
        df_ = calculate_distance(df_, xvar=xvar, yvar=yvar)
        # smooth?
        if smooth:
            df_ = smooth_traces(df_, window_size=window_size, return_same=True)
        dlist.append(df_)
    DF=pd.concat(dlist, axis=0) # dont reset index
    DF['bout_type'] = 'outstrip'
    DF.loc[DF['instrip'], 'bout_type'] = 'instrip'

    return DF

def calculate_turn_angle(df, xvar='ft_posx', yvar='ft_posy'):
    '''
    Calculate angle bw positions (arctan2(gradx, grady)). 

    Arguments:
        df -- _description_

    Keyword Arguments:
        xvar -- _description_ (default: {'ft_posx'})
        yvar -- _description_ (default: {'ft_posy'})

    Returns:
       df with 'turn_angle' added. 
    '''
    #ang_ = np.arctan2(np.gradient(df[yvar].values), np.gradient(df[xvar].values))
    df['turn_angle'] = np.arctan2(np.gradient(df[xvar]), np.gradient(df[yvar]))
    #ang_ = np.arctan2(df[yvar].diff(), df[xvar].diff())
    #df['turn_angle'] = ang_
    
    return df

def calculate_speed(df0, xvar='ft_posx', yvar='ft_posy'):
    '''
    Calculate speed as gradient/time. Blind to bouts.
    '''
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
        boutdur = df_.sort_values(by='time').iloc[-1]['time'] - df_.iloc[0]['time']
        boutdurs.update({boutnum: boutdur})

    return boutdurs

def filter_bouts_by_dur(df, bout_thresh=0.5, bout_varname='boutnum', 
                        count_varname='instrip', speed_varname='smoothed_speed', 
                        verbose=False, switch_method='previous'):
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
                if switch_method=='reverse':
                    # opposite of whatever it is
                    df.loc[df_.index, count_varname] = ~df_[count_varname]
                elif switch_method=='previous':
                    # prev bouts value
                    prev_value = df[df[bout_varname]==(boutnum-1)].iloc[-1][count_varname]
                    df.loc[df_.index, count_varname] = prev_value
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
    '''
    Smooths x- and y-vars, which seems to be more accurate than interpolating over 2d.

    Arguments:
        df -- _description_

    Keyword Arguments:
        xvar -- _description_ (default: {'ft_posx'})
        yvar -- _description_ (default: {'ft_posy'})
        window_size -- _description_ (default: {13})
        return_same -- _description_ (default: {True})

    Returns:
        _description_
    '''
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
def get_odor_grid_all_flies(df0, strip_width=50, strip_sep=200):
    '''
    Wraps get_odor_grid() in a loop for all ddatafiles.

    Arguments:
        df0 -- _description_

    Keyword Arguments:
        strip_width -- _description_ (default: {50})
        strip_sep -- _description_ (default: {200})

    Returns:
        odor_borders (dict) : 
            keys are trial_id (e.g., datestr-flyid)
            values are dicts, with each entry correponding to 1 odor corridor
    '''
    odor_borders={}
    for trial_id, currdf in df0.groupby(['trial_id']):
        ogrid, in_odor = get_odor_grid(currdf, strip_width=strip_width, strip_sep=strip_sep,
                                    use_crossings=True, verbose=False)
        if not in_odor:
            print(trial_id, "WARNING: Fly never in odor (cond={})".format(currdf['condition'].unique()))
        try:
            odor_borders.update({trial_id: ogrid})
        except Exception as e:
            print(e)
            print(ogrid)

    return odor_borders

def get_odor_grid(df, strip_width=10, strip_sep=200, use_crossings=True,
                    use_mean=True, verbose=False):
    '''
    Get all odor corridors in current fly's trajectory. 

    Arguments:
        df -- _description_

    Keyword Arguments:
        strip_width -- odor strip width (default: {10})
        strip_sep -- separation (mm) between corridors (default: {200})
        use_crossings -- use actual crossings (default: {True})
        use_mean -- use average of crossings, rather than max/min (default: {True})
        verbose -- _description_ (default: {True})

    Returns:
        odor_dict (dict) : keys are 'cIX' where IX is entry index, and values are tuple (min, max)
        odor_flag (bool) : boolean whether animal ever in order or no
    '''
    if df[df['instrip']].shape[0]==0:
        # fly never in odor
        if verbose:
            print("WARNING: Fly is never in odor, using default corridor.")
        curr_odor_xmin = -strip_width/2.
        curr_odor_xmax = strip_width/2.
        odor_grid = {'c0': (curr_odor_xmin, curr_odor_xmax)}
        odor_flag = False
    else:
        odor_grid = find_odor_grid(df, 
                        strip_width=strip_width, strip_sep=strip_sep)
        odor_grid = check_odor_grid(df, odor_grid, strip_width=strip_width, strip_sep=strip_sep, 
                        use_crossings=use_crossings, use_mean=use_mean, verbose=verbose)
        odor_flag = True

    return odor_grid, odor_flag


def find_borders(df, strip_width = 10, strip_spacing = 200):
    '''from andy'''
    from scipy import signal as sg
    x = df.ft_posx
    y = df.ft_posy
    x_idx = df.index[df.mfc2_stpt>0.01].tolist()[0]
    x0 = df.ft_posx[x_idx] # First point where the odor turn on
    duty = strip_width/(strip_width+strip_spacing)
    freq = 1/(strip_width+strip_spacing)
    x_temp = np.linspace(min(x)-strip_width, max(x)+strip_width, 1000)
    mult = 0.5*sg.square(2*np.pi*freq*(x_temp+strip_width/2-x0), duty=duty)+0.5
    x_borders,_ = sg.find_peaks(np.abs(np.gradient(mult)))
    x_borders = x_temp[x_borders]
    #y_borders = np.array([y.iloc[x_idx], max(y)])
    #t_borders = np.array([min(t), max(t)])
    #all_x_borders = [np.array([x_borders[i], x_borders[i]]) for i, _ in enumerate(x_borders)]
    #all_y_borders = [y_borders for i,_ in enumerate(x_borders)]
    #all_t_borders = [t_borders for i,_ in enumerate(x_borders)]
    return x_borders #all_x_borders, all_y_borders

 
def find_odor_grid(df, strip_width=10, strip_sep=200, plot=True): 
    '''
    Finds the odor boundaries based on odor width and grid separation

    Arguments:
        df -- _description_

    Keyword Arguments:
        strip_width -- _description_ (default: {10})
        strip_sep -- _description_ (default: {200})

    Returns:
        _description_
    '''
    # get first odor entry
    if not len(df['instrip'].unique())==2:
        print("WARNING: Fly only in odor or never. {}".format(df['trial_id'].unique()))
        if plot:
            fig = plot_trajectory(df)
    #assert len(df['instrip'].unique())==2, "Fly not in odor. {}".format(df['trial_id'].unique())
    curr_odor_xmin = df[df['instrip']].iloc[0]['ft_posx'] - (strip_width/2.)
    curr_odor_xmax = df[df['instrip']].iloc[0]['ft_posx'] + (strip_width/2.)

    # identify other grid crossings
    indf = df[df['instrip']].copy()
    # initiate grid dict
    odor_grid = {'c{}'.format(indf.iloc[0].name): (curr_odor_xmin, curr_odor_xmax)}

    indf = df[df['instrip']].copy()
    # where is the fly outside of current odor boundary but still instrip:
    # nextgrid_df = indf[ (indf['ft_posx']>(curr_odor_xmax.round(2)+strip_sep*0.5) | ((indf['ft_posx']<curr_odor_xmin.round(2)-strip_sep*0.5))].copy()
    nextgrid_df = indf[ (indf['ft_posx']>np.ceil(curr_odor_xmax)+strip_sep*0.5) \
                   | ((indf['ft_posx']<np.floor(curr_odor_xmin)-strip_sep*0.5)) ].copy()
    # loop through the remainder of odor strips in experiment until all strips found
    while nextgrid_df.shape[0] > 0:
        # get odor params of next corridor
        last_ix = nextgrid_df[nextgrid_df['instrip']].iloc[0].name
        next_odorp = get_odor_params(df.loc[last_ix:], strip_width=strip_width, 
                            entry_ix=last_ix, is_grid=True, get_all_borders=False)
        # update odor param dict
        odor_grid.update({'c{}'.format(last_ix): next_odorp['odor_boundary'][0]})
        (curr_odor_xmin, curr_odor_xmax) = next_odorp['odor_boundary'][0]
        # look for another odor corridor (outside of current odor boundary, but instrip)
        nextgrid_df = indf[ (indf['ft_posx'] >= (curr_odor_xmax+strip_sep)) \
                        | ((indf['ft_posx'] <= (curr_odor_xmin-strip_sep))) ]\
                        .loc[last_ix:].copy()

    return odor_grid


def check_odor_grid(df, odor_grid, strip_width=10, strip_sep=200, use_crossings=True,
                    use_mean=True, verbose=True):
    '''
    Use actual edge crossings to get odor boundary.
    Standard way is to do +/- half odor width at position of odor onset.

    Arguments:
        df (pd.DataFrame) : parsed .log file, must have instrip, ft_posx, ft_posy

    Keyword Arguments:
        strip_width -- odor corridor width in mm (default: {10, specific to imaging; typically, 50mm})
        strip_sep -- separation in mm of odor corridors (default: {200})
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
                                                ix=curr_grid_ix, strip_width=strip_width, strip_sep=strip_sep, 
                                                use_mean=use_mean)
                if verbose:
                    print('... {}: min {:.2f} vs traveled {:.2f}'.format(cnum, curr_odor_xmin, traveled_xmin))
                    print('... {}: max {:.2f} vs traveled {:.2f}'.format(cnum, curr_odor_xmax, traveled_xmax))
                    print("... True diff: {:.2f}".format(traveled_xmax - traveled_xmin))
            else:
                traveled_xmin = curr_odor_xmin
                traveled_xmax = curr_odor_xmax
                if verbose:
                    print("... this fly never crossed the edge")
            ctr = curr_odor_xmin + (curr_odor_xmax - curr_odor_xmin)/2.
            if verbose:
                print("... Estimated ctr is {:.2f}".format(ctr))
            if not crossed_edge:
                # at start of odor, animal doesnt move
                print("... Using default odor min/max, animal {} did not move in odor".format(df['fly_id'].unique()[0]))
                traveled_xmin = curr_odor_xmin
                traveled_xmax = curr_odor_xmax
            else:
                if traveled_xmax < ctr+strip_width*0.5: # animal never crosses right side
                    traveled_xmax = curr_odor_xmax
                    if verbose:
                        print("... setting travel xmax: {:.2f}".format(curr_odor_xmax))
                if traveled_xmin > ctr-strip_width*0.5:
                    traveled_xmin = curr_odor_xmin # animal never crosses left side (?)
                    if verbose:
                        print("... setting travel xmin: {:.2f}".format(curr_odor_xmin))
            # check width
            if abs(strip_width - (traveled_xmax - traveled_xmin)) < strip_width*0.25:
                if verbose:
                    print("... {} updating: ({:.2f}, {:.2f})".format(cnum, traveled_xmin, traveled_xmax))
                odor_grid.update({cnum: (traveled_xmin, traveled_xmax)})
            else:
                bad_corridors.append(cnum)
                if verbose:
                    print("... Difference was: {:.2f}".format(abs(strip_width - (traveled_xmax - traveled_xmin))))
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


def get_boundary_from_crossings(df, curr_odor_xmin, curr_odor_xmax, 
                     ix=0,
                    strip_sep=200, strip_width=10, use_mean=True,
                    verbose=False):
    # get left and right edge crossings
    right_xings = df[(df['ft_posx'] <= (curr_odor_xmax+strip_width*.5)) \
                & (df['ft_posx'] >= (curr_odor_xmax-strip_width*.5))].loc[ix:].copy()
    left_xings = df[(df['ft_posx'] <= (curr_odor_xmin+strip_width*.5)) \
                & (df['ft_posx'] >= (curr_odor_xmin-strip_width*.5))].loc[ix:].copy()

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
        traveled_xmin = traveled_xmax - strip_width
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
        traveled_xmax = traveled_xmin + strip_width
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

def plot_overlay_rdp_v_smoothed(b_, ax, xvar='ft_posx', yvar='ft_posy', epsilon=1,
            normalize_y=False):

    if 'rdp_{}'.format(xvar) not in b_.columns:
        add_rdp_by_bout(b_, epsilon=epsilon, xvar=xvar, yvar=yvar)

    rdp_var = 'rdp_{}'.format(xvar)
    rdp_y = 'rdp_{}'.format(yvar)
    offset_raw = b_['ft_posy'].iloc[0] if normalize_y else 0
    offset_rdp = b_[b_[rdp_var]]['ft_posy'].iloc[0] if normalize_y else 0
    if 'smoothed_ft_posx' in b_.columns:
        offset_smooth = b_['smoothed_ft_posy'].iloc[0] if normalize_y else 0

    yv = b_['ft_posy'] - offset_raw 
    yv_rdp = b_[b_[rdp_var]]['ft_posy'] - offset_rdp 
    if 'smoothed_ft_posy' in b_.columns:
        yv_smooth = b_['smoothed_ft_posy'] - offset_smooth 

    ax.plot(b_['ft_posx'], yv, 'w', alpha=1, lw=0.5)
    ax.plot(b_[b_[rdp_var]][xvar], yv_rdp, 'r', alpha=1, lw=0.5)
    if 'smoothed_ft_posx' in b_.columns:
        ax.plot(b_['smoothed_ft_posx'], yv_smooth, 'cornflowerblue', alpha=0.7)
#    ax.scatter(b_[b_[rdp_var]][xvar], b_[b_[rdp_y]][yvar], 
#               c=b_[b_[rdp_var]]['speed'], alpha=1, s=3)


def plot_overlay_rdp_v_smoothed_multi(df_, boutlist=None, nr=4, nc=6, distvar=None,
                                sharex=False, sharey=False,
                                rdp_epsilon=1.0, smooth_window=11, xvar='ft_posx', yvar='ft_posy'):
    if boutlist is None:
        #boutlist = list(np.arange(1, nr*nc))
        nbouts_plot = nr*nc
        boutlist = df_['boutnum'].unique()[0:nbouts_plot]
    fig, axes = pl.subplots(nr, nc, figsize=(nc*2, nr*2.5), sharex=sharex, sharey=sharey)
    for bi, bnum in enumerate(boutlist):
        ax = axes.flat[bi]
        b_ = df_[(df_['boutnum']==bnum)].copy() 
        plot_overlay_rdp_v_smoothed(b_, ax, xvar=xvar, yvar=yvar, normalize_y=sharey)
        if distvar is not None:
            dist_traveled = b_[distvar].sum()-b_[distvar].iloc[0]
            ax.set_title('{}: {:.2f}'.format(bnum, dist_traveled))
        else:
            ax.set_title(bnum)
        for ax in axes.flat:
            #ax.set_aspect('equal')
            ax.axis('off')
    legh = [mpl.lines.Line2D([0], [0], color='w', lw=2, label='orig'),
           mpl.lines.Line2D([0], [0], color='r', lw=2, label='RDP ({})'.format(rdp_epsilon))]
    if 'smoothed_ft_posx' in b_.columns: 
        legh.append(mpl.lines.Line2D([0], [0], color='b', lw=2, label='smoothed ({})'.format(smooth_window)))
   
    axes.flat[nc-1].legend(handles=legh, bbox_to_anchor=(1,1), loc='upper left')
    pl.subplots_adjust(bottom=0.2)
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

def convert_ccw(v):
    vv = v.copy()
    vv[v>0] -= 2*np.pi
#     if angle > 0:
#         angle -= 2 * math.pi
    #vv = (180 / np.pi) * v # rad to deg conversion
    return vv #v #(180 / math.pi) * angle  # rad to deg conversion

def convert_ft(v):
    vv = v.copy()
    vv[v<0] += 2*np.pi
    vv[v>2*np.pi] -= 2*np.pi
    return vv #v #(180 / math.pi) * angle  # rad to deg conversion

def convert_heading_csv(v):
    vv = v.copy()
    #vv[v<0] += 2*np.pi
    vv[v>2*np.pi] -= 2*np.pi
    return vv #v #(180 / math.pi) * angle  # rad to deg conversion

def make_continuous(mapvals):
    map_c = mapvals.copy()
    map_c = -1*map_c
    #map_c = map_c % (2*np.pi)
    return map_c

def rdp_to_heading(b_, xvar='ft_posx', yvar='ft_posy', theta_range=(-np.pi, np.pi)):
    '''
    Calculate vector between each point in RDP-simplified coordinates.
    Wraps radians to be within (-pi and pi). Values should correspond to 
    similar operation done to ft_heading values.

    Arguments:
        b_ (pd.DataFrame) : df corresponding to 1 bout
        TODO:  do results change a lot if RDP on full trajectory instead of each bout?

    Keyword Arguments:
        xvar -- _description_ (default: {'ft_posx'})
        yvar -- _description_ (default: {'ft_posy'})
        theta_range -- _description_ (default: {(0, 2*np.pi)})

    Returns:
        _description_
    '''
    rdp_var ='rdp_{}'.format(xvar)
    #rdp_y ='rdp_{}'.format(yvar)
    xv = b_[b_[rdp_var]][xvar]
    yv = b_[b_[rdp_var]][yvar]
#    if theta_range[0]==0:
#        angles = convert_cw(np.arctan2(np.gradient(xv*3), np.gradient(yv*3)) )
#        assert angles.min().round(1)>=0, "Min ({:.2f}) is not 0".format(angles.min())
#        assert np.ceil(angles.max())>np.pi, "Min ({:.2f}) is not 2pi".format(angles.max())
#    else:
    #print("-np.pi tp pi")
    angles = np.arctan2(np.gradient(xv*3), np.gradient(yv*3)) 
    assert theta_range[0]==-np.pi and theta_range[1]==np.pi, "Wrong theta range"
    #assert angles.min().round(1)>=0, "Min ({:.2f}) is not 0".format(angles.min())
    #assert angles.max().round(1)<=round(2*np.pi, 1), "Min ({:.2f}) is not 2pi".format(angles.max())
    # print('cw arctan2: ({:.2f}, {:.2f})'.format(angles.min(), angles.max()))
    #b_['rdp_arctan2'] = None
    b_.loc[b_[rdp_var], 'rdp_arctan2'] = util.unwrap_and_constrain_angles(angles)
    #b_['rdp_arctan2'] = b_['rdp_arctan2'].astype(float)
    return b_

def mean_heading_across_rdp(b_, xvar='ft_posx', yvar='ft_posy', 
                    heading_var='ft_heading', theta_range=(-np.pi, np.pi)):
    rdp_var='rdp_{}'.format(xvar) 
    ixs = b_[b_[rdp_var]].index.tolist()
    mean_angles=[] 
    for i, ix in enumerate(ixs[0:-1]):
        ang = sts.circmean(b_.loc[ix:ixs[i+1]][heading_var], \
                        high=theta_range[1], low=theta_range[0]) #high=2*np.pi, low=0)
        mean_angles.append(ang)
        if i==0:
            mean_angles.append(ang)
    mean_angles = np.array(mean_angles)

#    if theta_range[0]==0: 
#        assert np.floor(mean_angles.min())>=0, "Min ({:.2f}) is not > 0".format(mean_angles.min())
#        assert np.ceil(mean_angles.max())>np.floor(np.pi), "Min ({:.2f}) is not 2pi".format(mean_angles.max())
#    print('mean angles: ({:.2f}, {:.2f})'.format(min(mean_angles), max(mean_angles)))

    b_.loc[b_[rdp_var], 'mean_angle'] = util.unwrap_and_constrain_angles(mean_angles)

    return b_

def examine_heading_in_bout(b_, theta_range=(-np.pi, np.pi), xvar='ft_posx', yvar='ft_posy',
                        heading_var_og='ft_heading', heading_var='ft_heading', theta_cmap='hsv', 
                        leg_size=0.1, show_angles=False):
    fig, axn = pl.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)
    ax=axn[0, 0]
    sns.scatterplot(data=b_, x="ft_posx", y="ft_posy", ax=ax,
                    hue='time', s=4, edgecolor='none', palette='viridis')
    ax.legend(bbox_to_anchor=(-0.1, 1.01), ncols=2, loc='lower left', title='time')
    # --------------------------------------------------------- 
    # plot given heading (from ft, or wherever validating to)
    # ---------------------------------------------------------
    norm = mpl.colors.Normalize(theta_range[0], theta_range[1])

    ax=axn[0, 1]
    if 'deg' in heading_var_og:
        heading_norm = tuple(np.rad2deg(theta_range)) 
    else:
        heading_norm = norm
    sns.scatterplot(data=b_, x="ft_posx", y="ft_posy", ax=ax,
                    hue=heading_var_og, s=5, edgecolor='none', palette=theta_cmap,
                    hue_norm=heading_norm) #tuple(np.rad2deg(theta_range)))
    ax.legend(bbox_to_anchor=(-0.1, 1.01), ncols=2, loc='lower left', title=heading_var_og)
    wheel_axis = [0.75, 0.3, leg_size, leg_size] if show_angles else [0.75, 0.7, leg_size, leg_size]
    cax = util.add_colorwheel(fig, axes=wheel_axis, 
                    theta_range=theta_range, cmap=theta_cmap) 
    # ---------------------------------------------------------
    # plot calculated direction vectors 
    # ---------------------------------------------------------
    ax=axn[1, 0]; ax.set_title('rdp-arctan2')
    b_ = rdp_to_heading(b_, xvar='ft_posx', yvar='ft_posy', theta_range=theta_range)
    rdp_var ='rdp_{}'.format(xvar)
    # -- 
    ax.plot(b_[xvar], b_[yvar], 'w', lw=0.5)
    ax = plot_bout(b_[b_[rdp_var]], ax, hue_var='rdp_arctan2', cmap=theta_cmap, norm=norm,
                markersize=25, plot_legend=show_angles)
    ax = add_colored_lines(b_[b_[rdp_var]], ax, hue_var='rdp_arctan2', cmap=theta_cmap, norm=norm)
    # ---------------------------------------------------------
    # mean angles
    # ---------------------------------------------------------
    ax=axn[1,1]; ax.set_title('mean angles')
    ax.plot(b_[xvar], b_[yvar], 'w', lw=0.5)
    b_ = mean_heading_across_rdp(b_, heading_var=heading_var, theta_range=theta_range)
    ax = plot_bout(b_[b_[rdp_var]], ax, hue_var='mean_angle', norm=norm, cmap=theta_cmap,
                markersize=25, plot_legend=show_angles)
    ax = add_colored_lines(b_[b_[rdp_var]], ax, hue_var='mean_angle', cmap=theta_cmap, norm=norm)

    return fig

def plot_bout(b_, ax, xvar='ft_posx', yvar='ft_posy', hue_var='time', 
                norm=None, cmap='viridis', hue_title=None, alpha=0.7, 
                plot_legend=True, plot_cbar=False, ncols=1, markersize=10):

    if hue_title is None:
        hue_title = hue_var
    fig = ax.figure

    #sns.scatterplot(data=b_, x="ft_posx", y="ft_posy", ax=ax,
    #                hue='ft_heading_deg', s=4, edgecolor='none', palette=theta_cmap,
    #                hue_norm=tuple(np.rad2deg(theta_range)))

    ax = sns.scatterplot(data=b_, x=xvar, y=yvar, ax=ax, alpha=alpha,
                    hue=hue_var, s=markersize, edgecolor='none', 
                    palette=cmap, hue_norm=norm, legend=plot_legend)

    pl.setp(ax.collections, alpha=alpha)
    if plot_legend:
        if plot_cbar:
            ax.legend_.remove()
            sm =  mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.ax.set_title(hue_title, fontsize=10)
            cbar.ax.tick_params(labelsize=10)
        else:
            leg = ax.legend(bbox_to_anchor=(1, 1.), ncols=ncols, loc='upper left', \
                  title=hue_title, fontsize=6, frameon=False)
            ax.get_legend()._legend_box.align = "left"
            pl.setp(leg.get_title(),fontsize='x-small')
    #else:
    #    ax.legend_.remove()

    return ax

def add_colored_lines(b_, ax, xvar='ft_posx', yvar='ft_posy', 
                    hue_var='heading', cmap='hsv', norm=None):
    #if norm is None:
    #    mpl.colors.Normalize(theta_range[0], theta_range[1])
    assert norm is not None, "Must provide norm"
    xy = b_[[xvar, yvar]].values
    xy = xy.reshape(-1, 1, 2)
    huev = b_[hue_var].values
    #print(huev.dtype)
    segments = np.hstack([xy[:-1], xy[1:]])
    coll = mpl.collections.LineCollection(segments, cmap=cmap, norm=norm) #plt.cm.gist_ncar)
    coll.set_array(huev) #np.random.random(xy.shape[0]))
    ax.add_collection(coll)
    return ax

def examine_heading_at_stops(b_, xvar='ft_posx', yvar='ft_posy',
                    theta_range=(-np.pi, np.pi), theta_cmap='hsv', show_angles=False):
    fig, axn = pl.subplots(1, 4, figsize=(10, 4), sharex=True, sharey=True)
    ax=axn[0]
    cmap = pl.get_cmap("viridis")
    b_['time'] -= b_['time'].iloc[0]
    norm = pl.Normalize(b_['time'].min(), b_['time'].max())
    hue_title = 'time'    
    plot_bout(b_, ax, hue_var='time', norm=norm, cmap='viridis', 
                hue_title=hue_title, plot_cbar=True)
    # ---------------------
    ax=axn[1]
    cmap = pl.get_cmap("magma")
    norm = pl.Normalize(b_['speed'].min(), b_['speed'].max())
    plot_bout(b_, ax, hue_var='speed', norm=norm, cmap=cmap, 
                hue_title=None, plot_cbar=True, alpha=0.5)
    # -------------------------- 
    ax=axn[2]
    if b_.shape[0]>10000:
        skip_every=20
        print("skipping some")
    else:
        skip_every=1
    palette={True: 'r', False: 'w'}
    n_stops_in_bout = len(b_[b_['stopped']]['stopboutnum'].unique())
    hue_title =  'stopped (n={})'.format(n_stops_in_bout)
    ax = plot_bout(b_.iloc[0::skip_every], ax, hue_var='stopped', xvar='ft_posx', yvar='ft_posy',
                cmap=palette, hue_title=hue_title, ncols=1, alpha=0.5, plot_legend=True)
    # ---------------------
    ax=axn[3]; #ax.set_title('rdp-heading')
    theta_norm = mpl.colors.Normalize(theta_range[0], theta_range[1])
    b_ = rdp_to_heading(b_, xvar='ft_posx', yvar='ft_posy', theta_range=theta_range)
    rdp_var ='rdp_{}'.format(xvar)
    ax.plot(b_[xvar], b_[yvar], 'w', lw=0.5)
    b_['rdp_arctan2_deg'] = np.rad2deg(b_['rdp_arctan2'])
    ax = plot_bout(b_[b_[rdp_var]], ax, xvar='ft_posx', yvar='ft_posy', 
                hue_var='rdp_arctan2', norm=theta_norm, cmap=theta_cmap,
                markersize=20, plot_legend=show_angles)
    if show_angles:
        ax.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=6)
    ax = add_colored_lines(b_[b_[rdp_var]], ax, hue_var='rdp_arctan2', cmap=theta_cmap, norm=theta_norm)
    # ------
    # legend
    leg_size=0.1
    wheel_axis = [0.8, 0.3, leg_size, leg_size] if show_angles else [0.8, 0.6, leg_size, leg_size]
    cax = util.add_colorwheel(fig, axes=wheel_axis, 
                    theta_range=theta_range, cmap=theta_cmap) 
    #cax = util.add_colorwheel(fig, axes=[0.8, 0.6, 0.1, 0.1], theta_range=theta_range, cmap='hsv') 
    cax.set_title('rdp-heading', fontsize=7)
    # -----
    pl.subplots_adjust(right=0.8, top=0.7, wspace=0.8, bottom=0.2, left=0.1)

    return fig

def examine_heading_at_stops_vectors(b_, xvar='ft_posx', yvar='ft_posy',
                    theta_range=(-np.pi, np.pi), theta_cmap='hsv', show_angles=False, scale=1.5):
    fig, axn = pl.subplots(1, 3, figsize=(8, 4), sharex=True, sharey=True)
    # ---------------------
    ax=axn[0]
    vmin, vmax = b_['speed'].min(), b_['speed'].max()
    util.plot_vector_path(ax, b_[xvar].values, b_[yvar].values, 
                      b_['speed'].values, vmin=vmin, vmax=vmax, scale=scale)
    # -------------------------- 
    ax=axn[1]
    if b_.shape[0]>10000:
        skip_every=20
        print("skipping some")
    else:
        skip_every=1
    palette={True: 'r', False: 'w'}
    n_stops_in_bout = len(b_[b_['stopped']]['stopboutnum'].unique())
    hue_title =  'stopped (n={})'.format(n_stops_in_bout)
    ax = plot_bout(b_.iloc[0::skip_every], ax, hue_var='stopped', xvar='ft_posx', yvar='ft_posy',
                cmap=palette, hue_title=hue_title, ncols=1, alpha=0.5, plot_legend=True)
    # ---------------------
    ax=axn[2]; #ax.set_title('rdp-heading')
    theta_norm = mpl.colors.Normalize(theta_range[0], theta_range[1])
    b_ = rdp_to_heading(b_, xvar='ft_posx', yvar='ft_posy', theta_range=theta_range)
    rdp_var ='rdp_{}'.format(xvar)
    ax.plot(b_[xvar], b_[yvar], 'w', lw=0.5)
    b_['rdp_arctan2_deg'] = np.rad2deg(b_['rdp_arctan2'])
    ax = plot_bout(b_[b_[rdp_var]], ax, xvar='ft_posx', yvar='ft_posy', 
                hue_var='rdp_arctan2', norm=theta_norm, cmap=theta_cmap,
                markersize=20, plot_legend=show_angles)
    if show_angles:
        ax.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=6)
    ax = add_colored_lines(b_[b_[rdp_var]], ax, hue_var='rdp_arctan2', cmap=theta_cmap, norm=theta_norm)
    # ------
    # legend
    leg_size=0.1
    wheel_axis = [0.8, 0.3, leg_size, leg_size] if show_angles else [0.8, 0.6, leg_size, leg_size]
    cax = util.add_colorwheel(fig, axes=wheel_axis, 
                    theta_range=theta_range, cmap=theta_cmap) 
    #cax = util.add_colorwheel(fig, axes=[0.8, 0.6, 0.1, 0.1], theta_range=theta_range, cmap='hsv') 
    cax.set_title('rdp-heading', fontsize=7)
    # -----
    pl.subplots_adjust(right=0.8, top=0.7, wspace=0.6, bottom=0.2, left=0.1)

    return fig




# data processing
def get_speed_and_stops(b_, speed_thresh=1.0, stopdur_thresh=0.5,
                        xvar='ft_posx', yvar='ft_posy'):
    b_ = calculate_speed(b_, xvar=xvar, yvar=yvar)
    b_ = calculate_stops(b_, stop_thresh=speed_thresh, speed_varname='speed')
    b_ = parse_bouts(b_, count_varname='stopped', bout_varname='stopboutnum')
    b_ = filter_bouts_by_dur(b_, bout_thresh=stopdur_thresh, \
                               count_varname='stopped', bout_varname='stopboutnum')
    return b_

def mean_dir_after_stop(df, heading_var='ft_heading',theta_range=(-np.pi, np.pi),
                xvar='ft_posx', yvar='ft_posy'):
    rdp_var = 'rdp_{}'.format(xvar)
    #speed_thresh=1.0, stopdur_thresh=0.5,
    d_list = []
    i=0
    for bnum, b_ in df.groupby('boutnum'):
        # b_ = get_speed_and_stops(b_, speed_thresh=speed_thresh, stopdur_thresh=stopdur_thresh)
        xwind_dist = b_['crosswind_dist'].sum() - b_['crosswind_dist'].iloc[0]
        stopbouts = b_[b_['stopped']]['stopboutnum'].unique()
        #print(bnum, len(stopbouts))
        instrip_vals = b_['instrip'].unique()
        assert len(instrip_vals)==1, "Non-unique vals instrip (boutnum={})".format(bnum)

        for snum, s_ in b_[b_['stopboutnum'].isin(stopbouts)].groupby('stopboutnum'):
            start_ix = s_.iloc[0].name
            #end_ix = b_[b_[rdp_var]].loc[start_ix:].iloc[1].name
            rest_of_chunk = b_[b_[rdp_var]].loc[start_ix:]
            if rest_of_chunk.shape[0] <= 1:
                end_ix = rest_of_chunk.iloc[0].name
            else:
                end_ix = rest_of_chunk.iloc[1].name # pref to get next rdp
            s_chunk = b_.loc[start_ix:end_ix]
            d_ = pd.DataFrame({
                'fly_id': b_['fly_id'].unique()[0],
                'trial_id': b_['trial_id'].unique()[0],
                'condition': b_['condition'].unique()[0],
                'boutnum': bnum,
                'crosswind_dist': xwind_dist,
                'stopboutnum': snum,
                'instrip': instrip_vals[0],
                'meandir': np.rad2deg(sts.circmean(s_chunk[heading_var],\
                                high=theta_range[1], low=theta_range[0]))},
                index=[i]
            )
            i+=1
            d_list.append(d_)
    meandirs = pd.concat(d_list)
    return meandirs


def get_bout_metrics(df_, group_vars=['fly_id', 'condition', 'boutnum']):
    single_vals = [i for i in df_.columns if len(df_[i].unique())==1\
                  and i not in group_vars]
    single_metrics = df_[single_vals].drop_duplicates().reset_index(drop=True).squeeze()
    
    lin_vars = ['speed', 'upwind_speed', 'crosswind_speed']
    lin_metrics = df_[lin_vars].mean() #.reset_index(drop=True)
    
    misc = pd.Series({
        'duration': df_['time'].iloc[-1] - df_['time'].iloc[0],
        'upwind_dist_range': df_['ft_posy'].max() - df_['ft_posy'].min(),
        'upwind_dist_firstlast': df_['ft_posy'].iloc[-1] - df_['ft_posy'].iloc[0],
        'crosswind_dist_range': df_['ft_posx'].max() - df_['ft_posx'].min(),
        'crosswind_dist_firstlast': df_['ft_posx'].iloc[-1] - df_['ft_posx'].iloc[0],
        'path_length': df_['euclid_dist'].sum() -  df_['euclid_dist'].iloc[0],
        'average_heading': sts.circmean(df_['ft_heading'], low=-np.pi, high=np.pi),
        'rel_time': df_['rel_time'].iloc[0]
    }) #, index=[0])
    
    #metrics = pd.DataFrame(pd.concat([misc, lin_metrics, single_metrics])).T    
    metrics = pd.concat([misc, lin_metrics, single_metrics]) 
    return metrics





def summarize_stops_and_turns(df_, meanangs_, last_,  strip_width=10, strip_sep=200, 
                    xvar='ft_posx', yvar='ft_posy',
                    laststop_color='b', stop_color=[0.9]*3, theta_range=(-np.pi, np.pi), offset=20,
                    theta_cmap='hsv', instrip_palette={True: 'r', False: 'w'}, xlims=None,
                    stop_marker='.', stop_markersize=3, lw=0):
    
    trial_id = df_['trial_id'].unique()[0]
    # Construct figure and axis to plot on
    fig = pl.figure( figsize=(8, 6))
    ax = fig.add_subplot(1, 2, 1)
    #oparams = get_odor_params(df_, strip_width=strip_width, is_grid=True)
    strip_borders = find_strip_borders(df_, strip_width=strip_width, strip_sep=strip_sep, 
                                entry_ix=None)

    ax= plot_trajectory(df_, ax=ax, odor_bounds=strip_borders, #oparams['odor_boundary'], 
                              palette=instrip_palette)
    # plot all stops
    ax.scatter(df_[df_['stopped']][xvar], df_[df_['stopped']][yvar], marker=stop_marker,
                    s=stop_markersize, c=stop_color, linewidth=lw)
    # plot last stops
    ax.scatter(df_[df_['is_last']][xvar], df_[df_['is_last']][yvar], marker=stop_marker,
                    s=stop_markersize, c=laststop_color, linewidth=lw)
    if xlims is None:
        xlims = (df_[xvar].min()-offset, df_[xvar].max()+offset)
    if xlims is not None:
        ax.set_xlim(xlims) #xlims[fly_id])
    #lg = ax.legend()
    legh, labels = ax.get_legend_handles_labels()
    legh1 = [mpl.lines.Line2D([0], [0], color=c, lw=2) for c in [stop_color, laststop_color] ]
    legh.extend(legh1)
    ax.legend(handles=legh, labels=['outstrip', 'instrip', 'all stops', 'last stop'],
              bbox_to_anchor=(1.1, 1), loc='upper left', fontsize=6, frameon=False)

    # polar plot of turn angles at stops
    ax = fig.add_subplot(1, 2, 2, projection='polar')
    n1, _, _ = util.circular_hist(ax, np.deg2rad(meanangs_['meandir']), bins=40, 
                  facecolor=stop_color, density=False, edgecolor='none', alpha=0.9)
    n2, _, _ = util.circular_hist(ax, np.deg2rad(last_['meandir']), bins=40, 
                  facecolor=laststop_color, density=False, alpha=.7, edgecolor='none')
    ylim = max([n1.max(), n2.max()])
    ax.set_ylim([0, ylim+1])
    ax.set_xticklabels([])

    # legends
    axes=[0.85, 0.6, 0.1, 0.1]
    util.add_colorwheel(fig, cmap=theta_cmap, theta_range=theta_range, axes=axes)


    pl.subplots_adjust(left=0.1, wspace=0.5, top=0.8, right=0.85, bottom=0.2)
    return fig



# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------

def plot_trajectory_from_file(fpath, parse_filename=False, strip_width=10, strip_sep=200, ax=None):
    # load and process the csv data  
    df0 = load_dataframe(fpath, verbose=False, cond=None, 
                parse_filename=False)
    fly_id=None
    if parse_filename:
        # try to parse experiment details from the filename
        exp, datestr, fid, cond = parse_info_from_filename(fpath)
        print('Experiment: {}{}Fly ID: {}{}Condition: {}'.format(exp, '\n', fid, '\n', cond))
        fly_id = df0['fly_id'].unique()[0]
    else:
        fly_id = os.path.split(fpath)[-1]
        df0['fly_id'] = fly_id

    # get experimentally determined odor boundaries:
    #ogrid, in_odor = get_odor_grid(df0, strip_width=strip_width, 
    #                        strip_sep=strip_sep, use_crossings=True, verbose=False )
    #(odor_xmin, odor_xmax), = ogrid.values()
    #odor_bounds = list(ogrid.values())
    strip_borders = find_strip_borders(df0, strip_width=strip_width, strip_sep=strip_sep, 
                                entry_ix=None)
    title = os.path.splitext(os.path.split(fpath)[-1])[0]
    plot_trajectory(df0, odor_bounds=strip_borders, title=title, ax=ax)

    return ax

def plot_trajectory(df0, odor_bounds=[], ax=None,
        xvar='ft_posx', yvar='ft_posy',
        hue_varname='instrip', palette={True: 'r', False: 'w'}, 
        start_at_odor = True, odor_lc='lightgray', odor_lw=0.5, title='',
        markersize=0.5, alpha=1.0, 
        center=False, xlim=200, plot_odor_onset=True, plot_legend=True):

    # ---------------------------------------------------------------------
    if ax is None: 
        fig, ax = pl.subplots()
    if not isinstance(odor_bounds, list):
        odor_bounds = [odor_bounds]
    sns.scatterplot(data=df0, x=xvar, y=yvar, ax=ax, 
                    hue=hue_varname, s=markersize, edgecolor='none', palette=palette,
                    legend=plot_legend, alpha=alpha)
    # odor corridor
    for (odor_xmin, odor_xmax) in odor_bounds:
        plot_odor_corridor(ax, odor_xmin=odor_xmin, odor_xmax=odor_xmax)
    # odor start time
    if df0[df0['instrip']].shape[0]>0:
        odor_start_ix = df0[df0['instrip']].iloc[0][yvar]
        if plot_odor_onset:
            ax.axhline(y=odor_start_ix, color='w', lw=0.5, linestyle=':')
    ax.legend(bbox_to_anchor=(1,1), loc='upper left', title=hue_varname, frameon=False)
    ax.set_title(title)
    xmax=500
    if center:
        ax.set_xlim([-xlim, xlim]) 
    else:
        try:
            # Center corridor   
            xmin = np.floor(df0[xvar].min())
            xmax = np.ceil(df0[xvar].max())
            ax.set_xlim([xmin-20, xmax+20])
        except ValueError as e:
            xmax = 500
    pl.subplots_adjust(left=0.2, right=0.8)

    return ax

def plot_odor_corridor(ax, odor_xmin=-100, odor_xmax=100, \
                    odor_linecolor='gray', odor_linewidth=0.5,
                    offset=10):
    ax.axvline(odor_xmin, color=odor_linecolor, lw=odor_linewidth)
    ax.axvline(odor_xmax, color=odor_linecolor, lw=odor_linewidth)
    #xmin = min([odor_xmin-offset, min(ax.get_xlim())])
    #xmax = max([odor_xmax + offset, min(ax.get_xlim())])
    #ax.set_xlim([xmin, xmax])
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


def plot_all_flies(df_, hue_varname='instrip',  plot_borders=True,
                    palette={True: 'r', False: 'w'}, strip_width=50, strip_sep=200,
                    col_wrap=4, is_grid=True):
    trial_ids = list(df_.groupby('trial_id').groups.keys())
    g = sns.FacetGrid(df_, col='trial_id', col_wrap=col_wrap,
                    col_order=trial_ids)
    g.map_dataframe(sns.scatterplot, x="ft_posx", y="ft_posy", hue=hue_varname,
                s=0.5, edgecolor='none', palette=palette) #, palette=palette)
    g.set_titles(row_template = '{row_name}', col_template = '{col_name}', size=6)
    if plot_borders:
        # add strip borders
        for ax, trial_id in zip(g.axes.flat, trial_ids):
            odor_borders = find_strip_borders(df_[df_['trial_id']==trial_id],
                                    strip_width=strip_width, strip_sep=strip_sep, 
                                    get_all_borders=True, entry_ix=None, is_grid=is_grid)  
            #odor_half = strip_width/2.
            for (odor_xmin, odor_xmax) in odor_borders:
                plot_odor_corridor(ax, odor_xmin=odor_xmin, odor_xmax=odor_xmax) 
            ax.set_xlim([-300, 300])

    return g.fig

def plot_fly_by_condition(plotdf, strip_width=50, hue_varname='instrip', 
                            palette={True: 'r', False: 'w'}):
    '''
    create facet grid of fly (cols) by condition (rows)

    Arguments:
        plotdf -- _description_

    Keyword Arguments:
        strip_width -- _description_ (default: {50})
        hue_varname -- _description_ (default: {'instrip'})
        palette -- _description_ (default: {{True: 'r', False: 'w'}})

    Returns:
        _description_
    '''
    g = sns.FacetGrid(plotdf, col='fly_id', row='condition', 
                    col_order=list(plotdf.groupby('fly_id').groups.keys()),
                    row_order = list(plotdf.groupby('condition').groups.keys()))

    g.map_dataframe(sns.scatterplot, x="ft_posx", y="ft_posy", hue=hue_varname,
                s=0.5, edgecolor='none', palette=palette) #, palette=palette)
    g.set_titles(row_template = '{row_name}', col_template = '{col_name}', size=6)
    # add strip borders
    for ax in g.axes.flat:
        odor_half = strip_width/2.
        plot_odor_corridor(ax, odor_xmin=-odor_half, odor_xmax=odor_half) 
        ax.set_xlim([-300, 300])

    return g.fig

def get_id_columns():
    id_cols = ['fly_id', 'trial_id', 'condition', 'strip_type', 'instrip', \
            'bout_type', 'inout-strip', 'boutnum']
    return id_cols

def plot_metrics_displot(df, plot_vars, hue_var='instrip', row_var=None,
                    labels=[True, False], colors=['r', 'w'],
                    limit_xaxis=False, xlim=None, fontsize=7,
                    sharex=False, sharey=True, height=2):
    curr_palette = dict((k, v) for k, v in zip(labels, colors))
    #melt_vars = [c for c in df.columns if c not in plot_vars]

    id_cols = get_id_columns()
    id_vars = [c for c in id_cols if c in df.columns]
    df_ = df.melt(id_vars, var_name='varname', value_name='varvalue')
    plotdf = df_[df_['varname'].isin(plot_vars)]

    g = sns.displot(
        data = plotdf,
        x='varvalue', col='varname', hue=hue_var, row=row_var,
        aspect=1, height=height, lw=1, kind='ecdf',
        palette=curr_palette, 
        facet_kws={"sharex": sharex, "sharey": sharey}
    )
    g.set_titles(row_template='{row_name}', col_template='{col_name}', size=6)
    for ax in g.axes.flat:
        tstr = ax.get_title()
        ax.set_xlabel(tstr, fontsize=fontsize)
        ax.tick_params(which='both', axis='both', labelsize=fontsize)
        #ax.set_xticklabels(ax.get_xticks(), ax.get_xticklabels(), fontsize=fontsize)
        #ax.set_yticklabels(ax.get_yticks(), ax.get_yticklabels(), fontsize=fontsize)
        ax.set_ylabel('Proportion', fontsize=fontsize)
        ax.set_title('')
        ax.set_box_aspect(1)
    return g.fig

def plot_metrics_hist(df, plot_vars, hue_var='instrip', row_var=None,
                labels=[True, False], colors=['r', 'w'], 
                plot_log=False, cumulative=False, stat='probability', 
                kde=False, limit_xaxis=False, xlim=None, 
                sharex=False, sharey=True):
    '''
    Plot distributions of variables (plot_vars) as a row of plots.

    Arguments:
        df -- _description_

    Keyword Arguments:
        hue_var -- str, hue variable
        l1 -- hue label1 (default: {True})
        l2 -- hue label2 (default: {False})
        c1 -- color1 label (default: {'r'})
        c2 -- color2 label (default: {'w'})
        plot_log -- bool, plot x on log-scale (default: {False})
        cumulative -- bool, plot as CDF (default: {False})
        stat -- str, pick density for bar areas=1 and prob for heights=1 (default: {'probability'})
        kde -- bool, (default: {False})
        limit_xaxis -- bool, if long tail (default: {False})
        xlim -- tuple, list (default: {None})
        plot_vars -- list of vars to plot (default: {[]})

    Returns:
        _description_
    '''
    curr_palette=dict((k, v) for k, v in zip(labels, colors))

    id_cols = get_id_columns()
    id_vars = [c for c in id_cols if c in df.columns]
    df_ = df.melt(id_vars, var_name='varname', value_name='varvalue')
    plotdf = df_[df_['varname'].isin(plot_vars)]
    g = sns.FacetGrid(plotdf, col='varname', row=row_var,
                        sharex=sharex, sharey=sharey)
    g.map_dataframe(sns.histplot, x='varvalue', hue=hue_var,
                fill=False, element='step', 
                stat='probability', common_norm=False, 
                palette=curr_palette)
    if plot_log:
        g.set(yscale='log')

    g.set_titles(row_template='{row_name}', col_template='{col_name}', size=6)
    for ax in g.axes.flat:
        tstr = ax.get_title()
        #ylabel, xlabel = tstr.split(' | ')
        ax.set_xlabel(tstr) #, fontsize)=fontsize)
        ax.set_xticklabels(ax.get_xticklabels()) #, fontsize=fontsize)
        ax.set_yticklabels(ax.get_yticklabels()) #, fontsize=fontsize)
        ax.set_ylabel('Proportion') #, fontsize=fontsize)
        ax.set_title('')
        if limit_xaxis:
            ax.set_xlim([-10, 300])
        ax.set_box_aspect(1)
    sns.despine()

    # custom legend
    return g.fig


def check_mfc_vars(df0_all, file_id='filename'):
    mfc_vars = [c for c in df0_all.columns if 'mfc' in c]
    melt_cols = [c for c in df0_all.columns if c not in mfc_vars]
    meltdf = df0_all.melt(melt_cols)
    g = sns.FacetGrid(meltdf, col=file_id, col_wrap=5, height=2, sharex=False)
    g.map_dataframe(sns.scatterplot, x='ft_frame', y='value', hue='variable', 
                    edgecolor='none', s=3)
    g.set_titles(row_template='{row_name}', col_template='{col_name}', size=6)
    g.axes.flat[-1].legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=7)
    pl.subplots_adjust(top=0.9)

    return g.fig

