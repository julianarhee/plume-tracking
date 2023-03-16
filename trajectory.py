#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File           : trajectory.py
Created        : 2023/02/27 9:42:37
Project        : /Users/julianarhee/Repositories/plume-tracking
Author         : jyr
Email          : juliana.rhee@gmail.com
Last Modified  : 
'''

import os
import sys
import glob
import numpy as np
import pandas as pd

import seaborn as sns
import pylab as pl

# import some custom funcs
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import utils as util
import behavior as butil


def load_df(fpath):
    config  = butil.load_experiment_config(fpath)
    #if config is not 
    fn = os.path.split(fpath)[-1]

    if config is None:
        strip_width = 50
    else:
        if int(fn.split('_')[0].split('-')[0]) < 20230106:
            strip_width = 20
        else:
            strip_width = config['experiment']['strip_width'] 
    if fn in ['20230106-121556_strip_grid_fly1_000.log', '20230105-182650_strip_grid_fly1_004.log']:
        strip_width=20 
        strip_sep_default = 200
    elif 'strip_grid' in fpath:
        strip_sep_default = 500
    else:
        strip_sep_default = 1000
    strip_sep = config['experiment']['strip_spacing'] if config is not None else strip_sep_default # 400

    df0_full = butil.load_dataframe(fpath, remove_invalid=False)
    df0_full = butil.process_df(df0_full)

    odor_ix = df0_full[df0_full['instrip']].iloc[0].name

    try:
        odor_bounds = butil.find_strip_borders(df0_full, entry_ix=odor_ix,
                                strip_width=strip_width, strip_sep=strip_sep)
    except StopIteration:
        strip_sep=1000
        try:
            odor_bounds = butil.find_strip_borders(df0_full, entry_ix=odor_ix,
                                strip_width=strip_width, strip_sep=strip_sep)
        except StopIteration:
            return None, None

    oparams = {
        'odor_ix': odor_ix,
        'odor_bounds': odor_bounds,
        'strip_width': strip_width, 
        'strip_sep': strip_sep
    }
    return df0_full, oparams#odor_ix, odor_bounds, strip_width, strip_sep

def find_et_bouts(df0, odor_bounds, strip_width=50, strip_sep=1000, 
                   max_instrip_upwind_percent=0.4, 
                   max_crossovers=2, min_outside_bouts=5, 
                   min_global_upwind_dist=500):

    #max_instrip_upwind_percent = 0.4 #250 #250
    #min_global_upwind_dist = 300
    #max_crossovers=2
    #min_outside_bouts=5

    et_bouts = {}
    et_boutstats = {}
    et_passkey = {}
    for oi, ob in enumerate(odor_bounds):
        if len(odor_bounds)>1:
            curr_ob_min = min(ob)
            next_ob_min = max(ob)+strip_sep/2. #min(odor_bounds[oi+1]) if oi<(len(odor_bounds)-1) else max(ob)
            # get df up until *next* odor strip
            within_bounds = df0[ (df0['instrip']) \
                                & (df0['ft_posx']>=curr_ob_min) \
                                & (df0['ft_posx'] < next_ob_min)].copy()        
            #df_ = df0[ (df0['ft_poxs']>=curr_ob_min) & (df0['ft_posx'] < next_ob_min)].copy()
            start_bout, end_bout = within_bounds['boutnum'].min(), within_bounds['boutnum'].max()
            df_ = df0[(df0['boutnum']>=start_bout) & (df0['boutnum']<=end_bout)].copy()
            print(oi, start_bout, end_bout, next_ob_min, df_.shape)
        else:
            df_ = df0.copy()
            start_bout = df0[df0['instrip']].iloc[0]['boutnum'].min() + 1 # start from 1st outbout
            end_bout = df0[df0['instrip']]['boutnum'].max()
        #print("checking bouts for ET: ", start_bout, end_bout)
        if df_.shape[0]==0:
            continue
        #print( "{}: starts instrip {}".format(oi, df_['instrip'].iloc[0]))
        # calculate duration and N in/outbouts of current ET bout
        #start_bout = df_[df_['instrip']].iloc[0]['boutnum']+1 # measure from first outbout
        #end_bout = df_[df_['instrip']]['boutnum'].max() # until last inbout

        measure_bout = df_[(df_['boutnum']>=start_bout) & (df_['boutnum']<=end_bout)].copy()
        entry_ix = df_[df_['ft_posx']>ob[0]].iloc[0].name
        #et_params = butil.calculate_et_params(measure_bout)
        #etparams = butil.get_edgetracking_params(measure_bout, strip_width=strip_width, 
        #                                         split_at_crossovers=False)
        #print(etparams)
        
        #upwind_dist = measure_bout['upwind_dist'].sum()
        #n_bouts = len(measure_bout['boutnum'].unique())

        # check if ET
        is_et, etparams, curr_pass_key = butil.is_edgetracking(measure_bout, 
                            return_key=True,
                            strip_width=strip_width, 
                            split_at_crossovers=False,
                            crop_first_last=False,
                            min_outside_bouts=min_outside_bouts,
                            max_crossovers=max_crossovers,
                            #min_global_upwind_dist=min_global_upwind_dist, 
                            max_instrip_upwind_percent=max_instrip_upwind_percent
        )
        etparams[0].update({'key': 'c{}'.format(entry_ix)})
        et_boutstats.update({
            oi: etparams
            })

        if is_et:
            print("Is ET:", oi, is_et)
            et_bouts.update({'c{}'.format(entry_ix): ob})
        else:
            print("not et:", oi)
    
        et_passkey[oi] = curr_pass_key 

    return et_bouts, et_boutstats, et_passkey

def get_best_et_boutkey(et_boutstats):
    # NOTE:  dummy key 0 makes nested dict, using placeholder key because split_at_crossovers would
    # have multiple keys per ET bout (see get_edgetracking_params)

    # un-next dict of {et1: {0: {k: v}}} -- leftover from split_at_crossovers.
    param_values = [list(v.values())[0] for v in list(et_boutstats.values())]

    sorted_ets = sorted(param_values, \
                   key=lambda v: (float(v['global_upwind_dist']), v['n_outside_bouts']), reverse=True)
    et_id = [k for k, v in et_boutstats.items() \
             if all([v[i]==sorted_ets[i] for i in list(v.keys())])][0]
    
    et_boutkey = et_boutstats[et_id][0]['key']
    return et_boutkey

def select_best_et_bout(et_boutkey, et_bouts, df0_full, strip_sep=500):
    # select from 
    next_ob = et_bouts[et_boutkey][0]
    # print(next_ob)
    # start from previou outbout, if exists
    odor_ix = df0_full[df0_full['ft_posx']>=next_ob].iloc[0].name
    et_startbout = df0_full.loc[odor_ix]['boutnum'] -1 # start from previous OUT bout
    # make sure not including another strip, if exists
    past_current_strip_x = et_bouts[et_boutkey][1] + strip_sep/2
    et_lastbout = df0_full[ (df0_full['instrip']) \
                           & (df0_full['ft_posx']>=past_current_strip_x)]['boutnum'].min()
    if et_lastbout is not np.nan:
        et_lastbout -= 1
    else:
        et_lastbout = df0_full['boutnum'].max()
    print("Start/End bouts:", et_startbout, et_lastbout)
    df0 = df0_full[(df0_full['boutnum']>=et_startbout)
                  & (df0_full['boutnum']<=et_lastbout)].copy() # include prev bout for "entry into" odor 

    et_bouts = dict((k, v) for k, v in et_bouts.items() if k==et_boutkey)
    #et_bouts

    return df0, odor_ix, et_bouts

def check_and_flip_traj(df0_full, et_boutkey, et_boutstats, et_bouts, strip_width=50):
    all_et_starts = [int(v[0]['key'][1:]) for k, v in et_boutstats.items()]

    tmp_start_ix = int(et_boutkey[1:])-1
    ix = all_et_starts.index(tmp_start_ix + 1)
    if len(all_et_starts)>1:
        tmp_end_ix = all_et_starts[ix+1] if tmp_start_ix < max(all_et_starts)-1\
                            else df0_full.iloc[-1].name
    else:
        tmp_end_ix = df0_full.iloc[-1].name
    #print(tmp_start_ix, tmp_end_ix)
    dfp, obounds_fp = butil.check_entryside_and_flip(df0_full.loc[tmp_start_ix:tmp_end_ix].copy(), 
                                        odor_dict=et_bouts,
                                        strip_width=strip_width)
    return dfp, obounds_fp



def filter_first_instrip_last_outstrip(boutdf):
    '''
    Remove 1st instrip (odor starts on top of fly), and last outstrip (wandering off)
    '''

    d_list = []
    #t_list = []
    for fn, df_ in boutdf.groupby('filename'):
        first_instrip = df_[df_['instrip']]['boutnum'].min()
        last_instrip = df_[df_['instrip']]['boutnum'].max()
        tmpdf = df_[(df_['boutnum']<=last_instrip) & (df_['boutnum']>first_instrip)]
        d_list.append(tmpdf)
        #print(first_instrip, last_instrip, tmpdf.shape)
        #traj_ = etdf[etdf['filename']==fn].copy()
        #traj_tmp = traj_[(traj_['boutnum']<=last_instrip) & (traj_['boutnum']>first_instrip)]
        #t_list.append(traj_tmp)
    boutdf_filt = pd.concat(d_list, axis=0).reset_index(drop=True)
    print(boutdf_filt.shape)
    #trajdf_filt = pd.concat(t_list, axis=0).reset_index(drop=True)
    
    return boutdf_filt #, trajdf_filt



def upsample_trajectory(df_, max_nframes, xvar='ft_posx', yvar='ft_posy', offset=True):
    d_list=[]
    for (trial_id, ep, bnum), b_ in df_.groupby(['trial_id', 'epoch', 'boutnum']):
        px = np.pad(b_[xvar].values, (0, int(max_nframes-len(b_))), 'constant', constant_values=np.nan)
        py = np.pad(b_[yvar].values, (0, int(max_nframes-len(b_))), 'constant', constant_values=np.nan)
        d_ = pd.DataFrame({
                      'boutnum': [bnum]*len(px),
                      'epoch': [ep]*len(px),
                      'trial_id': [trial_id]*len(px),
                       xvar: px - px[0], 
                       yvar: py - py[0], 
                      'ix': np.arange(0, len(px)),
                      'instrip': [b_['instrip'].unique()[0]]*len(px)})
        d_list.append(d_)
    up_ = pd.concat(d_list).reset_index(drop=True)
    return up_



def calculate_tortuosity_metrics(df, xdist_cutoff=1.9, xvar='ft_posx', yvar='ft_posy'):

    last_outbout = df[~df['instrip']]['boutnum'].max()
    max_boutnum = df['boutnum'].max()

# util.set_sns_style(style='dark', min_fontsize=12)

