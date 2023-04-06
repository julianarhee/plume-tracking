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
import scipy as sp
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
                   max_instrip_upwind_percent=0.5, 
                   max_crossovers=3, min_outside_bouts=5, 
                   min_global_upwind_dist=100,
                   max_crossover_to_dist_ratio=2/500):
    '''
    Find edge tracking bouts in a trajectory. Starts from first OUT bout, ends with last IN bout.


    Arguments:
        df0 -- _description_
        odor_bounds -- _description_

    Keyword Arguments:
        strip_width -- _description_ (default: {50})
        strip_sep -- _description_ (default: {1000})
        max_instrip_upwind_percent -- _description_ (default: {0.4})
        max_crossovers -- _description_ (default: {2})
        min_outside_bouts -- _description_ (default: {5})
        min_global_upwind_dist -- _description_ (default: {500})

    Returns:
        _description_
    '''
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
            # start from 1st outbout
            start_bout = df0[df0['instrip']].iloc[0]['boutnum'].min() + 1
            # end at last inbout
            end_bout = df0[df0['instrip']]['boutnum'].max()
        #print("checking bouts for ET: ", start_bout, end_bout)
        if df_.shape[0]==0:
            continue

        df0_filt = df_[(df_['boutnum']>=start_bout) & (df_['boutnum']<=end_bout)].copy()
        entry_ix = df_[df_['ft_posx']>ob[0]].iloc[0].name

        # check if ET
        is_et, etparams, curr_pass_key = butil.is_edgetracking(df0_filt, 
                            return_key=True,
                            strip_width=strip_width, 
                            split_at_crossovers=False,
                            crop_first_last=False,
                            min_outside_bouts=min_outside_bouts,
                            max_crossovers=max_crossovers,
                            min_global_upwind_dist=min_global_upwind_dist, 
                            max_instrip_upwind_percent=max_instrip_upwind_percent,
                            max_crossover_to_dist_ratio=max_crossover_to_dist_ratio
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

def filter_first_instrip_last_outstrip(anydf, y_thresh=None):
    '''
    Remove 1st instrip (odor starts on top of fly), and last outstrip (wandering off). 
    
    Arguments:
        anydf: full pd.DataFrame (sorts by filename)
    Keyword Arguments:
        y_thresh (None, float): if specified, takes the first bout >= a given y-position threshold 
        (specifically for T-plume)
    Returns:
        anydf_filt: input dataframe, but with start/end bouts removed.
        
    '''
    d_list = []
    for fn, df_ in anydf.groupby('filename'):
        if y_thresh is not None:
            start_bout = df_[df_['ft_posy']>=y_thresh].iloc[0]['boutnum']+1
        else:
            start_bout = df_[df_['instrip']]['boutnum'].min()+1
        end_bout = df_[df_['instrip']]['boutnum'].max()
        if end_bout==1: # condition where fly just wanders off
            start_bout = 1 #df_[df_['instrip']]['boutnum'].min()
            end_bout = df_['boutnum'].max()
        tmpdf = df_[(df_['boutnum']<=end_bout) & (df_['boutnum']>=start_bout)]
        d_list.append(tmpdf)

    anydf_filt = pd.concat(d_list, axis=0).reset_index(drop=True)
    
    return anydf_filt #, trajdf_filt

## combining traces
def upsample_bout_trajectories(df_, npoints=1000):
    '''
    For all bouts of 1 condition type (instrip, pre-led, etc.),
    upsample by interpolating each bout curve.
    Shape of bout should be the same (start/end point, too), but just 
    scaled to match npoints.
    '''
    avg_x_in = []
    avg_y_in = []
    x_list=[]; y_list=[];
    x_list2=[]; y_list2=[];
    for bnum, b_ in df_.dropna().groupby('boutnum'):
        x = b_['ft_posx'].values - b_['ft_posx'].values[0]
        y = b_['ft_posy'].values - b_['ft_posy'].values[0]

        t = np.arange(len(x))
        t_common = np.linspace(t[0], t[-1], npoints)
        fx = sp.interpolate.interp1d(t, x)
        fy = sp.interpolate.interp1d(t, y)

        interpx =fx(t_common)
        interpy = fy(t_common)
        avg_x_in.append(interpx)
        avg_y_in.append(interpy)
        
        x_list.append(pd.DataFrame(data=x, columns=[bnum]))
        y_list.append(pd.DataFrame(data=y, columns=[bnum]))
        
        x_list2.append(pd.DataFrame(data=interpx, columns=[bnum]))
        y_list2.append(pd.DataFrame(data=interpy, columns=[bnum]))
        
    og_x = pd.concat(x_list, axis=1)
    og_y = pd.concat(y_list, axis=1)

    interp_x = pd.concat(x_list2, axis=1)
    interp_y = pd.concat(y_list2, axis=1)

    return interp_x, interp_y

def align_and_average_bout_center0(df_, end_at_zero=False):

    interp_x_in, interp_y_in = upsample_bout_trajectories(df_)
    #interp_x_out, interp_y_out = traj.upsample_bout_trajectories(df_[~df_['instrip']])
   
    offset_ix = -1 if end_at_zero else 0 
    interp_x_in = interp_x_in.apply(lambda x: x-x.iloc[offset_ix])
    interp_y_in = interp_y_in.apply(lambda x: x-x.iloc[offset_ix])
    
    #out_offset = interp_y_in.mean(axis=1).iloc[-1]
    #interp_x_out = interp_x_out.apply(lambda x: x-x.iloc[0]) 
    #interp_y_out = interp_y_out.apply(lambda x: x-x.iloc[0]) #+ out_offset

    avg_x_in = interp_x_in.mean(axis=1)
    avg_y_in = interp_y_in.mean(axis=1)
    #avg_x_out = interp_x_out.mean(axis=1)
    #avg_y_out = interp_y_out.mean(axis=1)

    return interp_x_in, interp_y_in, avg_x_in, avg_y_in


def plot_tortuosity_metrics(tortdf, cdf=False,
                            boutdir_palette={'outbound': 'cyan', 'inbound': 'violet'}):
    fig, axn = pl.subplots(1, 3, figsize=(8,4))
    ax=axn[0]
    ax=sns.histplot(data=tortdf, x='pathlength', hue='boutdir', ax=ax, palette=boutdir_palette,
                 cumulative=cdf, fill=False, element='poly', stat='probability',
                 common_norm=False)
    ax=axn[1]
    ax=sns.histplot(data=tortdf, x='tortuosity', hue='boutdir', ax=ax, palette=boutdir_palette,
                 cumulative=cdf, fill=False, element='poly', stat='probability',
                 common_norm=False)
    ax=axn[2]
    ax=sns.histplot(data=tortdf, x='efficiency', hue='boutdir', ax=ax, palette=boutdir_palette,
                 cumulative=cdf, fill=False, element='poly', stat='probability',
                 common_norm=False)
    for ai, ax in enumerate(axn):
        ax.set_box_aspect(1)
        if ai==2:
            h, l = ax.get_legend_handles_labels()
            #l = ax.legend_data.keys()
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), frameon=False)
        else:
            ax.legend_.remove()
    pl.subplots_adjust(left=0.1, right=0.85, wspace=0.5)
    return fig


# util.set_sns_style(style='dark', min_fontsize=12)

