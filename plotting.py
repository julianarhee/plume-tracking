#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File           : plotting.py
Created        : 2023/03/27 18:57:30
Project        : /Users/julianarhee/Repositories/plume-tracking
Author         : jyr
Email          : juliana.rhee@gmail.com
Last Modified  : 
'''
import numpy as np
import pandas as pd
import scipy.stats as spstats
import matplotlib as mpl
import pylab as pl
import seaborn as sns


import utils as util
import behavior as butil

## generic
# ticks
def set_outward_spines(ax):
    ax.tick_params(which='both', axis='both', length=2, width=0.5, color='w',
               direction='out', left=True)

def remove_spines(ax, axes=['right', 'top']):
    for pos in axes:
       ax.spines[pos].set_visible(False)



def custom_legend(labels, colors, use_line=True, lw=4, markersize=10):
    '''
    Returns legend handles

    Arguments:
        labels -- _description_
        colors -- _description_

    Keyword Arguments:
        use_line -- _description_ (default: {True})
        lw -- _description_ (default: {4})
        markersize -- _description_ (default: {10})

    Returns:
        _description_
    '''
    if use_line:
        legh = [mpl.lines.Line2D([0], [0], color=c, label=l, lw=lw) for c, l in zip(colors, labels)]
    else:
        legh = [mpl.lines.Line2D([0], [0], marker='o', color='w', label=l, lw=0,
                    markerfacecolor=c, markersize=markersize) for c, l in zip(colors, labels)]

    return legh
##

def zero_trajectory(df_):

    plotdf = df_.copy()
    offset_x = plotdf[plotdf['instrip']].iloc[0]['ft_posx']
    offset_y = plotdf[plotdf['instrip']].iloc[0]['ft_posy']
    plotdf['ft_posx'] = plotdf['ft_posx'].values - offset_x
    plotdf['ft_posy'] = plotdf['ft_posy'].values - offset_y
    
    odor_ix = plotdf[plotdf['instrip']].iloc[0].name

    return plotdf.loc[odor_ix:]

def zero_start_position(b_):
    b_['ft_posx_start0'] = b_['ft_posx'] - b_['ft_posx'].iloc[0]
    b_['ft_posy_start0'] = b_['ft_posy'] - b_['ft_posy'].iloc[0]
    return b_

def normalize_position(b_):
    b_['ft_posx_norm'] = util.convert_range(b_['ft_posx_start0'], newmin=0, newmax=1)
    b_['ft_posy_norm'] = util.convert_range(b_['ft_posy_start0'], newmin=0, newmax=1)
    return b_



def plot_zeroed_trajectory(df_, ax=None, traj_lw=1.5, odor_lw=1.0,
                        strip_width=50, strip_sep=1000, main_col='w',
                        bool_colors=['r'], bool_vars=['instrip']):
    if ax is None:
        fig, ax= pl.subplots()
    odor_ix = df_[df_['instrip']].iloc[0].name
    #plotdf = df_.loc[odor_ix:]
    # odor_ix = params[fn]['odor_ix']
    plotdf = zero_trajectory(df_)
    odor_ix = plotdf[plotdf['instrip']].iloc[0].name
    odor_bounds = butil.find_strip_borders(plotdf, entry_ix=odor_ix,
                                        strip_width=strip_width,
                                        strip_sep=strip_sep)
    # plot
    plotdf = plotdf.loc[odor_ix:].copy()
    
    ax.plot(plotdf['ft_posx'], plotdf['ft_posy'], lw=traj_lw, c=main_col)
    for ob in odor_bounds:
        butil.plot_odor_corridor(ax, odor_xmin=ob[0], 
                             odor_xmax=ob[1], odor_linewidth=odor_lw)

    for col, boolvar in zip(bool_colors, bool_vars):
        for bnum, b_ in plotdf[plotdf[boolvar]].groupby('boutnum'):
            ax.plot(b_['ft_posx'], b_['ft_posy'], lw=traj_lw, c=col)

    return ax


def plot_paired_inout_metrics(df_, nr=2, nc=3,
                xvarname='instrip', order=[False, True], xticklabels=['outstrip', 'instrip'],
                yvarnames=['duration', 'path_length',
                'crosswind_speed', 'upwind_speed', 
                'crosswind_dist_range', 'upwind_dist_range']):


    fig, axn = pl.subplots(nr, nc, figsize=(10,5)) #len(varnames))
    for ax, varn in zip(axn.flat, yvarnames):
        plot_paired_in_vs_out(varn, df_, ax=ax, 
                    xvarname=xvarname, order=order, xticklabels=xticklabels)
        #a = df_[df_['instrip']][['filename', varn]]
        a = df_['filename'].unique()
        dof = len(a)-1
        fig.text(0.75, 0.06, 'Wilcoxon signed-rank, n={} trajectories'.format(len(a)),
                fontsize=10)
    pl.subplots_adjust(left=0.1, wspace=0.5, hspace=0.5, right=0.95, bottom=0.2)
    
    return fig

def plot_paired_in_vs_out(varn, df_, ax=None, 
                xvarname='instrip', order=[False, True],
                xticklabels=['outstrip', 'instrip']):
    if ax is None:
        fig, ax = pl.subplots()
    # plot
    is_bool = False
    if df_[xvarname].dtype == 'bool': 
        df_[xvarname] = df_[xvarname].astype(int)
        is_bool = True
        order = order.astype(int)
    sns.stripplot(data=df_, x=xvarname, y=varn, ax=ax, order=order,
                    c='w', s=3, jitter=False)
    # plot paired lines
    v1, v2 = order
    for f, fd in df_.groupby('filename'):
        ax.plot([0, 1],
            [fd[fd[xvarname]==v1][varn], fd[fd[xvarname]==v2][varn]],
                'w', lw=0.5)
    # adjust ticks
    ax.tick_params(which='both', axis='both', length=2, width=0.5, color='w',
                   direction='out', left=True)
    for pos in ['right', 'top', 'bottom']:
       ax.spines[pos].set_visible(False)
    # adjust labels
    ax.set_xlabel('')
    ax.set_xlim([-0.5, 1.5])
    ax.set_xticklabels(xticklabels) #['outstrip', 'instrip'])
    # stats
    a = df_[df_[xvarname]==v1][varn].values
    b = df_[df_[xvarname]==v2][varn].values
    pdf = pd.DataFrame({'a': a, 'b': b})
    T, pv = spstats.wilcoxon(pdf["a"], pdf["b"], nan_policy='omit')
    if pv>=0.05:
        star = 'n.s.'
    else:
        star = '**' if pv<0.01 else '*'
    ax.set_title(star, fontsize=8)

    if is_bool: 
        df_[xvarname] = df_[xvarname].astype(bool)

    return ax


def flip_data_for_abutting_hists(boutdf_filt,
        hue='instrip', hue_values=[True, False],
        vars_to_abs = ['speed', 'upwind_speed', 'crosswind_speed'],
        vars_to_flip = ['duration', 'speed_abs', 'path_length', 
                        'path_length_x', 'path_length_y', 'crosswind_dist_range', 
                        'upwind_speed_abs', 'upwind_dist_range', 'crosswind_speed_abs']):
    '''
    Flip inside bout metrics so that in and out histograms and plots can be 
    visualized abutting each other. Need to do this for plot_sorted_distn_with_hist()
    
    Returns boutdf with `_flipped` appended to flipped variables, and `abs_` prefixed to absolute valued variables. 
    '''
    v1, v2 = hue_values
    for varn in vars_to_abs:
        new_varn = '{}_abs'.format(varn)
        boutdf_filt[new_varn]=None
        boutdf_filt[new_varn] = boutdf_filt[varn].abs()

    for varn in vars_to_flip:
        new_varn = '{}_flipped'.format(varn)
        boutdf_filt[new_varn] = None
        boutdf_filt.loc[boutdf_filt[hue]==v1, new_varn]  = -1*boutdf_filt.loc[boutdf_filt[hue]==v1, varn].values
        boutdf_filt.loc[boutdf_filt[hue]==v2, new_varn]  = 1*boutdf_filt.loc[boutdf_filt[hue]==v2, varn].values
        boutdf_filt[new_varn] = boutdf_filt[new_varn].astype(float)
        
    return boutdf_filt

def plot_sorted_distn_with_hist(varn, boutdf_filt, estimator='median',
                             plot_bars=False, errorbar=('ci', 95), varn_type='flipped',
                            hue='instrip', hue_values=[True, False], hue_colors=['r', 'w']):
    '''
    Plot each individual fly's bouts as dot plot, abutting inside and outside.
    Below dot plot, show abutting histograms aligned.

    '''
    xlabels = {
        'crosswind_dist_range': 'crosswind range from edge (mm)',
        'duration':'duration',
        'path_length': 'path length (mm)',
        'path_length_x': 'x-path length (mm)',
        'path_length_y': 'y-path length(mm)',
        'speed': 'abs. speed (mm/s)',
        'upwind_speed': 'abs. upwind speed (mm/s)',
        'crosswind_speed': 'abs. crosswind speed (mm/s)',
        'upwind_dist_range': 'upwind range (mm)',
        'crosswind_dist_range': 'crosswind range (mm)'
    }
    if varn_type=='abs':
        plotvar = '{}_{}_flipped'.format(varn, varn_type) 
    else:
        plotvar = '{}_flipped'.format(varn)

    if estimator == 'median':
        mean_boutdf_filt2 = boutdf_filt.groupby(['filename', hue]).median().reset_index()
    elif estimator == 'mean':
        mean_boutdf_filt2 = boutdf_filt.groupby(['filename', hue]).mean().reset_index()
    else:
        mean_boutdf_filt2 = boutdf_filt.groupby(['filename', hue]).max().reset_index()
    # sort by OUT
    v1, v2 = hue_values
    sorted_by_xwind_out = mean_boutdf_filt2[mean_boutdf_filt2[hue]==v2]\
                            .sort_values(by=plotvar)['filename'].values

    # ---- plot ------
    xlabel = xlabels[varn]
    palette = dict((k, v) for k, v in zip(hue_values, hue_colors))
    nbins = 20 if varn=='path length' else 200 #if 
    fig = pl.figure(figsize=(6,8)) # sharex=True)
    gs = mpl.gridspec.GridSpec(3,1, figure=fig )
    # create sub plots as grid
    ax1 = fig.add_subplot(gs[0:2])
    if plot_bars:
        assert estimator in ['median', 'mean'], "wrong estimator: {}".format(estimator)
        sns.barplot(data=boutdf_filt, x=plotvar, ax=ax1, 
                y='filename', order=sorted_by_xwind_out,
                hue=hue, palette=palette, edgecolor='none', dodge=False,
               errcolor=[0.5]*3, errorbar=('ci', 95), estimator=estimator,
                saturation=1, width=0.8)
    else:
        sns.stripplot(data=boutdf_filt, x=plotvar, ax=ax1, 
                    y='filename', order=sorted_by_xwind_out,
                    hue=hue, palette=palette, edgecolor='none', dodge=False,
                    alpha=0.5)
    ax1.set_yticklabels([i if i%2==0 else '' for i, f in enumerate(sorted_by_xwind_out)])
    ax1.set_ylabel('traj. id')
    ax1.set_xlabel('')
    sns.move_legend(ax1, bbox_to_anchor=(1,1), loc='lower right', frameon=False)

    # histogram underneath
    vmin, vmax = boutdf_filt[plotvar].min(), boutdf_filt[plotvar].max()
    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    for col, cond in zip(hue_colors, hue_values):
        df_ = boutdf_filt[boutdf_filt[hue]==cond]
        vals = df_[plotvar].values
        bins = np.linspace(vmin, vmax, nbins) #vals.min(), vals.max(), 200)
        ax2.hist(vals, bins, facecolor=col, edgecolor='none',  alpha=0.9)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('number of bouts')

    for ax in fig.axes:
        ax.tick_params(which='both', axis='both', length=2, width=0.5, color='w',
                       direction='out', left=True, bottom=True)
        for pos in ['right', 'top']:
           ax.spines[pos].set_visible(False)
        #ax.set_box_aspect(1)

    sns.despine(offset=4)

    return fig



def plot_sorted_distn_with_hist_instrip(varn, boutdf_filt, estimator='median',
                             plot_bars=False, errorbar=('ci', 95),
                                instrip_palette={True: 'r', False: 'w'}):
    '''
    Plot each individual fly's bouts as dot plot, abutting inside and outside.
    Below dot plot, show abutting histograms aligned.

    '''
    xlabels = {
        'crosswind_dist_range': 'crosswind range from edge (mm)',
        'duration':'duration',
        'path_length': 'path length (mm)',
        'abs_upwind_speed': 'abs. upwind speed (mm/s)',
        'abs_crosswind_speed': 'abs. crosswind speed (mm/s)',
        'upwind_dist_range': 'upwind range (mm)',
        'crosswind_dist_range': 'crosswind range (mm)'
    }
    plotvar = '{}_flipped'.format(varn)

    if estimator == 'median':
        mean_boutdf_filt2 = boutdf_filt.groupby(['filename', 'instrip']).median().reset_index()
    elif estimator == 'mean':
        mean_boutdf_filt2 = boutdf_filt.groupby(['filename', 'instrip']).mean().reset_index()
    else:
        mean_boutdf_filt2 = boutdf_filt.groupby(['filename', 'instrip']).max().reset_index()
    # sort by OUT
    sorted_by_xwind_out = mean_boutdf_filt2[~mean_boutdf_filt2['instrip']]\
                            .sort_values(by=plotvar)['filename'].values

    # ---- plot ------
    xlabel = xlabels[varn]
    nbins = 20 if varn=='path length' else 200 #if 
    fig = pl.figure(figsize=(6,8)) # sharex=True)
    gs = mpl.gridspec.GridSpec(3,1, figure=fig )
    # create sub plots as grid
    ax1 = fig.add_subplot(gs[0:2])
    if plot_bars:
        assert estimator in ['median', 'mean'], "wrong estimator: {}".format(estimator)
        sns.barplot(data=boutdf_filt, x=plotvar, ax=ax1, 
                y='filename', order=sorted_by_xwind_out,
                hue='instrip', palette=instrip_palette, edgecolor='none', dodge=False,
               errcolor=[0.5]*3, errorbar=('ci', 95), estimator=estimator,
                saturation=1, width=0.8)
    else:
        sns.stripplot(data=boutdf_filt, x=plotvar, ax=ax1, 
                    y='filename', order=sorted_by_xwind_out,
                    hue='instrip', palette=instrip_palette, edgecolor='none', dodge=False,
                    alpha=0.5)
    ax1.set_yticklabels([i if i%2==0 else '' for i, f in enumerate(sorted_by_xwind_out)])
    ax1.set_ylabel('traj. id')
    ax1.set_xlabel('')
    sns.move_legend(ax1, bbox_to_anchor=(1,1), loc='lower right', frameon=False)

    # histogram underneath
    vmin, vmax = boutdf_filt[plotvar].min(), boutdf_filt[plotvar].max()
    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    for col, (cond, df_) in zip(['w','r'], boutdf_filt.groupby('instrip')):
        vals = df_[plotvar].values
        bins = np.linspace(vmin, vmax, nbins) #vals.min(), vals.max(), 200)
        ax2.hist(vals, bins, facecolor=col, edgecolor='none',  alpha=0.9)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('number of bouts')

    for ax in fig.axes:
        ax.tick_params(which='both', axis='both', length=2, width=0.5, color='w',
                       direction='out', left=True, bottom=True)
        for pos in ['right', 'top']:
           ax.spines[pos].set_visible(False)
        #ax.set_box_aspect(1)

    sns.despine(offset=4)

    return fig


def plot_one_flys_trials(df_, instrip_palette={True: 'r', False: 'w'},
            incl_logs=[], aspect_ratio=2, y_thresh=None, 
            sharex=False, sharey=True):

    ntrials = len(df_['trial_id'].unique())
    fig, axn = pl.subplots(1, ntrials, figsize=(ntrials*2.5, 5))
    if len(df_['trial_id'].unique())==1:
        sns.scatterplot(data=df_, x="ft_posx", y="ft_posy", 
                    hue='instrip', ax=axn,
                    s=.5, edgecolor='none', palette=instrip_palette)
        sns.scatterplot(data=df_[df_['led_on']], 
                    x="ft_posx", y="ft_posy", hue='led_on', ax=axn,
                    s=.5, edgecolor='none', palette={True: 'y'}, legend=False)
        axn.set_box_aspect(aspect_ratio)

        if y_thresh is not None:
            axn.axhline(y=y_thresh, linestyle=':', c='w', linewidth=0.5)

        if df_['filename'].unique()[0] in incl_logs:
            trial_id = '*{}'.format(df_['trial_id'].unique()[0])
        else:
            trial_id = df_['trial_id'].unique()[0]
        currcond = df_['condition'].unique()[0]
        plot_title = "{}{}{}".format(trial_id, '\n', currcond)
        axn.set_title(plot_title, fontsize=5, loc='left')
        axn.legend(bbox_to_anchor=(1,1), loc='upper left')
    else:
        for ai, (ax, (trial_id, tdf_)) in enumerate(zip(axn.flat, df_.groupby('trial_id'))):
            sns.scatterplot(data=tdf_, x="ft_posx", y="ft_posy", 
                    hue='instrip', ax=ax,
                    s=.5, edgecolor='none', palette=instrip_palette)
            sns.scatterplot(data=tdf_[tdf_['led_on']], 
                    x="ft_posx", y="ft_posy", hue='led_on', ax=ax,
                    s=.5, edgecolor='none', palette={True: 'y'}, legend=False)
            ax.set_box_aspect(aspect_ratio)

            if y_thresh is not None:
                ax.axhline(y=y_thresh, linestyle=':', c='w', linewidth=0.5)

            if tdf_['filename'].unique()[0] in incl_logs:
                trial_id = '*{}'.format(tdf_['trial_id'].unique()[0])
            else:
                trial_id = tdf_['trial_id'].unique()[0]

            currcond = tdf_['condition'].unique()[0]
            plot_title = "{}{}{}".format(trial_id, '\n', currcond)
            ax.set_title(plot_title, fontsize=6, loc='left')
            if ai == (ntrials-1):
                ax.legend(bbox_to_anchor=(1,1.1), loc='lower right')
            else:
                ax.legend_.remove()

    return fig



def plot_array_of_trajectories(trajdf, sorted_eff=[], nr=5, nc=7, 
                            aspect_ratio=0.5, sharey=True,
                            bool_colors=['r'], bool_vars=['instrip'], title='filename'):

    if len(sorted_eff)==0:
        sorted_eff = sorted(trajdf['filename'].unique(), key=util.natsort)


    maxy = trajdf['ft_posy'].max() if not sharey else 1600
     
    fig, axn = pl.subplots(nr, nc, figsize=(15,8), sharex=True)
    for fi, fn in enumerate(sorted_eff): #(fn, df_) in enumerate(etdf.groupby('filename')):
        if fi >= nr*nc:
            break
        ax=axn.flat[fi]
        df_ = trajdf[trajdf['filename']==fn].copy()
        #eff_ix = float(mean_tortdf[mean_tortdf['filename']==fn]['efficiency_ix'].unique())
        # PLOT
        plot_zeroed_trajectory(df_, ax=ax, traj_lw=1.5, odor_lw=1.0,
                                     strip_width=50, #params[fn]['strip_width'],
                                     strip_sep=1000, #) #params[fn]['strip_sep'])
                                bool_colors=bool_colors,
                                bool_vars=bool_vars)
        # legend
        ax.axis('off')
        if fi==0:
            leg_xpos=-150; leg_ypos=0; leg_scale=100
            butil.vertical_scalebar(ax, leg_xpos=leg_xpos, leg_ypos=leg_ypos)
        #ax.set_box_aspect(3)
        ax.set_xlim([-150, 150])
        ax.set_ylim([-100, maxy])
        ax.axis('off')
        ax.set_aspect(aspect_ratio)
        if title=='filename':
            ax_title = fn
        else:
            ax_title = fn.split('_')[0] # use datetime str 
        ax.set_title('{}:\n{}'.format(fi, ax_title), fontsize=6, loc='left')

    for ax in axn.flat[fi:]:
        ax.axis('off')
        #ax.set_aspect(0.5)
        
    pl.tight_layout()
    pl.subplots_adjust(top=0.85, hspace=0.4, wspace=0.5) #left=0.1, right=0.9, wspace=1, hspace=1, bottom=0.1, top=0.8)
    return fig


