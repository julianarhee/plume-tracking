#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File           : stripgrid.py
Created        : 2022/11/7 11:53:30
Project        : /Users/julianarhee/Repositories/plume-tracking
Author         : Juliana Rhee
Last Modified  : 
'''
import os
import time
import sys
import scipy
import glob
import argparse
import importlib
import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
import pylab as pl
from datetime import datetime
# import some custom funcs
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import utils as util
import behavior as butil

util.set_sns_style(style='dark')


def main():
    parser = argparse.ArgumentParser(description='Plot individual datafile trajectories as subplots or set of single plots.')
    parser.add_argument('-R', '--rootdir', type=str, 
        default='/Users/julianarhee/Library/CloudStorage/GoogleDrive-edge.tracking.ru@gmail.com/My Drive/Edge_Tracking/Data/jyr',
        help='Base name for directories. Example: /Full/path/to/data/folder/Edge_Tracking/Data')
    
    parser.add_argument('-D', '--dstdir', type=str, 
        default='/Users/julianarhee/Documents/rutalab/data/figures',
        help='Base name for directories. Example: /Full/path/to/data/folder/Edge_Tracking/Data')

    parser.add_argument('-E', '--experiment', type=str, default='',
        help='experiment name Example: pam-activation/20221031')
    parser.add_argument('-w', '--strip_width', type=float, default=10,
        help='odor width, mm (default: 10)')
    parser.add_argument('-s', '--strip_sep', type=float, default=200,
        help='grid separation, mm (default: 200)')
    parser.add_argument('--plotgroup', type=bool, default=False,
        help='plot all flies in 1 figure')
    parser.add_argument('--plot_each_cond', type=bool, default=False,
        help='plot all flies for 1 cond in 1 figure')

    parser.add_argument('-v', '--verbose', type=bool, default=False,
        help='verbose, print all statements')
    parser.add_argument('-N', '--create_new', type=bool, default=False,
        help='Create new combined dataframe for all files')
    parser.add_argument('-x', '--remove_invalid', type=bool, default=True,
        help='Remove data with large skips')

    args = parser.parse_args()
    rootdir = args.rootdir
    experiment = args.experiment
    if '/' in experiment:
        experiment, session = experiment.split('/')
    else:
        session = '' #args.session

    #datestr = args.datestr
    dstdir = args.dstdir 
    verbose = args.verbose
    strip_width = args.strip_width
    strip_sep = args.strip_sep

    plot_group = args.plotgroup
    plot_each_cond = args.plot_each_cond
    start_at_odor = False
    create_new = args.create_new
    remove_invalid = args.remove_invalid

#    rootdir = '/Users/julianarhee/Library/CloudStorage/GoogleDrive-edge.tracking.ru@gmail.com\
#    /My Drive/Edge_Tracking/Data/jyr'
#    dstdir =  '/Users/julianarhee/Documents/rutalab/data/figures'
#
#    experiment = 'stripgrid'
#    session = '20221102'
#
#    strip_width=10
#    strip_sep=200
#    plot_group=False
#    verbose=False

    # create fig ID
    fig_id = os.path.join(rootdir.split(rootdir)[1], experiment, session)
    print("Fig ID: {}".format(fig_id))
    # Create output dir for figures
    src_dir = os.path.join(rootdir, experiment, session)
    save_dir = os.path.join(dstdir, '{}/{}'.format(experiment, session))
    save_dir = save_dir.replace(" ", "")

    print("Saving figures to:{}    {}".format('\n', save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

#    log_files = sorted([k for k in glob.glob(os.path.join(src_dir, '*.log'))\
#                    if 'odorpulse' not in k], key=util.natsort)
    log_files = butil.get_log_files(src_dir, verbose=True)

    # # Load dataframes
    df0 = butil.load_combined_df(os.path.join(src_dir, 'raw'), create_new=create_new, 
                                    remove_invalid=remove_invalid)
    condition_list = df0['condition'].unique()
    print("There are {} unique conditions:".format(len(condition_list)))
    for ci, cond in enumerate(condition_list):
        print(ci, cond)  

    fly_ids = sorted(df0['fly_id'].unique(), key=util.natsort)
    #print(fly_ids)

    # #### get borders
    # get odor border for each fly
    odor_borders={}
    for trial_id, currdf in df0.groupby('trial_id'):
        print(trial_id) #, ogrid)
        ogrid, oflag = butil.get_odor_grid(currdf, 
                                    strip_width=strip_width, strip_sep=strip_sep,
                                    use_crossings=True, verbose=False)
    
        odor_borders[trial_id] = list(ogrid.values())

    # ## plot traces
    hue_varname='instrip'
    palette={True: 'r', False: 'w'}
    odor_lc='lightgray'
    odor_lw=0.5

    if not plot_group:
        for fly_id, df_ in df0.groupby('fly_id'):
            condition_list = df_['condition'].unique()
            psize = 3.5
            fig, axn = pl.subplots(1, len(condition_list), figsize=(len(condition_list)*psize, psize),
                                    sharex=True, sharey=True)
            for ci, (cond, plotdf) in enumerate(df_.groupby('condition')):
                if len(condition_list)>1:
                    ax=axn[ci]
                else:
                    ax=axn
                fname = plotdf['filename'].unique()[0]
                if start_at_odor:
                    start_ix = plotdf[plotdf['instrip']].iloc[0].name
                else:
                    start_ix = plotdf.iloc[0].name
                oparams = butil.get_odor_params(plotdf, strip_width=strip_width, is_grid=True)
                butil.plot_trajectory(plotdf.loc[start_ix:], title=fname, ax=ax, center=True,
                                odor_bounds=oparams['odor_boundary'])
                ax.set_title(cond, fontsize=6)
                if ci==0:
                    ax.legend_.remove()
                else:
                    ax.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=6, frameon=False)
            pl.subplots_adjust(wspace=0.5, top=0.8, bottom=0.2)
            util.label_figure(fig, fig_id)
            figname = '{}'.format(fly_id)
            pl.savefig(os.path.join(save_dir, '{}.png'.format(figname))) #, dpi=dpi)
        print(save_dir)
    else:
    #%%
        if start_at_odor:
            d_list=[]
            for trial_id, df_ in df0.groupby('trial_id'):
                start_ix = df_[df_['instrip']].iloc[0].name
                d_list.append(df_.loc[start_ix:])
            plotdf = pd.concat(d_list)
        else:
            plotdf = df0.copy()
        # group plot
        condition_list = plotdf['condition'].unique()
        if len(condition_list)>1:
            if plot_each_cond:
                for cond, df_ in plotdf.groupby('condition'):
                    fig = butil.plot_all_flies(df_, hue_varname=hue_varname,
                                        palette=palette, strip_width=strip_width, col_wrap=4)
                    # save
                    figname = 'traj-all_{}'.format(cond)
                    util.label_figure(fig, fig_id)
                    pl.savefig(os.path.join(save_dir, '{}.png'.format(figname))) #, dpi=dpi)
                    print(save_dir, figname)
            else:
                fig = butil.plot_fly_by_condition(plotdf, strip_width=strip_width)
                # save
                figname = 'traj-all-by-cond'
                util.label_figure(fig, fig_id)
                pl.savefig(os.path.join(save_dir, '{}.png'.format(figname))) #, dpi=dpi)
                print(save_dir, figname)

        else:
            g = sns.FacetGrid(plotdf, col='trial_id', col_wrap=4,
                            col_order=list(plotdf.groupby('trial_id').groups.keys()))
            g.map_dataframe(sns.scatterplot, x="ft_posx", y="ft_posy", hue=hue_varname,
                        s=0.5, edgecolor='none', palette=palette) #, palette=palette)
            g.set_titles(row_template = '{row_name}', col_template = '{col_name}', size=6)

            # add odor corridor to facet grid
            for ai, (trial_id, currdf) in enumerate(plotdf.groupby('trial_id')):
                ax = g.axes[ai]
                for obound in odor_borders[trial_id]:
                    odor_xmin, odor_xmax = obound
                    butil.plot_odor_corridor(ax, odor_xmin=odor_xmin, odor_xmax=odor_xmax)
            figname = 'traj-all'
            util.label_figure(g.fig, fig_id)
            pl.savefig(os.path.join(save_dir, '{}.png'.format(figname))) #, dpi=dpi)
            print(save_dir, figname)



if __name__ == '__main__':
    main()
