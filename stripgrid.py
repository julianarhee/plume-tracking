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
    parser = argparse.ArgumentParser(description='Preprocessing steps.')
    parser.add_argument('-R', '--rootdir', type=str, 
        default='/Users/julianarhee/Library/CloudStorage/GoogleDrive-edge.tracking.ru@gmail.com/My Drive/Edge_Tracking/Data/jyr',
        help='Base name for directories. Example: /Full/path/to/data/folder/Edge_Tracking/Data')
    
    parser.add_argument('-D', '--dstdir', type=str, 
        default='/Users/julianarhee/Documents/rutalab/data/figures',
        help='Base name for directories. Example: /Full/path/to/data/folder/Edge_Tracking/Data')

    parser.add_argument('-E', '--experiment', type=str, default='',
        help='experiment name Example: pam-activation/20221031')
    parser.add_argument('-w', '--odor_width', type=float, default=10,
        help='odor width, mm (default: 10)')
    parser.add_argument('-s', '--grid_sep', type=float, default=200,
        help='grid separation, mm (default: 200)')
    parser.add_argument('--plotgroup', type=bool, default=False,
        help='plot all flies in 1 figure')
    parser.add_argument('-v', '--verbose', type=bool, default=False,
        help='verbose, print all statements')

    args = parser.parse_args()
    rootdir = args.rootdir
    experiment = args.experiment
    if '/' in experiment:
        experiment, session = experiment.split('/')
    else:
        session = args.session

    #datestr = args.datestr
    dstdir = args.dstdir 
    verbose = args.verbose
    odor_width = args.odor_width
    grid_sep = args.grid_sep

    plot_group = args.plotgroup
    start_at_odor = False

#    rootdir = '/Users/julianarhee/Library/CloudStorage/GoogleDrive-edge.tracking.ru@gmail.com\
#    /My Drive/Edge_Tracking/Data/jyr'
#    dstdir =  '/Users/julianarhee/Documents/rutalab/data/figures'
#
#    experiment = 'stripgrid'
#    session = '20221102'
#
#    odor_width=10
#    grid_sep=200
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

    log_files = sorted([k for k in glob.glob(os.path.join(src_dir, '*.log'))\
                    if 'odorpulse' not in k], key=util.natsort)
    print("Found {} tracking files.".format(len(log_files)))
    for fi, fpath in enumerate(log_files):
        dfn = os.path.split(fpath)[-1]
        print(fi, dfn)

    # load dataframes
    dlist = []
    for fpath in log_files:
        exp, date_str, fly_id, cond = butil.parse_info_from_file(fpath)
        if verbose:
            print(date_str, fly_id, cond)
        df_ = butil.load_dataframe(fpath, mfc_id=None, verbose=False, cond=cond)
        dlist.append(df_)
    df0 = pd.concat(dlist, axis=0)

    fly_ids = sorted(df0['fly_id'].unique(), key=util.natsort)
    #print(fly_ids)

    # #### get borders
    # get odor border for each fly
    odor_borders={}
    for trial_id, currdf in df0.groupby(['trial_id']):
        print(trial_id) #, ogrid)
        ogrid = butil.get_odor_grid(currdf, 
                                    odor_width=odor_width, grid_sep=grid_sep,
                                    use_crossings=True, verbose=False)
        odor_borders[trial_id] = list(ogrid.values())

    # ## plot traces
    hue_varname='instrip'
    palette={True: 'r', False: 'w'}
    odor_lc='lightgray'
    odor_lw=0.5

    if not plot_group:
        for trial_id, plotdf in df0.groupby('trial_id'):
            fig, ax = pl.subplots()
            title = os.path.splitext(os.path.split(fpath)[-1])[0] 
            if start_at_odor:
                start_ix = plotdf[plotdf['instrip']].iloc[0].name
            else:
                start_ix = plotdf.iloc[0].name
            butil.plot_trajectory(plotdf.loc[start_ix:], title=title, ax=ax)
            #sns.scatterplot(data=plotdf, x="ft_posx", y="ft_posy", hue=hue_varname,
            #            s=0.5, edgecolor='none', palette=palette, ax=ax)
            for obound in odor_borders[trial_id]:
                odor_xmin, odor_xmax = obound
                butil.plot_odor_corridor(ax, odor_xmin=odor_xmin, odor_xmax=odor_xmax)
            odor_start_ix = plotdf[plotdf['instrip']].iloc[0]['ft_posy']
            ax.axhline(y=odor_start_ix, color='w', lw=0.5, linestyle=':')
            #ax.plot(plotdf[plotdf['instrip']].iloc[0]['ft_posx'], 
            #        plotdf[plotdf['instrip']].iloc[0]['ft_posy'], '*', color='w')

            util.label_figure(fig, fig_id)
            figname = '{}'.format(trial_id)
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
        g = sns.FacetGrid(plotdf, col='trial_id', col_wrap=3, 
                        col_order=list(df0.groupby(['trial_id']).groups.keys()))
        g.map_dataframe(sns.scatterplot, x="ft_posx", y="ft_posy", hue=hue_varname,
                    s=0.5, edgecolor='none', palette=palette) #, palette=palette)
        # add odor corridor to facet grid
        for ai, (trial_id, currdf) in enumerate(plotdf.groupby(['trial_id'])):
            ax = g.axes[ai]
            for obound in odor_borders[trial_id]:
                odor_xmin, odor_xmax = obound
                butil.plot_odor_corridor(ax, odor_xmin=odor_xmin, odor_xmax=odor_xmax)

        util.label_figure(g.fig, fig_id)
        figname = 'trajectories_by_fly'
        pl.savefig(os.path.join(save_dir, '{}.png'.format(figname))) #, dpi=dpi)
        print(save_dir)



if __name__ == '__main__':
    main()