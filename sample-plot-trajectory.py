#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   sample-plot-trajectory.py
@Time    :   2022/10/28 17:32:55
@Author  :   julianarhee 
@Contact :   juliana.rhee@gmail.com
 
'''
import os
import glob
import sys

import argparse

import pandas as pd
import numpy as np
# plotting modules
import matplotlib as mpl
#mpl.use('nbagg')
import seaborn as sns
import pylab as pl
# import some custom funcs
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import utils as util
import behavior as butil

util.set_sns_style(style='dark') # plotting settings i like for Nbs

def select_logfile(experiment, datestr, user_input=False, 
        rootdir='/Users/julianarhee/Library/CloudStorage/GoogleDrive-edge.tracking.ru@gmail.com/My Drive/Edge_Tracking/Data'):

    src_dir = os.path.join(rootdir, experiment)

    # List all files and have user select index of file to process
    if user_input:
        # Get a list of all the data files
        log_files = sorted([k for k in glob.glob(os.path.join(src_dir, \
                            '*.log')) if 'lossed tracking' not in k], \
                            key=util.natsort) 
        print("Found {} tracking files.".format(len(log_files)))
        for i, fn in enumerate(log_files):
            print("{}: {}".format(i, os.path.split(fn)[-1]))
        file_ix = int(input('Select IX of file to run: ')) # 12 # select a file
        fpath = log_files[file_ix]
        print("Selected: {}".format(fpath))
    else:
        # Find datafile with datestr in it (there should only be one)
        log_files = sorted([k for k in glob.glob(os.path.join(src_dir, '{}*.log'.format(datestr)))], key=util.natsort)
        assert len(log_files)==1, "No unique file found in dir {}{} ...".format('\n', src_dir)
        fpath = log_files[0]
    
    return fpath

#%%
def main():
    parser = argparse.ArgumentParser(description='Quickly visualize fly trajectory')
    parser.add_argument('-R', '--rootdir', type=str, 
        default='/Users/julianarhee/Library/CloudStorage/GoogleDrive-edge.tracking.ru@gmail.com/My Drive/Edge_Tracking/Data/jyr',
        help='Base name for directories. Example: /Full/path/to/data/folder/Edge_Tracking/Data')
    parser.add_argument('-E', '--experiment', type=str, default='',
        help='experiment name Example: pam-activation')
    parser.add_argument('-d', '--datestr', default='MMDDYYYY-HHmm', action='store',
        help='MMDDYYYY-HHmm.log format')

    parser.add_argument('-w', '--strip_width', type=float, default=50,
        help='odor width, mm (default: 50)')
    parser.add_argument('-s', '--strip_sep', type=float, default=500,
        help='grid separation, mm (default: 500)')
    parser.add_argument('-v', '--verbose', default=False,
        action='store_true', help='verbose, print all statements')
    parser.add_argument('-O', '--start_at_odor', default=False,
        action='store_true', help='plot from odor onset')
    parser.add_argument('-z', '--zero_odor_start', default=False,
        action='store_true', help='zero is odor onset')
    parser.add_argument('-S', '--save', default=False,
        action='store_true', help='save to tmp folder')
    parser.add_argument('-f', '--origlr', default=False,
        action='store_true', help='use raw l/r, instead of flipping')
    parser.add_argument('-x', '--keep_all_frames', default=False,
        action='store_true', help='do not remove invalid skips')

    parser.add_argument('-m', '--markersize', default=0.5, type=float,
        action='store', help='markersize (default=0.5)')
    parser.add_argument('-H', '--hue_varname', default='instrip',
        action='store', help='hue variable (default=instrip)')
    parser.add_argument('-y', '--y_thresh', default=None,
        action='store', help='plot horizontal line at y_thresh (default=None)')

    args = parser.parse_args()
    rootdir = args.rootdir
    experiment = args.experiment
    datestr = args.datestr

    strip_width = args.strip_width
    strip_sep = args.strip_sep
    user_input = False
    parse_filename = False
    
    start_at_odor = args.start_at_odor
    zero_odor_start = args.zero_odor_start
    save = args.save
    markersize = args.markersize
    fliplr = not(args.origlr)
    remove_invalid = not(args.keep_all_frames)
    plot_errors = args.keep_all_frames # plot errors of keeping all

    hue_varname = args.hue_varname
    y_thresh = args.y_thresh
    plot_threshold = y_thresh is not None

    fpath = select_logfile(experiment, datestr, user_input=user_input, rootdir=rootdir)
    #fig, ax = pl.subplots()

    # load and process the csv data  
    df0 = butil.load_dataframe(fpath, verbose=False, #cond=None, 
                parse_filename=parse_filename, fliplr=fliplr, 
                remove_invalid=remove_invalid, plot_errors=plot_errors)
    fly_id=None
    if parse_filename:
        # try to parse experiment details from the filename
        exp, datestr, fid, cond = parse_info_from_filename(fpath)
        print('Experiment: {}{}Fly ID: {}{}Condition: {}'.format(exp, '\n', fid, '\n', cond))
        fly_id = df0['fly_id'].unique()[0]
    else:
        fly_id = os.path.split(fpath)[-1]
        df0['fly_id'] = fly_id

    df0 = butil.process_df(df0)
    strip_borders = butil.find_strip_borders(df0, 
                            strip_width=strip_width, strip_sep=strip_sep, 
                            entry_ix=None)
    title = os.path.splitext(os.path.split(fpath)[-1])[0]

    if hue_varname == 'odor_percent':
        values = df0[hue_varname].unique()
        colors = sns.color_palette('colorblind', n_colors=len(values))
        palette = dict((k, v) for k, v in zip(values, colors))
    else:
        palette = {True: 'r', False: 'w'} #hue_varname=='instrip'


    fig, ax = pl.subplots()
    ax = butil.plot_trajectory(df0, odor_bounds=strip_borders, title=title, ax=ax,
            start_at_odor=start_at_odor, zero_odor_start=zero_odor_start,
            markersize=markersize, hue_varname=hue_varname, palette=palette)

#    ax = butil.plot_trajectory_from_file(fpath, parse_filename=parse_filename, 
#                strip_width=strip_width, strip_sep=strip_sep, ax=None,
#                start_at_odor=start_at_odor, zero_odor_start=zero_odor_start,
#                markersize=markersize, fliplr=fliplr, 
#                remove_invalid=remove_invalid, plot_errors=plot_errors,
#                hue_varname=hue_varname, palette=palette)
   
    if plot_threshold:
        y_thresh = float(y_thresh)
#        if zero_odor_start:
#            odor_ix = df0[df0['instrip']].iloc[0].name
#            offset_x = df0.loc[odor_ix]['ft_posx']
#            offset_y = df0.loc[odor_ix]['ft_posy']
#            y_thresh = y_thresh - offset_y
        ax.axhline(y=y_thresh, lw=0.5, linestyle=':', c='w')
        if df0['ft_posy'].max() > y_thresh*2:
            ax.axhline(y=y_thresh*2, lw=0.5, linestyle=':', c='w')

    ax.set_aspect(1.0)
    # label figure and save
    fig_id = '{}: {}'.format(experiment, datestr)
    util.label_figure(ax.figure, fig_id)

    if save:
        save_dir = os.path.join(rootdir.split('/logfiles')[0], 'figures')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        _, fbase = os.path.split(fpath)
        figname = '{}.png'.format(os.path.splitext(fbase)[0])
        pl.savefig(os.path.join(save_dir, figname)) #, dpi=dpi)
        print("saved: {}".format(os.path.join(save_dir, figname)))

    pl.show()

#%%
if __name__ == '__main__':

    main()




