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

def plot_trajectory_from_file(fpath, parse_info=False,
            odor_width=10, grid_sep=200):
    # load and process the csv data  
    df0 = butil.load_dataframe(fpath, mfc_id=None, verbose=False, cond=None, 
                parse_info=False)
    fly_id=None
    if parse_info:
        # try to parse experiment details from the filename
        exp, fid, cond = butil.parse_info_from_file(fpath)
        print('Experiment: {}{}Fly ID: {}{}Condition: {}'.format(exp, '\n', fid, '\n', cond))
        fly_id = df0['fly_id'].unique()[0]

    # get experimentally determined odor boundaries:
    ogrid = butil.get_odor_grid(df0, odor_width=odor_width, grid_sep=grid_sep,
                                use_crossings=True, verbose=False, )
    #(odor_xmin, odor_xmax), = ogrid.values()
    odor_bounds = list(ogrid.values())

    # Set some plotting params 
    hue_varname='instrip'
    palette={True: 'r', False: 'w'}
    start_at_odor = True
    odor_width=50
    odor_lc='lightgray'
    odor_lw=0.5
    # ---------------------------------------------------------------------
    fig, ax = pl.subplots()
    sns.scatterplot(data=df0, x="ft_posx", y="ft_posy", ax=ax, 
                    hue=hue_varname, s=0.5, edgecolor='none', palette=palette)
    for (odor_xmin, odor_xmax) in odor_bounds:
        butil.plot_odor_corridor(ax, odor_xmin=odor_xmin, odor_xmax=odor_xmax)

    ax.legend(bbox_to_anchor=(1,1), loc='upper left', title=hue_varname)
    fname = os.path.splitext(os.path.split(fpath)[-1])[0]

    ax.set_title(fname)
    pl.subplots_adjust(left=0.2, right=0.8)

    # Center corridor
    xmax = np.ceil(df0['ft_posx'].abs().max())
    ax.set_xlim([-xmax-10, xmax+10])

    return fig

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
    parser = argparse.ArgumentParser(description='Preprocessing steps.')
    parser.add_argument('-R', '--rootdir', type=str, 
        default='/Users/julianarhee/Library/CloudStorage/GoogleDrive-edge.tracking.ru@gmail.com/My Drive/Edge_Tracking/Data/jyr',
        help='Base name for directories. Example: /Full/path/to/data/folder/Edge_Tracking/Data')
    parser.add_argument('-E', '--experiment', type=str, default='',
        help='experiment name Example: pam-activation')
    parser.add_argument('-d', '--datestr', default='MMDDYYYY-HHmm', action='store',
        help='MMDDYYYY-HHmm.log format')

    parser.add_argument('-w', '--odor_width', type=float, default=10,
        help='odor width, mm (default: 10)')
    parser.add_argument('-s', '--grid_sep', type=float, default=200,
        help='grid separation, mm (default: 200)')
    parser.add_argument('-v', '--verbose', type=bool, default=False,
        help='verbose, print all statements')

    args = parser.parse_args()
    rootdir = args.rootdir
    experiment = args.experiment
    datestr = args.datestr

    odor_width = args.odor_width
    grid_sep = args.grid_sep
    user_input = False
    parse_info = False

    fpath = select_logfile(experiment, datestr, user_input=user_input, rootdir=rootdir)
    fig = plot_trajectory_from_file(fpath, parse_info=parse_info, 
                odor_width=odor_width, grid_sep=grid_sep)

    # label figure and save
    fig_id = '{}: {}'.format(experiment, datestr)
    util.label_figure(fig, fig_id)

    pl.show()

    #pl.savefig(os.path.join(save_dir, '{}.png'.format(figname))) #, dpi=dpi)
    #print(os.path.join(save_dir, '{}.png'.format(figname)))

#%%
if __name__ == '__main__':

    main()




