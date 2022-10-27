#!/usr/bin/env python
# coding: utf-8

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

def plot_trajectory(experiment, fly, rootdir='/Users/julianarhee/Library/CloudStorage/GoogleDrive-edge.tracking.ru@gmail.com/My Drive/Edge_Tracking/Data'):

    src_dir = os.path.join(rootdir, experiment)
    # Create a label/ID for figures so you always know where the plotted data came from
    fig_id = '{}: {}'.format(src_dir.split('/My Drive')[1], fly)


    # get a list of all the data files
    try:
        log_files = sorted([k for k in glob.glob(os.path.join(src_dir, '*{}*.log'.format(fly)))\
                    if 'lossed tracking' not in k], key=util.natsort)
        assert len(log_files)>1, "Fly not found, trying with diff. case"
    except Exception as e:
        log_files = sorted([k for k in glob.glob(os.path.join(src_dir, '*.log'))\
                    if 'lossed tracking' not in k], key=util.natsort)
        print("Unable to find your fly file...")
        for fpath in log_files:
            print('...{}'.format(os.path.split(fpath)[-1]))
 
    print("Found {} tracking files.".format(len(log_files)))
    for i, fn in enumerate(log_files):
        print("{}: {}".format(i, os.path.split(fn)[-1]))

    file_ix = int(input('Select IX of file to run: ')) # 12 # select a file
    fpath = log_files[file_ix]
    print("Selected: {}".format(fpath))

    # try to parse experiment details from the filename
    exp, fid, cond = butil.parse_info_from_file(fpath)
    # load and process the csv data  
    df0 = butil.load_dataframe(fpath, mfc_id=None, verbose=False, cond=None)
    print('Experiment: {}{}Fly ID: {}{}Condition: {}'.format(exp, '\n', fid, '\n', cond))
    fly_id = df0['fly_id'].unique()[0]

    # get experimentally determined odor boundaries:
    odor_width=50
    grid_sep=2000 # only matters for 2p
    ogrid = butil.get_odor_grid(df0, 
                                odor_width=odor_width, grid_sep=grid_sep,
                                use_crossings=True, verbose=False)
    (odor_xmin, odor_xmax), = ogrid.values()

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
    butil.plot_odor_corridor(ax, odor_xmin=odor_xmin, odor_xmax=odor_xmax)
    ax.legend(bbox_to_anchor=(1,1), loc='upper left', title=hue_varname)
    ax.set_title(fly_id)
    pl.subplots_adjust(left=0.2, right=0.8)
    # Center corridor
    xmax = np.ceil(df0['ft_posx'].abs().max())
    ax.set_xlim([-xmax-10, xmax+10])
    # label figure and save
    util.label_figure(fig, fig_id)
    figname = 'trajectory_{}'.format(fly_id)

    pl.show()

    #pl.savefig(os.path.join(save_dir, '{}.png'.format(figname))) #, dpi=dpi)
    #print(os.path.join(save_dir, '{}.png'.format(figname)))



#%%
def main():
    parser = argparse.ArgumentParser(description='Preprocessing steps.')
    parser.add_argument('-R', '--rootdir', type=str, 
        default='/Users/julianarhee/Library/CloudStorage/GoogleDrive-edge.tracking.ru@gmail.com/My Drive/Edge_Tracking/Data',
        help='Base name for directories. Example: /Full/path/to/data/folder/Edge_Tracking/Data')
    parser.add_argument('-E', '--experiment', type=str, default='',
        help='experiment name Example: pam-activation')
    parser.add_argument('-f', '--fly', default='', action='store',
        help='fly id (ex: "fly9")')

    args = parser.parse_args()
    rootdir = args.rootdir
    experiment = args.experiment
    fly = args.fly
    
    plot_trajectory(experiment, fly)

    
#%%
if __name__ == '__main__':

    main()




