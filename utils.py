#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2022/09/21 10:00:55
@Author  :   julianarhee 
@Contact :   juliana.rhee@gmail.com
 
'''
#%%
import os
import re
import seaborn as sns
import numpy as np
import matplotlib as mpl
import pylab as pl

# General
# Abstract struct class
class DictStruct:
    def __init__ (self, *argv, **argd):
        if len(argd):
            # Update by dictionary
            self.__dict__.update (argd)
        else:
            # Update by position
            attrs = filter (lambda x: x[0:2] != "__", dir(self))
            for n in range(len(argv)):
                setattr(self, attrs[n], argv[n])

class struct():
    pass

# ----------------------------------------------------------------------
# General functions
# ----------------------------------------------------------------------
def atoi(text):
    return int(text) if text.isdigit() else text

def natsort(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def isnumber(n):
    try:
        float(n)   # Type-casting the string to `float`.
                   # If string is not a valid `float`,
                   # it'll raise `ValueError` exception
    except ValueError:
        return False
    except TypeError:
        return False

    return True

def make_ordinal(n):
    '''
    Convert an integer into its ordinal representation::

        make_ordinal(0)   => '0th'
        make_ordinal(3)   => '3rd'
        make_ordinal(122) => '122nd'
        make_ordinal(213) => '213th'
    from: @FlorianBrucker:
    https://stackoverflow.com/questions/9647202/ordinal-numbers-replacement
    '''
    n = int(n)
    suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    return str(n) + suffix


# ----------------------------------------------------------------------
# Calculation
# ----------------------------------------------------------------------
def rotate_coordinates(x, y, theta):
    '''
    Rotate coordinates by theta (in radians).

    Arguments:
        x (np.array, list) : setof x-coords
        y (np.array, list) : set of y-coords
        theta (float) : theta in radians 

    Returns:
        rotx, roty : transformed coords
    '''
    rotx = x*np.cos(theta) + y*np.sin(theta)
    roty = -x*np.sin(theta) + y*np.cos(theta)
    return rotx, roty

def fliplr_coordinates(x, y):
    return x*-1, y

# ----------------------------------------------------------------------
# Data processing 
# ----------------------------------------------------------------------

def temporal_downsample(trace, windowsz):
    tmp1=np.concatenate((np.ones(windowsz)*trace.values[0], trace, np.ones(windowsz)*trace.values[-1]),0)
    tmp2=np.convolve(tmp1, np.ones(windowsz)/windowsz, 'same')
    tmp2=tmp2[windowsz:-windowsz]

    return tmp2

def smooth_timecourse(in_trace, win_size=41):
    '''
    Don't Use this one
    '''
    #smooth trace
    win_half = int(round(win_size/2))
    trace_pad = np.pad(in_trace, ((win_half, win_half)), 'reflect') # 'symmetric') #'edge')

    smooth_trace = np.convolve(trace_pad, np.ones((win_size,))*(1/float(win_size)),'valid')
    
    return smooth_trace


# ----------------------------------------------------------------------
# Visualization 
# ----------------------------------------------------------------------
def label_figure(fig, fig_id):
    fig.text(0.01, 0.97, fig_id, fontsize=8)


def set_sns_style(style='dark'):
    font_styles = {
                    'axes.labelsize': 8, # x and y labels
                    'axes.titlesize': 10, # axis title size
                    'figure.titlesize': 10,
                    'xtick.labelsize': 7, # fontsize of tick labels
                    'ytick.labelsize': 7,  
                    'legend.fontsize': 6,
                    'legend.title_fontsize': 7
        }
    if style=='dark':
        custom_style = {
                    'axes.labelcolor': 'white',
                    'axes.edgecolor': 'white',
                    'grid.color': 'gray',
                    'xtick.color': 'white',
                    'ytick.color': 'white',
                    'text.color': 'white',
                    'axes.facecolor': 'black',
                    'axes.grid': False,
                    'figure.facecolor': 'black'}
        custom_style.update(font_styles)

#        pl.rcParams['figure.facecolor'] = 'black'
#        pl.rcParams['axes.facecolor'] = 'black'
        sns.set_style("dark", rc=custom_style)

    pl.rcParams['savefig.dpi'] = 400

def add_colorwheel(fig, cmap='hsv', axes=[0.7, 0.7, 0.3, 0.3], fontsize=7,
                   theta_range=[-np.pi, np.pi], deg2plot=None, theta_units='rad'):
    display_axes = fig.add_axes(axes, projection='polar')
    # display_axes._direction = max(theta_range) #2*np.pi ## This is a nasty hack - using the hidden field to 
                                      ## multiply the values such that 1 become 2*pi
                                      ## this field is supposed to take values 1 or -1 only!!
    #norm = mpl.colors.Normalize(0.0, 2*np.pi)
    if theta_units=='deg':
        theta_range = np.deg2rad(theta_range)

    norm = mpl.colors.Normalize(theta_range[0], theta_range[1])

    # Plot the colorbar onto the polar axis
    # note - use orientation horizontal so that the gradient goes around
    # the wheel rather than centre out
    quant_steps = 2056
    cb = mpl.colorbar.ColorbarBase(display_axes, cmap=mpl.cm.get_cmap(cmap, quant_steps),
                                       norm=norm, orientation='horizontal')
    # aesthetics - get rid of border and axis labels                                   
    cb.outline.set_visible(False)                                 
    #display_axes.set_axis_off()
    #display_axes.set_rlim([-1,1])
    if deg2plot is not None:
        display_axes.plot([0, deg2plot], [0, 1], 'k')
    
    #display_axes.set_theta_zero_location('W')
    return display_axes

def add_colorwheel(fig, cmap='hsv', axes=[0.7, 0.7, 0.3, 0.3], 
                   theta_range=[-np.pi, np.pi], deg2plot=None):
    display_axes = fig.add_axes(axes, projection='polar')
    #display_axes._direction = max(theta_range) #2*np.pi ## This is a nasty hack - using the hidden field to 
                                      ## multiply the values such that 1 become 2*pi
                                      ## this field is supposed to take values 1 or -1 only!!
    #norm = mpl.colors.Normalize(0.0, 2*np.pi)
    norm = mpl.colors.Normalize(theta_range[0], theta_range[1])

    # Plot the colorbar onto the polar axis
    # note - use orientation horizontal so that the gradient goes around
    # the wheel rather than centre out
    quant_steps = 2056
    cb = mpl.colorbar.ColorbarBase(display_axes, cmap=mpl.cm.get_cmap(cmap, quant_steps),
                                       norm=norm, orientation='horizontal')
    # aesthetics - get rid of border and axis labels                                   
    cb.outline.set_visible(False)                                 
    #display_axes.set_axis_off()
    #display_axes.set_rlim([-1,1])
    if deg2plot is not None:
        display_axes.plot([0, deg2plot], [0, 1], 'k')
    
    display_axes.set_theta_zero_location('N')
    display_axes.set_theta_direction(-1)  # theta increasing clockwise

    return display_axes

def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True, facecolor=[0.7]*3):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    # x = (x+np.pi) % (2*np.pi) - np.pi
    # Force bins to partition entire circle
    if not gaps:
        #bins = np.linspace(-np.pi, np.pi, num=bins+1)
        bins = np.linspace(0, 2*np.pi, num=bins+1)
    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)
    # Compute width of each bin
    widths = np.diff(bins)
    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n
    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, width=widths, #align='edge', 
                     edgecolor='w', fill=True, linewidth=0.5, facecolor=facecolor,
                    alpha=0.5)
    # Set the direction of the zero angle
    ax.set_theta_offset(offset)
    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)  # theta increasing clockwise
     
    return n, bins, patches


