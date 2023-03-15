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
import platform
import seaborn as sns
import numpy as np
import matplotlib as mpl
import pylab as pl

# General
def get_os():
    return platform.system()

def get_rootdir():
    if get_os() == 'Linux':
        rootdir = '/home/julianarhee/edgetracking-googledrive/Edge_Tracking/Data'
    elif get_os() == 'Darwin':
        rootdir = '/Users/julianarhee/Library/CloudStorage/GoogleDrive-edge.tracking.ru@gmail.com/My Drive/Edge_Tracking/Data'
    else:
        rootdir=None
        print("Unknown os: {}".format(get_os()))
    return rootdir    

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

def convert_range(oldval, newmin=None, newmax=None, oldmax=None, oldmin=None):
    if oldmax is None: #and len(oldval)>1:
        oldmax = np.nanmax(oldval)
    if oldmin is None: # and len(oldval)>1:
        oldmin = np.nanmin(oldval)

    oldrange = (oldmax - oldmin)
    newrange = (newmax - newmin)
    newval = (((oldval - oldmin) * newrange) / oldrange) + newmin
    return newval

def flatten(l):
    return [item for sublist in l for item in sublist]

# ----------------------------------------------------------------------
# Calculation
# ----------------------------------------------------------------------
def calculate_tortuosity(traj_coords):
    '''
    Calculates tortuosity of path as ratio of path length and dist between start and end points.
    1 is low-tortuosity (perfect), and inf is high tortuosity.
    '''
    # distance between start and end points
    maxdist = euclidean(traj_coords[0], traj_coords[-1])
    # total path length
    pathlength = path_length(traj_coords)
    # tortuosity as ratio
    tortuosity = pathlength/maxdist
    
    return tortuosity

def path_length(traj, axis='both'): #df, xvar='ft_posx', yvar='ft_posy'):

    if axis=='x':
        pl = np.sqrt(np.diff(traj[:, 0])**2).sum()
    elif axis=='y':
        pl = np.sqrt(np.diff(traj[:, 1])**2).sum()
    else:
        #dists = np.linalg.norm(df[[xvar, yvar]].diff(axis=0).dropna(), axis=1)
        pl = np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()

    return pl


def euclidean(point1, point2):
    x0, y0 = point1
    x1, y1 = point2
    return np.sqrt( (x1-x0)**2 + (y1-y0)**2)


def signed_angle_from_target(b_, target_angle=-np.pi, varname='heading'):
    """
    Calculate signed angle from left edge, unit vector (-1, 0), -np.pi.
    Assumes values go from -pi, pi. Make jumps from 180 to -180 continuous with np.rad2deg(angbetween) % 360.
    Use utils.circular_dist() for shortest angle, constrained [0, pi).
    """
    vals = b_[varname].copy().values
    # If subtracting -pi/2, values in lower-right quadrant should be flipped
    # to make -90 (or 270) to +90 map continuously as -90 to -270 (bottom half) 
    flip_discontinous_quadrant = np.where(b_[varname]>np.pi/2)[0]
    # flip the values
    flipped_vals = -np.pi - (np.pi-vals[flip_discontinous_quadrant])
    # 
    vals[flip_discontinous_quadrant] = flipped_vals
    angbetween = (-np.pi/2 - vals ) #% np.pi/2
    return angbetween

def circular_mean(values):
    # calculate mean fo cos/sin components separately, input angles in [0, 2pi).
    mean_cos = np.mean(np.cos(values))
    mean_sin = np.mean(np.sin(values))
    x = np.arctan2(mean_sin, mean_cos)

    return x

def circular_dist(angle1, angle2):
    ''' takes smaller of the arc lengths, lies in [0, pi)'''
    return np.pi - abs(np.pi - abs(angle1-angle2))


def circular_median(values):
    '''
    Median as min. distance to all other observations in sample.

    Arguments:
        values -- _description_

    Returns:
        _description_
    '''
    dist = [sum([circular_dist(mid_angle, angle) for angle in values])\
                for mid_angle in values]
    if not len(values) % 2:
        sorted_dist = np.argsort(dist)
        mid_angles = values[sorted_dist[0:2]]
        return np.mean(mid_angles)
    else:
        return values[np.agmin(dist)]



def get_CoM(df_, xvar='ft_posx', yvar='ft_posy'):
    '''
    Calculate center of mass from coords x0, y0 in dataframe df_
    '''
    x = df_[xvar].values
    y = df_[yvar].values
    m=np.ones(df_[xvar].shape)

    #m = np.ones(x.shape)
    cgx = np.sum(x*m)/np.sum(m)
    cgy = np.sum(y*m)/np.sum(m)
    
    return cgx, cgy

def unwrap_and_constrain_angles(phases):
    '''
    Equivalent to unwrap, then constraining within (-pi, pi)
    ft_heading is [0, 2*np.pi), so must account for diffs that wrap bw 0 and 2pi 
    and make continuous.
    default is pi -- unwraps p s.t. adjacent diffs never greater than pi
    
    Arguments:
         phases (should be in radians)
    '''
    p = (phases + np.pi) % (2 * np.pi ) - np.pi

    return p

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
def label_figure(fig, fig_id, x=0.01, y=0.98):
    fig.text(x, y, fig_id, fontsize=8)


def set_sns_style(style='dark', min_fontsize=6):
    font_styles = {
                    'axes.labelsize': min_fontsize+1, # x and y labels
                    'axes.titlesize': min_fontsize+1, # axis title size
                    'figure.titlesize': min_fontsize+4,
                    'xtick.labelsize': min_fontsize, # fontsize of tick labels
                    'ytick.labelsize': min_fontsize,  
                    'legend.fontsize': min_fontsize,
                    'legend.title_fontsize': min_fontsize+1
        }
    for k, v in font_styles.items():
        pl.rcParams[k] = v

    pl.rcParams['axes.linewidth'] = 0.5

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
    pl.rcParams['figure.figsize'] = [6,4]


#def add_colorwheel(fig, cmap='hsv', axes=[0.8, 0.8, 0.1, 0.1], fontsize=7,
#                   theta_range=[-np.pi, np.pi], deg2plot=None, theta_units='rad'):
#    display_axes = fig.add_axes(axes, projection='polar')
#    # display_axes._direction = max(theta_range) #2*np.pi ## This is a nasty hack - using the hidden field to 
#                                      ## multiply the values such that 1 become 2*pi
#                                      ## this field is supposed to take values 1 or -1 only!!
#    #norm = mpl.colors.Normalize(0.0, 2*np.pi)
#    if theta_units=='deg':
#        theta_range = np.deg2rad(theta_range)
#
#    norm = mpl.colors.Normalize(theta_range[0], theta_range[1])
#
#    # Plot the colorbar onto the polar axis
#    # note - use orientation horizontal so that the gradient goes around
#    # the wheel rather than centre out
#    quant_steps = 2056
#    cb = mpl.colorbar.ColorbarBase(display_axes, cmap=mpl.cm.get_cmap(cmap, quant_steps),
#                                       norm=norm, orientation='horizontal')
#    # aesthetics - get rid of border and axis labels                                   
#    cb.outline.set_visible(False)                                 
#    #display_axes.set_axis_off()
#    #display_axes.set_rlim([-1,1])
#    if deg2plot is not None:
#        display_axes.plot([0, deg2plot], [0, 1], 'k')
#    
#    #display_axes.set_theta_zero_location('W')
#    return display_axes
#
def add_colorwheel(fig, cmap='hsv', axes=[0.8, 0.8, 0.1, 0.1], 
                   theta_range=[-np.pi, np.pi], deg2plot=None):

    '''
    Assumes values go from 0-->180, -180-->0. (radians).
    ''' 
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

def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True, 
                    edgecolor='w', facecolor=[0.7]*3, alpha=0.7, lw=0.5):
    """
    Produce a circular histogram of angles on ax.
    From: https://stackoverflow.com/questions/22562364/circular-polar-histogram-in-python

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
                     edgecolor=edgecolor, fill=True, linewidth=lw, facecolor=facecolor,
                    alpha=alpha)
    # Set the direction of the zero angle
    ax.set_theta_offset(offset)
    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])
        #ax.tick_params(which='both', axis='both', size=0)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)  # theta increasing clockwise
     
    return n, bins, patches

def colorbar_from_mappable(ax, norm, cmap, hue_title='', axes=[0.85, 0.3, 0.01, 0.4]): #pad=0.05):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig = ax.figure
    #ax.legend_.remove()
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=pad)
    cax = fig.add_axes(axes) 
    sm =  mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax) #ax=ax)
    cbar.ax.set_title(hue_title, fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    #pl.colorbar(im, cax=cax)

def plot_vector_path(ax, x, y, c, scale=1.5, width=0.005, headwidth=5, pivot='tail', 
                    colormap=mpl.cm.plasma, vmin=None, vmax=None, hue_title='',
                    axes=[0.8, 0.3, 0.01, 0.4]):
    if vmin is None:
        #vmin, vmax = b_[hue_param].min(), b_[hue_param].max()
        vmin, vmax = c.min(), c.max()
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    #colors=b_[hue_param]

#     uu = b_['ft_posx'].shift(periods=-1) - b_['ft_posx']
#     vv = b_['ft_posy'].shift(periods=-1) - b_['ft_posy']
#     ax.quiver(b_['ft_posx'].values, b_['ft_posy'].values, uu, vv, color=colormap(norm(colors)), 
#               angles='xy', scale_units='xy', scale=1.5)
    uu = np.roll(x, -1) - x # b_['ft_posx']
    vv = np.roll(y, -1) - y #b_['ft_posy'].shift(periods=-1) - b_['ft_posy']
    uu[-1]=np.nan
    vv[-1]=np.nan
    ax.quiver(x, y, uu, vv, color=colormap(norm(c)), 
              angles='xy', scale_units='xy', scale=scale, pivot=pivot,
              width=width, headwidth=headwidth)
    colorbar_from_mappable(ax, norm, cmap=colormap, hue_title=hue_title, axes=axes)
    return ax


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
