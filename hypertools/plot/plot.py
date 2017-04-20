#!/usr/bin/env python

# built in packages
from __future__ import division
from builtins import zip
import sys
import warnings
import re
import itertools

# external code
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import mpl_toolkits.mplot3d.axes3d as p3
import seaborn as sns
import pandas as pd
import numpy as np

# internal code
from .._shared.helpers import *
from .static import static_plot
from .animate import animated_plot
from ..tools.cluster import cluster
from ..tools.df2mat import df2mat
from ..tools.reduce import reduce as reduceD
from ..tools.normalize import normalize as normalizer
from .plot_trajectory import plot_trajectory
from .static_helpers import *
from .animate_helpers import *

## MAIN FUNCTION ##
def plot(x, *args, **kwargs):
    """
    Plots dimensionality reduced data and parses plot arguments

    Parameters
    ----------
    x : Numpy array, DataFrame or list of arrays/dfs
        Data for the plot. The form should be samples (rows) by features (cols).

    color(s) (list): A list of colors for each line to be plotted.
        Can be named colors, RGB values (e.g. (.3, .4, .1)) or hex codes.
        If defined, overrides palette. See here for list of named colors.
        Note: must be the same length as X.

    group : list of str, floats or ints
        A list of group labels. Length must match the number of rows in your
        dataset. If the data type is numerical, the values will be mapped to
        rgb values in the specified palette. If the data type is strings,
        the points will be labeled categorically. To label a subset of points,
        use None (i.e. ['a', None, 'b','a']).

    linestyle(s) : list
        A list of line styles

    marker(s) : list
        A list of marker types

    palette : str
        A matplotlib or seaborn color palette

    labels : list
        A list of labels for each point. Must be dimensionality of data (x).
        If no label is wanted for a particular point, input None.

    legend : list
        A list of string labels to be plotted in a legend (one for each list
        item).

    ndims : int
        An `int` representing the number of dims to plot in. Must be 1,2, or 3.
        NOTE: Currently only works with static plots.

    normalize : str or False
        If set to 'across', the columns of the input data will be z-scored
        across lists (default). If set to 'within', the columns will be
        z-scored within each list that is passed. If set to 'row', each row of
        the input data will be z-scored. If set to False, the input data will
        be returned (default is False).

    n_clusters : int
        If n_clusters is passed, HyperTools will perform k-means clustering
        with the k parameter set to n_clusters. The resulting clusters will
        be plotted in different colors according to the color palette.

    animate : bool
        If True, plots the data as an animated trajectory (default: False).

    show : bool
        If set to False, the figure will not be displayed, but the figure,
        axis and data objects will still be returned (see Outputs)
        (default: True).

    save_path str :
        Path to save the image/movie. Must include the file extension in the
        save path (i.e. save_path='/path/to/file/image.png'). NOTE: If saving
        an animation, FFMPEG must be installed (this is a matplotlib req).
        FFMPEG can be easily installed on a mac via homebrew brew install
        ffmpeg or linux via apt-get apt-get install ffmpeg. If you don't
        have homebrew (mac only), you can install it like this:
        /usr/bin/ruby -e "$(curl -fsSL
        https://raw.githubusercontent.com/Homebrew/install/master/install)".

    explore : bool
        Displays user defined labels will appear on hover. If no labels are
        passed, the point index and coordinate will be plotted. To use,
        set explore=True. Note: Explore more is currently only supported
        for 3D static plots.

    duration (animation only) : float
        Length of the animation in seconds (default: 30 seconds)

    tail_duration (animation only) : float
        Sets the length of the tail of the data (default: 2 seconds)

    rotations (animation only) : float
        Number of rotations around the box (default: 2)

    zoom (animation only) : float
        Zoom, positive numbers will zoom in (default: 0)

    chemtrails (animation only) : bool
        Added trail with change in opacity (default: False)

    Returns
    ----------
    fig, ax, data : Matplotlib.Figure.figure, Matplotlib.Axes.axis, Numpy array
        By default, the plot function outputs a figure handle
        (matplotlib.figure.Figure), axis handle (matplotlib.axes._axes.Axes)
        and data (list of numpy arrays), e.g. fig,axis,data = hyp.plot(x)

        If animate=True, the plot function additionally outputs an animation
        handle (matplotlib.animation.FuncAnimation)
        e.g. fig,axis,data,line_ani = hyp.plot(x, animate=True).

    """

    # turn data into common format - a list of arrays
    x = format_data(x)

    ## HYPERTOOLS-SPECIFIC ARG PARSING ##

    if 'colors' in kwargs:
        kwargs['color'] = kwargs['colors']
        del kwargs['colors']

    if 'linestyles' in kwargs:
        kwargs['linestyle'] = kwargs['linestyles']
        del kwargs['linestyles']

    if 'markers' in kwargs:
        kwargs['marker'] = kwargs['markers']
        del kwargs['markers']

    if 'normalize' in kwargs:
        normalize = kwargs['normalize']
        x = normalizer(x, normalize=normalize, internal=True)
        del kwargs['normalize']
    else:
        x = normalizer(x, normalize=False, internal=True)

    # reduce dimensionality of the data
    if 'ndims' in kwargs:
        ndims=kwargs['ndims']
        x = reduceD(x,ndims, internal=True)
        del kwargs['ndims']
    elif x[0].shape[1]>3:
        x = reduceD(x,3, internal=True)
        ndims=3
    else:
        ndims=x[0].shape[1]

    if 'n_clusters' in kwargs:
        n_clusters=kwargs['n_clusters']

        cluster_labels = cluster(x, n_clusters=n_clusters, ndims=ndims)
        x = reshape_data(x,cluster_labels)
        del kwargs['n_clusters']

        if 'group' in kwargs:
            warnings.warn('n_clusters overrides group, ignoring group.')
            del kwargs['group']

    if 'group' in kwargs:
        group=kwargs['group']
        del kwargs['group']

        if 'color' in kwargs:
            warnings.warn("Using group, color keyword will be ignored.")
            del kwargs['color']

        # if list of lists, unpack
        if any(isinstance(el, list) for el in group):
            group = list(itertools.chain(*group))

        # if all of the elements are numbers, map them to colors
        if all(isinstance(el, int) or isinstance(el, float) for el in group):
            group = vals2bins(group)
        elif all(isinstance(el, str) for el in group):
            group = group_by_category(group)

        # reshape the data according to group
        x = reshape_data(x,group)

        # interpolate lines if they are grouped
        if all([symbol not in args for symbol in Line2D.markers.keys()]):
            x = patch_lines(x)

    if 'style' in kwargs:
        sns.set(style=kwargs['style'])
        del kwargs['style']
    else:
        sns.set(style="whitegrid")

    if 'palette' in kwargs:
        sns.set_palette(palette=kwargs['palette'], n_colors=len(x))
        palette = sns.color_palette(palette=kwargs['palette'], n_colors=len(x))
        del kwargs['palette']
    else:
        sns.set_palette(palette="hls", n_colors=len(x))
        palette=sns.color_palette(palette="hls", n_colors=len(x))

    if 'animate' in kwargs:
        animate=kwargs['animate']
        del kwargs['animate']
        # if animate mode, pass the color palette via kwargs so we can build a legend
        kwargs['color_palette']=palette
    else:
        animate='static'

    if 'zoom' in kwargs:
        zoom=kwargs['zoom']
        del kwargs['zoom']
    else:
        zoom=0

    if 'chemtrails' in kwargs:
        chemtrails= kwargs['chemtrails']
        del kwargs['chemtrails']
    else:
        chemtrails=False

    if 'rotations' in kwargs:
        rotations=kwargs['rotations']
        del kwargs['rotations']
    else:
        rotations=2

    if 'duration' in kwargs:
        duration=kwargs['duration']
        del kwargs['duration']
    else:
        duration=30

    if 'frame_rate' in kwargs:
        frame_rate=kwargs['frame_rate']
        del kwargs['frame_rate']
    else:
        frame_rate=50

    if 'tail_duration' in kwargs:
        tail_duration=kwargs['tail_duration']
        del kwargs['tail_duration']
    else:
        tail_duration=2

    if 'return_data' in kwargs:
        return_data = kwargs['return_data']
        del kwargs['return_data']
    else:
        return_data=False

    if 'legend' in kwargs:

        legend = kwargs['legend']
        del kwargs['legend']
    else:
        legend=False

    if 'color_palette' in kwargs:
        palette = kwargs['color_palette']
        del kwargs['color_palette']

    if 'save_path' in kwargs:
        save=True
        save_path = kwargs['save_path']
        del kwargs['save_path']
    else:
        save=False

    # handle show flag
    if 'show' in kwargs:
        show=kwargs['show']
        del kwargs['show']
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    else:
        show=True
        import matplotlib.pyplot as plt

    # handle labels flag
	if 'labels' in kwargs:
		labels=kwargs['labels']
		del kwargs['labels']
	else:
		labels=False

	# handle explore flag
	if 'explore' in kwargs:
		assert x[0].ndim>1, "Explore mode is currently only supported for 3D plots."
		kwargs['picker']=True
		del kwargs['explore']
		explore=True
	else:
		explore=False

    ##PARSE LEFTOVER MATPLOTLIB ARGS##
    args_list = parse_args(x,args)

    ##PARSE LEFTOVER MATPLOTLIB KWARGS##
    kwargs_list = parse_kwargs(x,kwargs)

    # scale and center the data
    x = center(x)
    x = scale(x)

    # interpolate if line plot - NEED TO CHECK IF LINE
    interp_val = frame_rate*duration/(x[0].shape[0] - 1)
    x = interp_array_list(x, interp_val=interp_val)

    # set axis style to cube
    axis_style='cube'

    # plot
    if animate is 'static':
        fig, ax, data = plot_static(x, args_list, kwargs_list)
        ax.view_init(elev=10, azim=0)
        ax.dist=8-zoom

        if axis_style is 'cube':
            ax.set_axis_off()
            plot_cube(ax)

        if labels:
            add_labels(data,labels)
    else:
        fig, ax, data, ani = animated_plot(x, animate, *args, **kwargs)

    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=frame_rate, bitrate=1800)
        line_ani.save(save_path, writer=writer)

    # add legend
    if legend:
        proxies = [plt.Rectangle((0, 0), 1, 1, fc=palette[idx]) for idx,label in enumerate(legend)]
        ax.legend(proxies,legend)

    if show:
        plt.show()
