#!/usr/bin/env python

##PACKAGES##
from __future__ import division, absolute_import, print_function, unicode_literals
from builtins import zip
import numpy as np
import matplotlib
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from .._shared.helpers import *
from ..tools.reduce import reduce as reduceD
import six

##MAIN FUNCTION##
def animated_plot(x, *args, **kwargs):
    """
    Implements animated trajectory plot

    INPUTS:
    -numpy array(s)
    -list of numpy arrays

    OUTPUTS:
    -returns fig, ax, data, line_ani. fig, ax and line_ani are matplotlib figure, axis, and animation handles,
    respectively. Data is a numpy array of (reduced) data. (Optional, to use set return_data=True)
    """

    assert x[0].shape[1]>2, "Hypertools currently only supports animation for data with > 2 dims."

    ## HYPERTOOLS-SPECIFIC ARG PARSING ##
    kwargs = default_args(x, **kwargs)
    for k,v in six.iteritems(kwargs):
        next = kwargs[k]
        exec k + ' = next' in locals()
    import matplotlib.pyplot as plt #needs to happen after loading defaults
    kwargs = remove_hyper_args(x, **kwargs)

    ##SUB FUNCTIONS##
    def plot_cube(scale, const=1):
        cube = {
            "top"    : ( [[-1,1],[-1,1]], [[-1,-1],[1,1]], [[1,1],[1,1]] ),
            "bottom" : ( [[-1,1],[-1,1]], [[-1,-1],[1,1]], [[-1,-1],[-1,-1]] ),
            "left"   : ( [[-1,-1],[-1,-1]], [[-1,1],[-1,1]], [[-1,-1],[1,1]] ),
            "right"  : ( [[1,1],[1,1]], [[-1,1],[-1,1]], [[-1,-1],[1,1]] ),
            "front"  : ( [[-1,1],[-1,1]], [[-1,-1],[-1,-1]], [[-1,-1],[1,1]] ),
            "back"   : ( [[-1,1],[-1,1]], [[1,1],[1,1]], [[-1,-1],[1,1]] )
            }

        plane_list = []
        for side in cube:
            (Xs, Ys, Zs) = (
                np.asarray(cube[side][0])*scale*const,
                np.asarray(cube[side][1])*scale*const,
                np.asarray(cube[side][2])*scale*const
                )
            plane_list.append(ax.plot_wireframe(Xs, Ys, Zs, rstride=1, cstride=1, color='black', linewidth=2))
        return plane_list

    def update_lines(num, data_lines, lines, trail_lines, cube_scale, tail_duration):
        if hasattr(update_lines, 'planes'):
            for plane in update_lines.planes:
                plane.remove()

        update_lines.planes = plot_cube(cube_scale)
        ax.view_init(elev=10, azim=rotations*(360*(num/data_lines[0].shape[0])))
        ax.dist=8-zoom

        for line, data, trail in zip(lines, data_lines, trail_lines):
            if num<=tail_duration:
                    line.set_data(data[0:num+1, 0:2].T)
                    line.set_3d_properties(data[0:num+1, 2])
            else:
                line.set_data(data[num-tail_duration:num+1, 0:2].T)
                line.set_3d_properties(data[num-tail_duration:num+1, 2])
            if chemtrails:
                trail.set_data(data[0:num + 1, 0:2].T)
                trail.set_3d_properties(data[0:num + 1, 2])
        return lines,trail_lines

    args_list = parse_args(x, args)
    kwargs_list = parse_kwargs(x, kwargs)

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    if type(x) is not list:
        x = [x]

    interp_val = frame_rate*duration/(x[0].shape[0] - 1)
    x = interp_array_list(x, interp_val=interp_val)
    x = center(x)
    x = scale(x)

    if tail_duration==0:
        tail_duration = 1
    else:
        tail_duration = int(frame_rate*tail_duration)

    lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], linewidth=3, *args_list[idx], **kwargs_list[idx])[0] for idx,dat in enumerate(x)]
    trail = [
        ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], alpha=.3, linewidth=3, *args_list[idx], **kwargs_list[idx])[0]
        for idx, dat in enumerate(x)]

    ax.set_axis_off()

    # Get cube scale from data
    cube_scale = 1

    # Setting the axes properties
    ax.set_xlim3d([-cube_scale, cube_scale])
    ax.set_ylim3d([-cube_scale, cube_scale])
    ax.set_zlim3d([-cube_scale, cube_scale])

    #add legend
    if legend:
        proxies = [plt.Rectangle((0, 0), 1, 1, fc=palette[idx]) for idx,label in enumerate(legend_data)]
        ax.legend(proxies,legend_data)

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, update_lines, x[0].shape[0], fargs=(x, lines, trail, cube_scale, tail_duration),
                                   interval=1000/frame_rate, blit=False, repeat=False)
    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=frame_rate, bitrate=1800)
        line_ani.save(save_path, writer=writer)

    if show:
        plt.show()

    if return_data:
        return fig,ax,x,line_ani
