# coding: utf-8

""" ...explain... """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import gary.potential as gp
from gary.units import galactic

__all__ = ['animate_orbits', 'make_disk_galaxy', 'make_spheroid',
           'make_halo', 'make_elliptical', 'animate_3d']

def animate_orbits(x, y, interval, nframes=None, ntrails=5,
                   xlim=None, ylim=None, labels=False,
                   hline=False, vline=False,
                   figsize=(6,6),
                   star_color='#555555', star_size=16, star_alpha=1.,
                   figure_color='#ffffff', axes_visible=True, full_orbit=False,
                   **animation_kwargs):
    """
    Animate the input orbits.

    Parameters
    ----------
    x : array_like
        Should have shape ``(ntimes,norbits)``.
    y : array_like
        Should have shape ``(ntimes,norbits)``.
    interval : int
    nframes : int (optional)
    ntrails : int (optional)
    xlim, ylim : str (optional)
    labels : iterable (optional)
    **animation_kwargs
        Passed to ``matplotlib``'s ``FuncAnimation``.

    """

    ntimes,norbits = x.shape

    if x.shape != y.shape:
        raise ValueError("Shapes of input time series must match.")

    if nframes is None:
        nframes = ntimes

    # create a simple animation
    fig,ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor(figure_color)
    ax.patch.set_facecolor(figure_color)

    if xlim is not None:
        ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)

    # remove ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if vline:
        ax.axvline(0., zorder=-1000, color='#bbbbbb')

    if hline:
        ax.axhline(0., zorder=-1000, color='#bbbbbb')

    if full_orbit:
        for i in range(norbits):
            ax.plot(x[:,i], y[:,i], marker=None, linestyle='-',
                    color='#444444', lw=3., zorder=-100)

    ax.xaxis.set_visible(axes_visible)
    ax.yaxis.set_visible(axes_visible)

    # axis labels
    if labels:
        ax.set_xlabel(r"{0}".format(labels[0]), fontsize=30)
        ax.set_ylabel(r"{0}".format(labels[1]), fontsize=30,
                      rotation='horizontal', labelpad=16)

    # initialize drawing points
    pts = ax.scatter(x[0], y[0], marker='o', s=star_size, c=star_color, alpha=star_alpha)

    # container for
    if ntrails > 0:
        trails = []
        alphas = np.linspace(star_alpha,0,ntrails+1)[1:]**1.5
        sizes = star_size / (np.ones(ntrails))  # np.linspace(4,2,ntrails)
        for i,(alpha,ms) in enumerate(zip(alphas,sizes)):
            trails.append(ax.scatter(x[0], y[0], marker='o',
                                     s=ms, alpha=alpha, c=star_color))

    def animate(i):
        pts.set_offsets(np.vstack((x[i], y[i])).T)

        if ntrails > 0:
            ii = i - ntrails
            for j,trail in zip(range(len(trails))[::-1],trails):
                if ii+j < 0:
                    continue
                # trail.set_data(x[ii+j], y[ii+j])
                trail.set_offsets(np.vstack((x[ii+j], y[ii+j])).T)
            return pts, trails
        else:
            return pts,

    # squishy
    fig.tight_layout()

    anim = FuncAnimation(fig, animate,
                         frames=nframes, interval=int(interval),
                         **animation_kwargs)

    return anim

def animate_3d(x, y, z, interval, nframes=None,
               xlim=None, ylim=None, zlim=None, figsize=(6,6),
               figure_color='#ffffff', axes_visible=True,
               **animation_kwargs):
    """
    Animate the input orbits.

    Parameters
    ----------
    x : array_like
        Should have shape ``(ntimes,norbits)``.
    y : array_like
        Should have shape ``(ntimes,norbits)``.
    interval : int
    nframes : int (optional)
    ntrails : int (optional)
    xlim, ylim : str (optional)
    labels : iterable (optional)
    **animation_kwargs
        Passed to ``matplotlib``'s ``FuncAnimation``.

    """
    from mpl_toolkits.mplot3d import Axes3D

    ntimes,norbits = x.shape

    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError("Shapes of input time series must match.")

    if nframes is None:
        nframes = ntimes

    # create a simple animation
    fig = plt.figure(figsize=figsize)
    ax = Axes3D(fig)

    fig.patch.set_facecolor(figure_color)
    ax.patch.set_facecolor(figure_color)

    if xlim is not None:
        ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)

    if zlim is not None:
        ax.set_zlim(*zlim)

    for i in range(norbits):
        ax.plot(x[:,i], y[:,i], z[:,i], marker=None, linestyle='-',
                color='#444444', lw=1., zorder=-100)

    # lines
    ax.plot([0., 0.], [0.,0.], [0.,20.], c='#1a9641', lw=2, zorder=-1000) # g
    ax.plot([0., 0.], [0.,25.], [0.,0.], c='#d7191c', lw=2, zorder=-1000) # r
    ax.plot([0., 30.], [0.,0.], [0.,0.], c='#0571b0', lw=2, zorder=-1000) # b

    ax.set_axis_off()

    # axis labels
    # if labels:
    #     ax.set_xlabel(r"{0}".format(labels[0]), fontsize=30)
    #     ax.set_ylabel(r"{0}".format(labels[1]), fontsize=30,
    #                   rotation='horizontal', labelpad=16)

    def animate(i):
        elev = (((i/400.) * 180) % 180.) - 90.
        azim = ((i/500.) * 360.) % 360.
        ax.view_init(elev=elev, azim=azim)
        return None

    anim = FuncAnimation(fig, animate,
                         frames=nframes, interval=int(interval),
                         **animation_kwargs)

    return anim

def make_disk_galaxy(nstars):
    """

    """

    # axisymmetric potential
    potential = gp.LM10Potential(halo=dict(q1=1.,q2=1.,q3=1.))

    # initial conditions
    # --> position
    phi = np.random.uniform(0, 2*np.pi, size=nstars)
    R = np.random.uniform(2., 10., size=nstars)
    x0 = R*np.cos(phi)
    y0 = R*np.sin(phi)
    z0 = np.random.normal(0., 0.1, size=nstars)
    xyz0 = np.vstack((x0,y0,z0)).T

    # --> velocity
    Vphi = np.random.normal(0.215, 0.02, size=nstars)
    VR = np.random.normal(0., 0.02, size=nstars)
    vx0 = -Vphi*np.sin(phi) + VR*np.cos(phi)
    vy0 = Vphi*np.cos(phi) + VR*np.sin(phi)
    vz0 = np.random.normal(0., 0.015, size=nstars)
    vxyz0 = np.vstack((vx0,vy0,vz0)).T

    w0 = np.hstack((xyz0,vxyz0))

    return potential, w0

def make_spheroid(nstars):

    # axisymmetric potential
    potential = gp.LM10Potential(halo=dict(q1=1.,q2=1.,q3=1.))

    # initial conditions
    # --> position
    # R = np.sqrt(np.random.uniform(0., 1., nstars))
    R = np.random.uniform(0.1,0.5,nstars)
    phi = np.random.uniform(0,2*np.pi,nstars)
    theta = np.arccos(2*np.random.uniform(0,1,nstars)-1)

    x0 = R*np.cos(phi)*np.sin(theta)
    y0 = R*np.sin(phi)*np.sin(theta)
    z0 = R*np.cos(theta)
    xyz0 = np.vstack((x0,y0,z0)).T
    r = np.sqrt(x0**2 + y0**2 + z0**2)

    # --> velocity
    vr = np.random.normal(0., 0.025, size=nstars)
    # vtan = np.sqrt(V**2 - vr**2)
    # vtan = V
    vtan = np.sqrt(potential.G*potential.mass_enclosed(xyz0) / r)

    velocity_angle = np.linspace(0.,2*np.pi,nstars)
    theta_dot = np.cos(velocity_angle) * vtan / r
    sin_theta_phi_dot = np.sin(velocity_angle) * vtan / r

    sint,cost = np.sin(theta),np.cos(theta)
    sinp,cosp = np.sin(phi),np.cos(phi)
    phi_dot = sin_theta_phi_dot / sint
    vx0 = sint*cosp*vr + r*cost*cosp*theta_dot - r*sint*sinp*phi_dot
    vy0 = sint*sinp*vr + r*cost*sinp*theta_dot + r*sint*cosp*phi_dot
    vz0 = cost*vr - r*sint*theta_dot
    vxyz0 = np.vstack((vx0,vy0,vz0)).T

    # L = np.cross(xyz0, vxyz0)
    # Lmag = np.linalg.norm(L,axis=1)

    w0 = np.hstack((xyz0,vxyz0))

    return potential, w0

def make_halo(nstars):

    # axisymmetric potential
    potential = gp.LM10Potential(halo=dict(q1=1.,q2=1.,q3=1.))

    # initial conditions
    # --> position
    # R = np.sqrt(np.random.uniform(0., 1., nstars))
    R = np.random.uniform(10.,20.,nstars)
    phi = np.random.uniform(0,2*np.pi,nstars)
    theta = np.arccos(2*np.random.uniform(0,1,nstars)-1)

    x0 = R*np.cos(phi)*np.sin(theta)
    y0 = R*np.sin(phi)*np.sin(theta)
    z0 = R*np.cos(theta)
    xyz0 = np.vstack((x0,y0,z0)).T
    r = np.sqrt(x0**2 + y0**2 + z0**2)

    # --> velocity
    vr = np.random.normal(0., 0.05, size=nstars)
    vtan = np.sqrt(potential.G*potential.mass_enclosed(xyz0) / r)

    velocity_angle = np.linspace(0.,2*np.pi,nstars)
    theta_dot = np.cos(velocity_angle) * vtan / r
    sin_theta_phi_dot = np.sin(velocity_angle) * vtan / r

    sint,cost = np.sin(theta),np.cos(theta)
    sinp,cosp = np.sin(phi),np.cos(phi)
    phi_dot = sin_theta_phi_dot / sint
    vx0 = sint*cosp*vr + r*cost*cosp*theta_dot - r*sint*sinp*phi_dot
    vy0 = sint*sinp*vr + r*cost*sinp*theta_dot + r*sint*cosp*phi_dot
    vz0 = cost*vr - r*sint*theta_dot
    vxyz0 = np.vstack((vx0,vy0,vz0)).T

    # L = np.cross(xyz0, vxyz0)
    # Lmag = np.linalg.norm(L,axis=1)

    w0 = np.hstack((xyz0,vxyz0))

    return potential, w0

def make_elliptical(nstars, qz):

    # triaxial potential
    potential = gp.LeeSutoTriaxialNFWPotential(v_c=0.2, r_s=20., a=1., b=0.95, c=qz, units=galactic)

    # initial conditions
    # --> position
    # R = np.sqrt(np.random.uniform(0., 1., nstars))
    phi = np.random.uniform(0,2*np.pi,nstars)
    theta = np.arccos(2*np.random.uniform(0,1,nstars)-1)
    xx = np.cos(phi)*np.sin(theta)
    yy = np.sin(phi)*np.sin(theta)
    zz = np.cos(theta)
    Rmax = 25. / np.sqrt(xx**2 + (yy/0.95)**2 + (zz/qz)**2)
    R = np.random.uniform(1.,Rmax,nstars)

    x0 = R*np.cos(phi)*np.sin(theta)
    y0 = R*np.sin(phi)*np.sin(theta)
    z0 = R*np.cos(theta)
    xyz0 = np.vstack((x0,y0,z0)).T
    r = np.sqrt(x0**2 + y0**2 + z0**2)

    # --> velocity
    vr = np.random.normal(0., 0.05, size=nstars)
    vtan = np.sqrt(potential.G*potential.mass_enclosed(xyz0) / r)

    velocity_angle = np.linspace(0.,2*np.pi,nstars)
    theta_dot = np.cos(velocity_angle) * vtan / r
    sin_theta_phi_dot = np.sin(velocity_angle) * vtan / r

    sint,cost = np.sin(theta),np.cos(theta)
    sinp,cosp = np.sin(phi),np.cos(phi)
    phi_dot = sin_theta_phi_dot / sint
    vx0 = sint*cosp*vr + r*cost*cosp*theta_dot - r*sint*sinp*phi_dot
    vy0 = sint*sinp*vr + r*cost*sinp*theta_dot + r*sint*cosp*phi_dot
    vz0 = cost*vr - r*sint*theta_dot
    vxyz0 = np.vstack((vx0,vy0,vz0)).T

    # L = np.cross(xyz0, vxyz0)
    # Lmag = np.linalg.norm(L,axis=1)

    w0 = np.hstack((xyz0,vxyz0))

    return potential, w0
