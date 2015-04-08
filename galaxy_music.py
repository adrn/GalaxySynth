# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import astropy.units as u
import matplotlib.pyplot as plt
from  matplotlib.animation import FuncAnimation
import numpy as np
from scipy.signal import argrelmin
# --
import gary.dynamics as gd
import gary.integrate as gi
import gary.potential as gp
from gary.units import galactic

# Project
import pyo
import musak

def make_anim(w, ix, ntrails=25):
    ntimes,norbits,_ = w.shape
    ax_names = ['x','y','z']

    # create a simple animation
    fig,ax = plt.subplots(1, 1, figsize=(6,6))
    ax.set_xlim(-25,25)
    ax.set_ylim(-25,25)

    # remove ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # axis labels
    ax.set_xlabel(r"${0}$".format(ax_names[ix[0]]), fontsize=30)
    ax.set_ylabel(r"${0}$".format(ax_names[ix[1]]), fontsize=30, rotation='horizontal', labelpad=16)

    fig.tight_layout()

    # initialize drawing points
    pts = ax.scatter(w[0,:,ix[0]], w[0,:,ix[1]],
                     marker='o', s=8, c='#666666')
    # pts2 = ax.scatter(w[0,:,ix[0]], w[0,:,ix[1]],
    #                   marker='o', s=16, c='#ff0000')
    # pts.set_offsets(np.vstack((w[20,:,ix[0]], w[20,:,ix[1]])))
    # plt.show()
    # return

    # trails = []

    def animate(i):
        # ii = i - ntrails

        # pts.set_data(w[i,:,ix[0]], w[i,:,ix[1]])
        pts.set_offsets(np.vstack((w[i,:,ix[0]], w[i,:,ix[1]])).T)
        # pts.set_offsets([w[i,0,ix[0]], w[i,0,ix[1]]])

        # for j,trail in zip(range(len(trails))[::-1],trails):
        #     if ii+j < 0:
        #         continue
        #     trail.set_data(w[ii+j,:,ix[0]], w[ii+j,:,ix[1]])
        return pts,  # ,trails

    anim = FuncAnimation(fig, animate,
                         frames=100, interval=50)  # , blit=True)
    # anim.save("/Users/adrian/Downloads/derp.mp4")
    plt.show()
    return
    return anim

def main():
    np.random.seed(42)

    norbits = 4
    nsteps = 10000

    # axisymmetric potential
    # pot = gp.LogarithmicPotential(v_c=1., r_h=0.1, q1=1., q2=1., q3=0.6,
    #                               units=galactic)
    pot = gp.MiyamotoNagaiPotential(m=2E12, a=6.5, b=.26, units=galactic)

    # ------------------------------------------------------------------------
    # initial conditions

    # position
    x0 = np.random.uniform(3., 10., size=norbits)
    y0 = np.zeros_like(x0)
    z0 = np.random.normal(0., 0.25, size=norbits)
    xyz0 = np.vstack((x0,y0,z0)).T

    # velocity
    vx0 = np.zeros_like(x0)
    vy0 = np.random.normal(1., 0.1, size=norbits)
    vz0 = np.random.normal(0., 0.005, size=norbits)
    vxyz0 = np.vstack((vx0,vy0,vz0)).T

    w0 = np.hstack((xyz0,vxyz0))

    # integrate orbits
    t,w = pot.integrate_orbit(w0, dt=2.5, nsteps=nsteps)
    R = np.sqrt(w[:,:,0]**2 + w[:,:,1]**2)
    phi = np.arctan2(w[:,:,1], w[:,:,0])
    z = w[:,:,2]

    # Animate the orbits
    # make_anim(w, [0,1], ntrails=10)

    # find when orbits cross
    # estimate periods -- find longest and shortest periods
    periods = np.zeros((norbits,3))
    for i in range(norbits):
        periods[i,0] = gd.peak_to_peak_period(t,R[:,i])
        periods[i,1] = gd.peak_to_peak_period(t,phi[:,i])
        periods[i,2] = gd.peak_to_peak_period(t,z[:,i])
    # z is the fastest period, φ is the slowest

    # -------------------------------
    # playground!

    def quantize(x, nbins, min=None, max=None):
        if min is None:
            min = x.min()
        if max is None:
            max = x.max()
        q = np.round((x - min) / (max - min) * (nbins-1)).astype(int)
        q[x > max] = max
        q[x < min] = min
        return q

    # variable length arrays
    phi_cross = np.array([argrelmin(pphi**2)[0] for pphi in phi.T])
    z_cross = np.array([argrelmin(zz**2)[0] for zz in z.T])

    s = pyo.Server().boot()

    # define a scale object
    mode = 'ionian'
    high_freqs = musak.Scale(key='C', mode=mode, octave=4).freqs
    high_freqs = np.append(high_freqs, high_freqs*2.)
    high_freqs = np.append(high_freqs, high_freqs*2.)

    low_freqs = musak.Scale(key='C', mode=mode, octave=3).freqs
    low_freqs = np.append(low_freqs, low_freqs*2.)
    low_freqs = np.append(low_freqs, low_freqs*2.)

    tone_dur = 1.

    q_R = quantize(R, nbins=7*3, min=3., max=20.)

    _cache = []
    # for j in range(1,nsteps):
    for j in range(1,100):
        j_freqs = []
        for k in range(norbits):
            if j in z_cross[k]:
                j_freqs.append(high_freqs[q_R[j,k]])

            elif j in phi_cross[k]:
                j_freqs.append(low_freqs[q_R[j,k]])

        if len(j_freqs) == 0:
            continue

        delay = t[j]/5.
        print(t[j], delay)
        env = pyo.Fader(fadein=tone_dur*0.02, fadeout=tone_dur*0.02, dur=tone_dur*0.9, mul=0.1).play(delay=delay, dur=tone_dur+0.1)
        osc = pyo.Sine(freq=j_freqs, mul=env).mix(voices=norbits)
        verb = pyo.Freeverb(osc, size=0.5, damp=0.5, bal=1., add=0).out(delay=delay, dur=tone_dur)
        osc.out(delay=delay, dur=tone_dur)

        _cache.append(osc)
        _cache.append(env)

    s.start()
    s.stop()

    s.gui(locals())


if __name__ == "__main__":
    main()
