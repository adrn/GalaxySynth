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

    ax.axvline(0., zorder=-1000)
    ax.axhline(0., zorder=-1000)

    # axis labels
    ax.set_xlabel(r"${0}$".format(ax_names[ix[0]]), fontsize=30)
    ax.set_ylabel(r"${0}$".format(ax_names[ix[1]]), fontsize=30, rotation='horizontal', labelpad=16)

    fig.tight_layout()

    # for i in range(w.shape[1]):
    #     ax.plot(w[:1000,i,ix[0]], w[:1000,i,ix[1]], marker=None, zorder=-1000)

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
                         frames=100, interval=200)  # , blit=True)

    return anim

def main():
    np.random.seed(42)

    norbits = 8
    nsteps = 10000
    dt = 2.5

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
    t,w = pot.integrate_orbit(w0, dt=dt, nsteps=nsteps)
    R = np.sqrt(w[:,:,0]**2 + w[:,:,1]**2)
    phi = np.arctan2(w[:,:,1], w[:,:,0])
    z = w[:,:,2]

    # Animate the orbits
    # anim_xy = make_anim(w, [0,1], ntrails=10)
    # anim_xy.save("/Users/adrian/projects/galaxy-synthesizer/output/xy.mov", bitrate=-1)

    # anim_xz = make_anim(w, [0,2], ntrails=10)
    # anim_xz.save("/Users/adrian/projects/galaxy-synthesizer/output/xz.mov", bitrate=-1)

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
    phi_cross = np.array([argrelmin(pphi)[0] for pphi in (phi % np.pi).T])
    z_cross = np.array([argrelmin(zz**2)[0] for zz in z.T])

    s = pyo.Server(audio="offline", nchnls=2, sr=44100).boot()
    s.recordOptions(dur=20.,
                    filename="/Users/adrian/projects/galaxy-synthesizer/output/le_test.wav",
                    fileformat=0)

    # define a scale object
    mode = 'aeolian'
    key = 'C'
    high_freqs = musak.Scale(key=key, mode=mode, octave=4).freqs
    high_freqs = np.append(high_freqs, high_freqs*2.)
    high_freqs = np.append(high_freqs, high_freqs*2.)

    low_freqs = musak.Scale(key=key, mode=mode, octave=2).freqs
    low_freqs = np.append(low_freqs, low_freqs*2.)
    low_freqs = np.append(low_freqs, low_freqs*2.)

    hi_tone_dur = 0.1
    lo_tone_dur = 0.5

    q_R = quantize(R, nbins=7*3, min=3., max=20.)

    _cache = []
    # for j in range(1,nsteps):
    for j in range(1,100):
        delay = t[j] / dt / 5.
        print(t[j], delay)

        hif = []
        lof = []
        for k in range(norbits):
            if j in z_cross[k]:
                hif.append(high_freqs[q_R[j,k]])

            if j in phi_cross[k]:
                print('phi cross', j)
                lof.append(low_freqs[q_R[j,k]])

        if len(hif) > 0:
            env = pyo.Fader(fadein=hi_tone_dur*0.02, fadeout=hi_tone_dur*0.02,
                            dur=hi_tone_dur*0.9, mul=0.01).play(delay=delay, dur=hi_tone_dur+0.1)
            osc = pyo.Sine(freq=hif, mul=env).mix(voices=norbits*4)
            osc.out(delay=delay, dur=hi_tone_dur)

            _cache.append(osc)
            _cache.append(env)

        if len(lof) > 0:
            lo_env = pyo.Fader(fadein=lo_tone_dur*0.02, fadeout=lo_tone_dur*0.02,
                               dur=lo_tone_dur*0.9, mul=0.05).play(delay=delay, dur=lo_tone_dur+0.1)
            lo_osc = pyo.Sine(freq=lof, mul=lo_env).mix(voices=norbits*4)
            lo_osc.out(delay=delay, dur=lo_tone_dur)
            # lo_chorus = pyo.Chorus(lo_osc).out(delay=delay, dur=lo_tone_dur)

            _cache.append(lo_osc)
            _cache.append(lo_env)
            # _cache.append(lo_chorus)

        # wave = pyo.SquareTable(order=15).normalize()
        # osc = pyo.Osc(table=wave, freq=j_freqs, mul=env).mix(voices=norbits)
        # verb = pyo.Freeverb(osc, size=1., damp=0.8, bal=0.85).out(delay=delay, dur=tone_dur*2)
        # chorus = pyo.Chorus(osc).out(delay=delay, dur=tone_dur*2.)

        # _cache.append(wave)
        # _cache.append(chorus)
        # _cache.append(verb)

    # print(t[phi_cross[0][0]])

    s.recstart()
    s.start()
    s.recstop()
    s.stop()
    # s.gui(locals())

if __name__ == "__main__":
    main()
