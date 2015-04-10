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
import numpy as np
from scipy.signal import argrelmin
# --
import gary.dynamics as gd
import gary.integrate as gi
import gary.potential as gp
from gary.units import galactic

# Project
import pyo
import synthetic as syn

# Global parameters
mode = 'phrygian'
key = 'C'

# integration
nsteps = 2000
dt = 0.25

# animation
nframes = 600
delay_fac = 0.1
downsample_anim = 1
interval = int(1000 * delay_fac * downsample_anim)

BULGE_COLOR = '#F7E739'
DISK_COLOR = '#4085B9'
HALO_COLOR = '#9B261F'

def bulge_no_disk(norbits, make_animations=False, make_audio=False,
                  path="/Users/adrian/projects/galaxy-synthesizer/output/bulge_no_disk/"):

    # output path
    if not os.path.exists(path):
        os.mkdir(path)

    # orbits
    pot,w0 = syn.make_spheroid(norbits)
    t,w = pot.integrate_orbit(w0, dt=dt, nsteps=nsteps,
                              Integrator=gi.DOPRI853Integrator)
    print("Done integrating")

    # Animate the orbits
    if make_animations:
        anim_xy = syn.animate_orbits(w[::downsample_anim,:,0], w[::downsample_anim,:,1],
                                     nframes=nframes, interval=interval, figsize=(8,8),
                                     star_alpha=0.75, star_color=BULGE_COLOR,
                                     figure_color='#000000',
                                     xlim=(-1,1), ylim=(-1,1))
        anim_xy.save(os.path.join(path,"xy.mp4"),
                     bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
                     savefig_kwargs={'facecolor':'#000000'})

        anim_xz = syn.animate_orbits(w[::downsample_anim,:,0], w[::downsample_anim,:,2],
                                     nframes=nframes, interval=interval, figsize=(8,8),
                                     star_alpha=0.75, star_color=BULGE_COLOR,
                                     figure_color='#000000',
                                     xlim=(-1,1), ylim=(-1,1))
        anim_xz.save(os.path.join(path,"xz.mp4"),
                     bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
                     savefig_kwargs={'facecolor':'#000000'})

def disk_no_bulge(norbits, make_animations=False, make_audio=False,
                  path="/Users/adrian/projects/galaxy-synthesizer/output/disk_no_bulge/"):

    # output path
    if not os.path.exists(path):
        os.mkdir(path)

    # orbits
    pot,w0 = syn.make_disk_galaxy(norbits)
    t,w = pot.integrate_orbit(w0, dt=dt, nsteps=nsteps,
                              Integrator=gi.DOPRI853Integrator)
    print("Done integrating")

    # Animate the orbits
    if make_animations:
        anim_xy = syn.animate_orbits(w[::downsample_anim,:,0], w[::downsample_anim,:,1],
                                     nframes=nframes, interval=interval, figsize=(8,8),
                                     star_color=DISK_COLOR, star_size=64,
                                     figure_color='#000000',
                                     xlim=(-14,14), ylim=(-14,14))
        anim_xy.save(os.path.join(path,"xy.mp4"),
                     bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
                     savefig_kwargs={'facecolor':'#000000'})

        anim_xz = syn.animate_orbits(w[::downsample_anim,:,0], w[::downsample_anim,:,2],
                                     nframes=nframes, interval=interval, figsize=(8,8),
                                     star_alpha=0.5, star_color=DISK_COLOR, star_size=64,
                                     figure_color='#000000',
                                     xlim=(-14,14), ylim=(-2,2))
        anim_xz.save(os.path.join(path,"xz.mp4"),
                     bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
                     savefig_kwargs={'facecolor':'#000000'})

        # R = np.sqrt(w[:,:,0]**2 + w[:,:,1]**2)
        # anim_xz = syn.animate_orbits(R[::downsample_anim], w[::downsample_anim,:,2],
        #                              nframes=nframes, interval=interval,
        #                              star_alpha=0.5, star_color=DISK_COLOR,
        #                              figure_color='#000000', hline=True,
        #                              xlim=(0,16), ylim=(-0.5,0.5), figsize=(8,8))
        # anim_xz.save(os.path.join(path,"Rz.mp4"),
        #              bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
        #              savefig_kwargs={'facecolor':'#000000'})

    if make_audio:
        # ---------------------------------------------------------------
        # Playground
        # s = pyo.Server(nchnls=2, sr=44100).boot()
        s = pyo.Server(audio="offline", nchnls=1, sr=44100).boot()
        s.recordOptions(dur=60.,
                        filename=os.path.join(path, "{0}_{1}.wav".format(key,mode)),
                        fileformat=0)

        # define a scale object
        mk_hi = syn.MasterKey(key=key, mode=mode, octave=(3,4))
        mk_lo = syn.MasterKey(key=key, mode=mode, octave=(2,3))

        # -----------------------------------------------------------------
        # disk
        events = syn.cyl_orbit_to_events2(t, w[:,:norbits],
                                          mk_hi.midi_notes, mk_lo.midi_notes)

        cache = []
        for d,n,ph in zip(*events)[:300]:
            d = d / dt * delay_fac
            print(d, n)
            # amp = (np.array(ph)*0.03 + 0.07).tolist()
            # c = syn.simple_sine(d, n, amp, dur=1.)
            amp = (np.array(ph)*0.04 + 0.06).tolist()
            c = syn.filtered_square(d, n, amp, dur=2.)
            cache.append(c)

        s.recstart()
        s.start()
        s.recstop()
        s.stop()

def disk_with_bulge(norbits, make_animations=False, make_audio=False,
                    path="/Users/adrian/projects/galaxy-synthesizer/output/disk_with_bulge/"):

    # output path
    if not os.path.exists(path):
        os.mkdir(path)

    # orbits
    pot,w0_disk = syn.make_disk_galaxy(norbits)
    pot,w0_sphe = syn.make_spheroid(norbits)
    w0 = np.vstack((w0_disk,w0_sphe))
    t,w = pot.integrate_orbit(w0, dt=dt, nsteps=nsteps,
                              Integrator=gi.DOPRI853Integrator)
    print("Done integrating")

    star_colors = [DISK_COLOR]*norbits + [BULGE_COLOR]*norbits

    # Animate the orbits
    if make_animations:
        anim_xy = syn.animate_orbits(w[::downsample_anim,:,0], w[::downsample_anim,:,1],
                                     nframes=nframes, interval=interval, figsize=(8,8),
                                     star_color=star_colors, star_size=64,
                                     figure_color='#000000',
                                     xlim=(-14,14), ylim=(-14,14))
        anim_xy.save(os.path.join(path,"xy.mp4"),
                     bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
                     savefig_kwargs={'facecolor':'#000000'})

        anim_xz = syn.animate_orbits(w[::downsample_anim,:,0], w[::downsample_anim,:,2],
                                     nframes=nframes, interval=interval, figsize=(8,8),
                                     star_alpha=0.5, star_color=star_colors, star_size=64,
                                     figure_color='#000000',
                                     xlim=(-14,14), ylim=(-2,2))
        anim_xz.save(os.path.join(path,"xz.mp4"),
                     bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
                     savefig_kwargs={'facecolor':'#000000'})

        # R = np.sqrt(w[:,:,0]**2 + w[:,:,1]**2)
        # anim_xz = syn.animate_orbits(R[::downsample_anim], w[::downsample_anim,:,2],
        #                              nframes=nframes, interval=interval,
        #                              star_alpha=0.5, star_color=star_colors,
        #                              figure_color='#000000', hline=True,
        #                              xlim=(0,16), ylim=(-0.5,0.5), figsize=(8,8))
        # anim_xz.save(os.path.join(path,"Rz.mp4"),
        #              bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
        #              savefig_kwargs={'facecolor':'#000000'})

    if make_audio:
        # ---------------------------------------------------------------
        # Playground
        # s = pyo.Server(nchnls=2, sr=44100).boot()
        s = pyo.Server(audio="offline", nchnls=1, sr=44100).boot()
        s.recordOptions(dur=60.,
                        filename=os.path.join(path, "{0}_{1}.wav".format(key,mode)),
                        fileformat=0)

        # define a scale object
        mk_hi = syn.MasterKey(key=key, mode=mode, octave=(3,4))
        mk_lo = syn.MasterKey(key=key, mode=mode, octave=(2,3))

        # -----------------------------------------------------------------
        # disk
        events = syn.cyl_orbit_to_events2(t, w[:,:norbits],
                                          mk_hi.midi_notes, mk_lo.midi_notes)

        cache = []
        for d,n,ph in zip(*events)[:250]:
            d = d / dt * delay_fac
            print(d, n)
            # amp = (np.array(ph)*0.03 + 0.07).tolist()
            # c = syn.simple_sine(d, n, amp, dur=1.)
            amp = (np.array(ph)*0.04 + 0.06).tolist()
            c = syn.filtered_square(d, n, amp, dur=2.)
            cache.append(c)

        # -----------------------------------------------------------------
        # bulge
        mk = syn.MasterKey(key=key, mode=mode, octave=(4,5))
        events = syn.xyz_orbit_to_events(t, w[:,norbits:], mk.midi_notes)

        for d,n in zip(*events)[:600]:
            d = d / dt * delay_fac
            print(d, n)
            # c = syn.filtered_square(d, n, 0.05, dur=1.)
            c = syn.simple_sine(d, n, 0.008, dur=0.25)
            cache.append(c)

        s.recstart()
        s.start()
        s.recstop()
        s.stop()

def disk_with_bulge_halo(norbits, make_animations=False, make_audio=False,
                         path="/Users/adrian/projects/galaxy-synthesizer/output/disk_with_bulge_halo/"):

    # output path
    if not os.path.exists(path):
        os.mkdir(path)

    # orbits
    pot,w0_disk = syn.make_disk_galaxy(norbits)
    pot,w0_sphe = syn.make_spheroid(norbits)
    pot,w0_halo = syn.make_halo(norbits)
    w0 = np.vstack((w0_disk,w0_sphe,w0_halo))
    t,w = pot.integrate_orbit(w0, dt=dt, nsteps=nsteps,
                              Integrator=gi.DOPRI853Integrator)
    print("Done integrating")

    star_colors = [DISK_COLOR]*norbits + [BULGE_COLOR]*norbits + [HALO_COLOR]*norbits

    # Animate the orbits
    if make_animations:
        anim_xy = syn.animate_orbits(w[::downsample_anim,:,0], w[::downsample_anim,:,1],
                                     nframes=nframes, interval=interval, figsize=(8,8),
                                     star_alpha=0.75, star_size=64,
                                     star_color=star_colors,
                                     figure_color='#000000',
                                     xlim=(-25,25), ylim=(-25,25), ntrails=0)
        anim_xy.save(os.path.join(path,"xy.mp4"),
                     bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
                     savefig_kwargs={'facecolor':'#000000'})

        anim_xz = syn.animate_orbits(w[::downsample_anim,:,0], w[::downsample_anim,:,2],
                                     nframes=nframes, interval=interval, figsize=(8,8),
                                     star_alpha=0.5, star_color=star_colors, star_size=64,
                                     figure_color='#000000',
                                     xlim=(-25,25), ylim=(-25,25))
        anim_xz.save(os.path.join(path,"xz.mp4"),
                     bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
                     savefig_kwargs={'facecolor':'#000000'})

    if make_audio:
        # ---------------------------------------------------------------
        # Playground
        # s = pyo.Server(nchnls=2, sr=44100).boot()
        s = pyo.Server(audio="offline", nchnls=1, sr=44100).boot()
        s.recordOptions(dur=60.,
                        filename=os.path.join(path, "{0}_{1}.wav".format(key,mode)),
                        fileformat=0)

        # define a scale object
        mk_hi = syn.MasterKey(key=key, mode=mode, octave=(3,4))
        mk_lo = syn.MasterKey(key=key, mode=mode, octave=(2,3))

        # -----------------------------------------------------------------
        # disk
        events = syn.cyl_orbit_to_events2(t, w[:,:norbits],
                                          mk_hi.midi_notes, mk_lo.midi_notes)

        cache = []
        for d,n,ph in zip(*events)[:250]:
            d = d / dt * delay_fac
            print(d, n)
            # amp = (np.array(ph)*0.03 + 0.07).tolist()
            # c = syn.simple_sine(d, n, amp, dur=1.)
            amp = (np.array(ph)*0.04 + 0.06).tolist()
            c = syn.filtered_square(d, n, amp, dur=2.)
            cache.append(c)

        # -----------------------------------------------------------------
        # bulge
        mk = syn.MasterKey(key=key, mode=mode, octave=(4,5))
        events = syn.xyz_orbit_to_events(t, w[:,norbits:], mk.midi_notes)

        for d,n in zip(*events)[:500]:
            d = d / dt * delay_fac
            print(d, n)
            # c = syn.filtered_square(d, n, 0.05, dur=1.)
            c = syn.simple_sine(d, n, 0.007, dur=0.25)
            cache.append(c)

        # -----------------------------------------------------------------
        # halo
        # mk = syn.MasterKey(key=key, mode=mode, octave=(1,2))
        # events = syn.halo_orbit_to_events(t, w[:,2*norbits:], mk.midi_notes)
        mk = syn.MasterKey(key=key, mode=mode, octave=2)
        x_pool = mk.midi_notes[::2]  # roots
        y_pool = mk.midi_notes[1::2]  # weirdos
        z_pool = [x + 12 for x in mk.midi_notes[::2]]  # lower roots
        events = syn.elliptical_orbit_to_events2(t, w[:,2*norbits:], x_pool, y_pool, z_pool)

        for d,n in zip(*events)[:300]:
            d = d / dt * delay_fac
            print('halo', d, n)
            c = syn.filtered_saw(d, n, 0.15, dur=3.5)
            cache.append(c)

        s.recstart()
        s.start()
        s.recstop()
        s.stop()

def elliptical(norbits, qz, make_animations=False, make_audio=False, direct_freqs=False,
               path="/Users/adrian/projects/galaxy-synthesizer/output/elliptical_E0/"):

    # output path
    if not os.path.exists(path):
        os.mkdir(path)

    _dt = dt*4

    # orbits
    pot,w0 = syn.make_elliptical(norbits, qz=qz)
    t,w = pot.integrate_orbit(w0, dt=_dt, nsteps=nsteps,
                              Integrator=gi.DOPRI853Integrator)
    print("Done integrating")

    # Animate the orbits
    if make_animations:
        anim_xy = syn.animate_orbits(w[::downsample_anim,:,0], w[::downsample_anim,:,1],
                                     nframes=nframes, interval=interval, figsize=(8,8),
                                     star_color=HALO_COLOR, star_size=64,
                                     figure_color='#000000',
                                     xlim=(-30,30), ylim=(-30,30))
        anim_xy.save(os.path.join(path,"xy.mp4"),
                     bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
                     savefig_kwargs={'facecolor':'#000000'})

        anim_xz = syn.animate_orbits(w[::downsample_anim,:,0], w[::downsample_anim,:,2],
                                     nframes=nframes, interval=interval, figsize=(8,8),
                                     star_alpha=0.5, star_color=HALO_COLOR, star_size=64,
                                     figure_color='#000000',
                                     xlim=(-30,30), ylim=(-30,30))
        anim_xz.save(os.path.join(path,"xz.mp4"),
                     bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
                     savefig_kwargs={'facecolor':'#000000'})

    if make_audio:

        if direct_freqs:
            s = pyo.Server(audio="offline", nchnls=1, sr=44100).boot()
            s.recordOptions(dur=60.,
                            filename=os.path.join(path, "orbit_freqs.wav".format(key,mode)),
                            fileformat=0)

            # elliptical halo
            events = syn.elliptical_orbit_to_events(t, w)

            cache = []
            for d,f in zip(*events)[:100]:
                d = d / dt * delay_fac
                print('halo', d, f)
                c = syn.filtered_saw_chord(d, f, 0.2, dur=3.5)
                cache.append(c)

            s.recstart()
            s.start()
            s.recstop()
            s.stop()

        else:
            s = pyo.Server(audio="offline", nchnls=1, sr=44100).boot()
            s.recordOptions(dur=60.,
                            filename=os.path.join(path, "{0}_{1}.wav".format(key,mode)),
                            fileformat=0)

            # elliptical halo
            mk2 = syn.MasterKey(key=key, mode=mode, octave=2)
            mk3 = syn.MasterKey(key=key, mode=mode, octave=3)
            mk4 = syn.MasterKey(key=key, mode=mode, octave=4)

            x_pool = mk2.midi_notes[::2].tolist() + mk3.midi_notes[::2].tolist()  # roots
            y_pool = mk2.midi_notes[1::2].tolist() + mk3.midi_notes[1::2].tolist() + mk4.midi_notes[1::2].tolist()  # weirdos
            z_pool = mk3.midi_notes[::2].tolist() + mk4.midi_notes[::2].tolist()  # higher roots
            events = syn.elliptical_orbit_to_events2(t, w, x_pool, y_pool, z_pool)

            cache = []
            for d,n in zip(*events)[:100]:
                d = d / dt * delay_fac
                print('halo', d, n)
                c = syn.filtered_saw(d, n, 0.2, dur=3.5)
                cache.append(c)

            s.recstart()
            s.start()
            s.recstop()
            s.stop()


def single_disk_orbit_animate(path="/Users/adrian/projects/galaxy-synthesizer/output/single_disk_orbit/"):

    # output path
    if not os.path.exists(path):
        os.mkdir(path)

    # orbits
    pot,w0 = syn.make_disk_galaxy(1)
    t,w = pot.integrate_orbit(w0, dt=dt, nsteps=nsteps,
                              Integrator=gi.DOPRI853Integrator)

    R = np.sqrt(w[:,0,0]**2 + w[:,0,1]**2)
    phi = np.arctan2(w[:,0,1], w[:,0,0]) % (2*np.pi)
    z = w[:,0,2]

    nframes = 1000
    anim_xy = syn.animate_orbits(w[::2,:,0], w[::2,:,1],
                                 nframes=nframes, interval=interval/4, figsize=(8,8),
                                 star_alpha=1., star_size=64,
                                 star_color=DISK_COLOR, hline=True,
                                 figure_color='#000000', full_orbit=True,
                                 xlim=(-10,10), ylim=(-10,10), ntrails=25)
    anim_xy.save(os.path.join(path,"xy.mp4"),
                 bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
                 savefig_kwargs={'facecolor':'#000000'})

    anim_xz = syn.animate_orbits(w[::2,:,0], w[::2,:,2],
                                 nframes=nframes, interval=interval/4, figsize=(8,8),
                                 star_alpha=1., star_size=64,
                                 star_color=DISK_COLOR, hline=True,
                                 figure_color='#000000', full_orbit=True,
                                 xlim=(-10,10), ylim=(-2,2), ntrails=25)
    anim_xz.save(os.path.join(path,"xz.mp4"),
                 bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
                 savefig_kwargs={'facecolor':'#000000'})

    # ---
    fig,axes = plt.subplots(3,1,figsize=(10.67,8))
    fig.patch.set_facecolor('#000000')

    for ax in axes:
        ax.patch.set_facecolor('#000000')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_frame_on(False)

    axes[0].plot(t, R, marker=None, linestyle='-',
                 color='#aaaaaa', lw=3., zorder=-100)
    axes[0].set_ylim(R.min()-0.2, R.max()+0.2)

    axes[1].plot(t, phi, marker=None, linestyle='-',
                 color='#aaaaaa', lw=3., zorder=-100)
    axes[1].set_ylim(-0.1,2*np.pi+0.1)

    axes[2].plot(t, z, marker=None, linestyle='-',
                 color='#aaaaaa', lw=3., zorder=-100)

    fig.tight_layout()
    fig.savefig(os.path.join(path, "3-panel.png"), facecolor='#000000')

    s = pyo.Server(audio="offline", nchnls=1, sr=44100).boot()
    s.recordOptions(dur=60.,
                    filename=os.path.join(path, "{0}_{1}.wav".format(key,mode)),
                    fileformat=0)

    # define a scale object
    mk_hi = syn.MasterKey(key=key, mode=mode, octave=(3,4))
    mk_lo = syn.MasterKey(key=key, mode=mode, octave=(2,3))

    # -----------------------------------------------------------------
    # disk
    events = syn.cyl_orbit_to_events2(t, w[::8], mk_hi.midi_notes, mk_lo.midi_notes)

    cache = []
    for d,n,ph in zip(*events)[:300]:
        d = d / dt * delay_fac
        print(d, n)
        # amp = (np.array(ph)*0.03 + 0.07).tolist()
        # c = syn.simple_sine(d, n, amp, dur=1.)
        amp = (np.array(ph)*0.04 + 0.06).tolist()
        c = syn.filtered_square(d, n, amp, dur=2.)
        cache.append(c)

    s.recstart()
    s.start()
    s.recstop()
    s.stop()

def single_bulge_orbit_animate(path="/Users/adrian/projects/galaxy-synthesizer/output/single_bulge_orbit/"):

    # output path
    if not os.path.exists(path):
        os.mkdir(path)

    # orbits
    pot,w0 = syn.make_spheroid(3)
    w0 = w0[2:]
    t,w = pot.integrate_orbit(w0, dt=dt, nsteps=nsteps,
                              Integrator=gi.DOPRI853Integrator)

    nframes = 1000
    anim_xy = syn.animate_orbits(w[::2,:,0], w[::2,:,1],
                                 nframes=nframes, interval=interval/4, figsize=(8,8),
                                 star_alpha=1., star_size=64,
                                 star_color=BULGE_COLOR,
                                 figure_color='#000000', full_orbit=True,
                                 xlim=(-1,1), ylim=(-1,1), ntrails=25)
    anim_xy.save(os.path.join(path,"xy.mp4"),
                 bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
                 savefig_kwargs={'facecolor':'#000000'})

    anim_xz = syn.animate_orbits(w[::2,:,0], w[::2,:,2],
                                 nframes=nframes, interval=interval/4, figsize=(8,8),
                                 star_alpha=1., star_size=64,
                                 star_color=BULGE_COLOR,
                                 figure_color='#000000', full_orbit=True,
                                 xlim=(-1,1), ylim=(-1,1), ntrails=25)
    anim_xz.save(os.path.join(path,"xz.mp4"),
                 bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
                 savefig_kwargs={'facecolor':'#000000'})

    s = pyo.Server(audio="offline", nchnls=1, sr=44100).boot()
    s.recordOptions(dur=60.,
                    filename=os.path.join(path, "{0}_{1}.wav".format(key,mode)),
                    fileformat=0)

    # -----------------------------------------------------------------
    # bulge
    mk = syn.MasterKey(key=key, mode=mode, octave=(4,5))
    events = syn.xyz_orbit_to_events(t, w[::8], mk.midi_notes)

    cache = []
    for d,n in zip(*events)[:500]:
        d = d / dt * delay_fac
        print(d, n)
        # c = syn.filtered_square(d, n, 0.05, dur=1.)
        c = syn.simple_sine(d, n, 0.007, dur=0.25)
        cache.append(c)

    s.recstart()
    s.start()
    s.recstop()
    s.stop()

def single_halo_orbit_animate(path="/Users/adrian/projects/galaxy-synthesizer/output/single_halo_orbit/"):

    # output path
    if not os.path.exists(path):
        os.mkdir(path)

    # orbits
    pot,w0 = syn.make_halo(3)
    w0 = w0[0:1]
    t,w = pot.integrate_orbit(w0, dt=dt, nsteps=nsteps,
                              Integrator=gi.DOPRI853Integrator)

    nframes = 1000
    anim_xy = syn.animate_orbits(w[::2,:,0], w[::2,:,1],
                                 nframes=nframes, interval=interval/4, figsize=(8,8),
                                 star_alpha=1., star_size=64,
                                 star_color=HALO_COLOR, hline=True, vline=True,
                                 figure_color='#000000', full_orbit=True,
                                 xlim=(-30,30), ylim=(-30,30), ntrails=25)
    anim_xy.save(os.path.join(path,"xy.mp4"),
                 bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
                 savefig_kwargs={'facecolor':'#000000'})

    anim_xz = syn.animate_orbits(w[::2,:,0], w[::2,:,2],
                                 nframes=nframes, interval=interval/4, figsize=(8,8),
                                 star_alpha=1., star_size=64,
                                 star_color=HALO_COLOR, hline=True, vline=True,
                                 figure_color='#000000', full_orbit=True,
                                 xlim=(-30,30), ylim=(-30,30), ntrails=25)
    anim_xz.save(os.path.join(path,"xz.mp4"),
                 bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
                 savefig_kwargs={'facecolor':'#000000'})

    s = pyo.Server(audio="offline", nchnls=1, sr=44100).boot()
    s.recordOptions(dur=60.,
                    filename=os.path.join(path, "{0}_{1}.wav".format(key,mode)),
                    fileformat=0)

    # -----------------------------------------------------------------
    # halo
    mk = syn.MasterKey(key=key, mode=mode, octave=2)
    x_pool = mk.midi_notes[::2]  # roots
    y_pool = mk.midi_notes[1::2]  # weirdos
    z_pool = [x + 12 for x in mk.midi_notes[::2]]  # lower roots
    events = syn.elliptical_orbit_to_events2(t, w[::8], x_pool, y_pool, z_pool)

    cache = []
    for d,n in zip(*events)[:300]:
        d = d / dt * delay_fac
        print('halo', d, n)
        c = syn.filtered_saw(d, n, 0.15, dur=3.5)
        cache.append(c)

    s.recstart()
    s.start()
    s.recstop()
    s.stop()

def single_elliptical_orbit_animate(path="/Users/adrian/projects/galaxy-synthesizer/output/single_elliptical_orbit/"):

    # output path
    if not os.path.exists(path):
        os.mkdir(path)

    _dt = dt*4

    # orbits
    pot,w0 = syn.make_elliptical(5, qz=0.8)
    w0 = [25., 0., 0., 0.02, 0.01, 0.02]
    t,w = pot.integrate_orbit(w0, dt=_dt, nsteps=nsteps,
                              Integrator=gi.DOPRI853Integrator)
    t = t[::2]
    w = w[::2]
    nframes = 1000

    # anim_xy = syn.animate_orbits(w[::downsample_anim,:,0], w[::downsample_anim,:,1],
    #                              nframes=nframes, interval=interval/4, figsize=(8,8),
    #                              star_alpha=1., star_size=64,
    #                              star_color=DISK_COLOR,
    #                              figure_color='#000000', full_orbit=True,
    #                              xlim=(-30,30), ylim=(-30,30), ntrails=25)
    # anim_xy.save(os.path.join(path,"xy.mp4"),
    #              bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
    #              savefig_kwargs={'facecolor':'#000000'})

    # anim_xz = syn.animate_orbits(w[::downsample_anim,:,0], w[::downsample_anim,:,2],
    #                              nframes=nframes, interval=interval/4, figsize=(8,8),
    #                              star_alpha=1., star_size=64,
    #                              star_color=DISK_COLOR,
    #                              figure_color='#000000', full_orbit=True,
    #                              xlim=(-30,30), ylim=(-30,30), ntrails=25)
    # anim_xz.save(os.path.join(path,"xz.mp4"),
    #              bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
    #              savefig_kwargs={'facecolor':'#000000'})

    # ---
    fig,axes = plt.subplots(3,1,figsize=(10.67,8))
    fig.patch.set_facecolor('#000000')

    for ax in axes:
        ax.patch.set_facecolor('#000000')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_frame_on(False)

    for i in range(3):
        axes[i].plot(t, w[:,0,i], marker=None, linestyle='-',
                     color='#aaaaaa', lw=3., zorder=-100)
        axes[i].set_ylim(w[:,0,:3].min()-0.5, w[:,0,:3].max()+0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(path, "3-panel.png"), facecolor='#000000')

def tubes(path="/Users/adrian/projects/galaxy-synthesizer/output/tubes/"):

    # output path
    if not os.path.exists(path):
        os.mkdir(path)

    _dt = dt*4

    # orbits
    pot = gp.LeeSutoTriaxialNFWPotential(v_c=0.2, r_s=20.,
                                         a=1., b=0.9, c=0.8, units=galactic)

    w0 = np.array([[15., 0., 4., 0., 0.2, 0.05],
                   [4., 15., 0., 0., 0.05, 0.2],
                   [15., 0., 0., 0., 0.01, 0.18],
                   [15., 0.5, 0.5, -0.01, 0.05, 0.02],
                   [0., 0.5, 15, -0.01, 0.022, 0.05]])

    t,w = pot.integrate_orbit(w0, dt=_dt, nsteps=50000,
                              Integrator=gi.DOPRI853Integrator)
    print("integrated")

    # short axis tube
    # ix = 0
    # anim_xyz = syn.animate_3d(w[::2,ix:ix+1,0], w[::2,ix:ix+1,1], w[::2,ix:ix+1,2],
    #                           nframes=1000, interval=interval/4, figsize=(8,8),
    #                           figure_color='#000000',
    #                           xlim=(-30,30), ylim=(-30,30), zlim=(-30,30))
    # anim_xyz.save(os.path.join(path,"short_tube.mp4"),
    #               bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
    #               savefig_kwargs={'facecolor':'#000000'})

    # long axis tube
    ix = 1
    anim_xyz = syn.animate_3d(w[::2,ix:ix+1,0], w[::2,ix:ix+1,1], w[::2,ix:ix+1,2],
                              nframes=1000, interval=interval/4, figsize=(8,8),
                              figure_color='#000000',
                              xlim=(-30,30), ylim=(-30,30), zlim=(-30,30))
    anim_xyz.save(os.path.join(path,"long_tube.mp4"),
                  bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
                  savefig_kwargs={'facecolor':'#000000'})

    # # intermediate axis tube
    # ix = 2
    # anim_xyz = syn.animate_3d(w[::2,ix:ix+1,0], w[::2,ix:ix+1,1], w[::2,ix:ix+1,2],
    #                           nframes=1000, interval=interval/4, figsize=(8,8),
    #                           figure_color='#000000',
    #                           xlim=(-30,30), ylim=(-30,30), zlim=(-30,30))
    # anim_xyz.save(os.path.join(path,"int_tube.mp4"),
    #               bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
    #               savefig_kwargs={'facecolor':'#000000'})

    # # box
    # ix = 3
    # anim_xyz = syn.animate_3d(w[::2,ix:ix+1,0], w[::2,ix:ix+1,1], w[::2,ix:ix+1,2],
    #                           nframes=1000, interval=interval/4, figsize=(8,8),
    #                           figure_color='#000000',
    #                           xlim=(-30,30), ylim=(-30,30), zlim=(-30,30))
    # anim_xyz.save(os.path.join(path,"box1.mp4"),
    #               bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
    #               savefig_kwargs={'facecolor':'#000000'})

    # # box
    # ix = 4
    # anim_xyz = syn.animate_3d(w[::2,ix:ix+1,0], w[::2,ix:ix+1,1], w[::2,ix:ix+1,2],
    #                           nframes=1000, interval=interval/4, figsize=(8,8),
    #                           figure_color='#000000',
    #                           xlim=(-30,30), ylim=(-30,30), zlim=(-30,30))
    # anim_xyz.save(os.path.join(path,"box2.mp4"),
    #               bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
    #               savefig_kwargs={'facecolor':'#000000'})


def main():
    seed = 8675309

    # np.random.seed(seed)
    # single_disk_orbit_animate()

    # np.random.seed(seed)
    # single_bulge_orbit_animate()

    # np.random.seed(seed)
    # single_halo_orbit_animate()

    np.random.seed(seed)
    single_elliptical_orbit_animate()

    # np.random.seed(seed)
    # tubes()

    # these are just for visualization earlier in the talk
    # disk_no_bulge(norbits=1024, make_animations=True, make_audio=False,
    #               path="/Users/adrian/projects/galaxy-synthesizer/output/disk_no_bulge1024/")

    # bulge_no_disk(norbits=1024, make_animations=True, make_audio=False,
    #               path="/Users/adrian/projects/galaxy-synthesizer/output/bulge_no_disk1024/")

    # elliptical(norbits=1024, make_animations=True, make_audio=False, qz=0.95,
    #            path="/Users/adrian/projects/galaxy-synthesizer/output/elliptical_E0_1024/")

    # elliptical(norbits=1024, make_animations=True, make_audio=False, qz=0.6,
    #            path="/Users/adrian/projects/galaxy-synthesizer/output/elliptical_E3_1024/")

    # these are to make music
    # np.random.seed(seed)
    # disk_no_bulge(norbits=16, make_animations=True, make_audio=True)

    # np.random.seed(seed)
    # disk_with_bulge(norbits=16, make_animations=True, make_audio=True)

    # np.random.seed(seed)
    # disk_with_bulge_halo(norbits=16, make_animations=False, make_audio=True)

    # E0
    # np.random.seed(seed)
    # elliptical(norbits=32, make_animations=True, make_audio=True, qz=0.95, direct_freqs=False,
    #            path="/Users/adrian/projects/galaxy-synthesizer/output/elliptical_E0/")

    # np.random.seed(seed)
    # elliptical(norbits=256, make_animations=True, make_audio=True, qz=0.95, direct_freqs=True,
    #            path="/Users/adrian/projects/galaxy-synthesizer/output/elliptical_E0256/")

    # E3?
    # np.random.seed(seed)
    # elliptical(norbits=32, make_animations=True, make_audio=True, qz=0.6, direct_freqs=False,
    #            path="/Users/adrian/projects/galaxy-synthesizer/output/elliptical_E3/")

    # np.random.seed(seed)
    # elliptical(norbits=256, make_animations=True, make_audio=True, qz=0.6, direct_freqs=True,
    #            path="/Users/adrian/projects/galaxy-synthesizer/output/elliptical_E3256/")

if __name__ == "__main__":
    main()
