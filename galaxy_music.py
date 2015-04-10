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
mode = 'aeolian'
key = 'G'

# integration
nsteps = 2000
dt = 0.25

# animation
nframes = 150
delay_fac = 0.1
downsample_anim = 1
interval = int(1000 * delay_fac * downsample_anim)

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
                                     star_alpha=0.75, star_color='#F7E739',
                                     figure_color='#000000',
                                     xlim=(-1,1), ylim=(-1,1))
        anim_xy.save(os.path.join(path,"xy.mp4"),
                     bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
                     savefig_kwargs={'facecolor':'#000000'})

        anim_xz = syn.animate_orbits(w[::downsample_anim,:,0], w[::downsample_anim,:,2],
                                     nframes=nframes, interval=interval, figsize=(8,8),
                                     star_alpha=0.75, star_color='#F7E739',
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
                                     star_color='#4085B9', figure_color='#000000',
                                     xlim=(-14,14), ylim=(-14,14))
        anim_xy.save(os.path.join(path,"xy.mp4"),
                     bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
                     savefig_kwargs={'facecolor':'#000000'})

        anim_xz = syn.animate_orbits(w[::downsample_anim,:,0], w[::downsample_anim,:,2],
                                     nframes=nframes, interval=interval, figsize=(8,8),
                                     star_alpha=0.5, star_color='#4085B9',
                                     figure_color='#000000',
                                     xlim=(-14,14), ylim=(-14,14))
        anim_xz.save(os.path.join(path,"xz.mp4"),
                     bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
                     savefig_kwargs={'facecolor':'#000000'})

        R = np.sqrt(w[:,:,0]**2 + w[:,:,1]**2)
        anim_xz = syn.animate_orbits(R[::downsample_anim], w[::downsample_anim,:,2],
                                     nframes=nframes, interval=interval,
                                     star_alpha=0.5, star_color='#4085B9',
                                     figure_color='#000000', hline=True,
                                     xlim=(0,16), ylim=(-0.5,0.5), figsize=(8,8))
        anim_xz.save(os.path.join(path,"Rz.mp4"),
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
        for d,n,ph in zip(*events)[:150]:
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

    # Animate the orbits
    if make_animations:
        anim_xy = syn.animate_orbits(w[::downsample_anim,:,0], w[::downsample_anim,:,1],
                                     nframes=nframes, interval=interval, figsize=(8,8),
                                     star_color='#4085B9', figure_color='#000000',
                                     xlim=(-14,14), ylim=(-14,14))
        anim_xy.save(os.path.join(path,"xy.mp4"),
                     bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
                     savefig_kwargs={'facecolor':'#000000'})

        anim_xz = syn.animate_orbits(w[::downsample_anim,:,0], w[::downsample_anim,:,2],
                                     nframes=nframes, interval=interval, figsize=(8,8),
                                     star_alpha=0.5, star_color='#4085B9',
                                     figure_color='#000000',
                                     xlim=(-14,14), ylim=(-14,14))
        anim_xz.save(os.path.join(path,"xz.mp4"),
                     bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
                     savefig_kwargs={'facecolor':'#000000'})

        R = np.sqrt(w[:,:,0]**2 + w[:,:,1]**2)
        anim_xz = syn.animate_orbits(R[::downsample_anim], w[::downsample_anim,:,2],
                                     nframes=nframes, interval=interval,
                                     star_alpha=0.5, star_color='#4085B9',
                                     figure_color='#000000', hline=True,
                                     xlim=(0,16), ylim=(-0.5,0.5), figsize=(8,8))
        anim_xz.save(os.path.join(path,"Rz.mp4"),
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
        for d,n,ph in zip(*events)[:150]:
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

        for d,n in zip(*events)[:300]:
            d = d / dt * delay_fac
            print(d, n)
            # c = syn.filtered_square(d, n, 0.05, dur=1.)
            c = syn.simple_sine(d, n, 0.01, dur=0.25)
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

    # Animate the orbits
    if make_animations:
        anim_xy = syn.animate_orbits(w[::downsample_anim,:,0], w[::downsample_anim,:,1],
                                     nframes=nframes, interval=interval, figsize=(8,8),
                                     star_color='#4085B9', figure_color='#000000',
                                     xlim=(-25,25), ylim=(-25,25))
        anim_xy.save(os.path.join(path,"xy.mp4"),
                     bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'],
                     savefig_kwargs={'facecolor':'#000000'})

        anim_xz = syn.animate_orbits(w[::downsample_anim,:,0], w[::downsample_anim,:,2],
                                     nframes=nframes, interval=interval, figsize=(8,8),
                                     star_alpha=0.5, star_color='#4085B9',
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
        for d,n,ph in zip(*events)[:150]:
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

        for d,n in zip(*events)[:300]:
            d = d / dt * delay_fac
            print(d, n)
            # c = syn.filtered_square(d, n, 0.05, dur=1.)
            c = syn.simple_sine(d, n, 0.01, dur=0.25)
            cache.append(c)

        # -----------------------------------------------------------------
        # halo
        mk = syn.MasterKey(key=key, mode=mode, octave=1)
        events = syn.halo_orbit_to_events(t, w[:,2*norbits:], mk.midi_notes[::2])

        for d,n in zip(*events)[:200]:
            d = d / dt * delay_fac
            print(d, n)
            c = syn.filtered_saw(d, n, 0.2, dur=5.)
            cache.append(c)

        s.recstart()
        s.start()
        s.recstop()
        s.stop()

def main():
    np.random.seed(4)

    # these are just for visualization earlier in the talk
    # disk_no_bulge(norbits=1024, make_animations=True, make_audio=False,
    #               path="/Users/adrian/projects/galaxy-synthesizer/output/disk_no_bulge1024/")

    # bulge_no_disk(norbits=1024, make_animations=True, make_audio=False,
    #               path="/Users/adrian/projects/galaxy-synthesizer/output/bulge_no_disk1024/")

    # these are to make music
    # np.random.seed(4)
    # disk_no_bulge(norbits=8, make_animations=False, make_audio=True)

    # np.random.seed(4)
    # disk_with_bulge(norbits=8, make_animations=False, make_audio=True)

    np.random.seed(4)
    disk_with_bulge_halo(norbits=8, make_animations=True, make_audio=False)

    # wave = pyo.SquareTable(order=15).normalize()
    # osc = pyo.Osc(table=wave, freq=j_freqs, mul=env).mix(voices=norbits)
    # verb = pyo.Freeverb(osc, size=1., damp=0.8, bal=0.85).out(delay=delay, dur=tone_dur*2)
    # chorus = pyo.Chorus(osc).out(delay=delay, dur=tone_dur*2.)

if __name__ == "__main__":
    main()
