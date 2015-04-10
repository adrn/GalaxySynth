# coding: utf-8

""" Simplify interfacing with PYO. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from astropy import log as logger
import pyo

# Project
# ...

__all__ = ['simple_sine', 'filtered_square', 'filtered_saw']

def simple_sine(delay, note, amp, dur=1.):

    # -- Nice, angelic --
    env = pyo.Adsr(attack=dur*0.1, sustain=0.707, decay=0.1*dur, release=dur*0.5,
                   dur=dur*0.9, mul=amp).play(delay=delay, dur=2.5*dur)
    osc = pyo.Sine(freq=pyo.midiToHz(note), mul=env).mix(1)
    osc.out(delay=delay, dur=dur)

    return osc, env

def filtered_square(delay, note, amp, dur=1.):

    a = np.sqrt(np.linspace(1.,0.,15))
    t = pyo.HarmTable(a.tolist())

    dur *= 2.

    env = pyo.Fader(fadein=.02, fadeout=0.02, dur=dur*0.9, mul=amp).play(dur=2.5*dur, delay=delay)

    adsr = pyo.Adsr(attack=dur*0.05, sustain=0.707, decay=0.1*dur, release=dur*0.7,
                    dur=dur*0.9, mul=amp).play(dur=2.5*dur, delay=delay)
    osc = pyo.Osc(t, freq=pyo.midiToHz(note), mul=adsr).mix(1)

    rev = pyo.Freeverb(osc, size=1., damp=0.5, bal=1., mul=env).play(dur=2.5*dur, delay=delay)
    # rev.out(delay=delay, dur=dur)

    eq = pyo.Biquad(rev, freq=500, q=1., type=2).play(dur=2.5*dur, delay=delay)
    eq.out(delay=delay, dur=dur)
    # eq = None

    return osc, env, rev, eq

def filtered_saw(delay, note, amp, dur=1.):
    t = pyo.SawTable(order=15).normalize()

    dur *= 2.

    env = pyo.Fader(fadein=.02, fadeout=0.02, dur=dur*0.9, mul=amp).play(dur=2.5*dur, delay=delay)

    adsr = pyo.Adsr(attack=dur*0.05, sustain=0.707, decay=0.1*dur, release=dur*0.7,
                    dur=dur*0.9, mul=amp).play(dur=2.5*dur, delay=delay)
    osc = pyo.Osc(t, freq=pyo.midiToHz(note), mul=adsr).mix(1)

    rev = pyo.Freeverb(osc, size=1., damp=0.5, bal=1., mul=env).play(dur=2.5*dur, delay=delay)
    # rev.out(delay=delay, dur=dur)

    eq = pyo.Biquad(rev, freq=800, q=1., type=0).play(dur=2.5*dur, delay=delay)
    eq.out(delay=delay, dur=dur)
    # eq = None

    return osc, env, rev, eq
