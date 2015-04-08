# coding: utf-8

""" Turn a collection of orbits into something we can make into music. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from scipy.signal import argrelmin

__all__ = ['cyl_orbit_to_events', 'xyz_orbit_to_events']

def quantize(x, nbins, min=None, max=None):
    if min is None:
        min = x.min()

    if max is None:
        max = x.max()

    if max > min:
        q = np.round((x - min) / (max - min) * (nbins-1)).astype(int)
        q[x > max] = nbins-1
        q[x < min] = 0
    else:
        max = -max
        min = -min
        x = -x.copy()
        q = np.round((x - min) / (max - min) * (nbins-1)).astype(int)
        q[x > max] = nbins-1
        q[x < min] = 0

    return q

def cyl_orbit_to_events(t, w, midi_pool_hi, midi_pool_lo):
    """
    Convert an orbit to MIDI events using cylindrical coordinates and rules.

    For cylindrical orbits, crossing the disk midplane (x-y plane) triggers a
    high note. Crossing the x-z plane triggers a low note. The pitch of the note
    is set by the cylindrical radius at the time of either crossing. Smaller
    radius triggers a higher pitch note.

    Parameters
    ----------
    t : array_like
    w : array_like
    midi_pool : array_like

    """

    R = np.sqrt(w[:,:,0]**2 + w[:,:,1]**2)
    phi = np.arctan2(w[:,:,1], w[:,:,0]) % (2*np.pi)
    z = w[:,:,2]

    # variable length arrays
    phi_cross = np.array([argrelmin(pphi)[0] for pphi in phi.T])
    z_cross = np.array([argrelmin(zz**2)[0] for zz in z.T])

    # quantize R orbit
    nbins_hi = len(midi_pool_hi)
    q_R_hi = quantize(R, nbins=nbins_hi, min=R.max(), max=R.min())
    nbins_lo = len(midi_pool_lo)
    q_R_lo = quantize(R, nbins=nbins_lo, min=R.max(), max=R.min())

    delays = []
    notes = []
    for j in range(w.shape[0]):
        _no = []
        for i in range(w.shape[1]):
            if j in z_cross[i]:
                _no.append(midi_pool_hi[q_R_hi[j,i]])

            if j in phi_cross[i]:
                _no.append(midi_pool_lo[q_R_lo[j,i]])

        if len(_no) > 0:
            delays.append(t[j])
            notes.append(np.unique(_no).tolist())

    return delays, notes

def xyz_orbit_to_events(t, w, midi_pool_hi, midi_pool_lo):
    """
    Convert an orbit to MIDI events using Cartesian coordinates and rules.

    For Cartesian orbits...

    Parameters
    ----------
    t : array_like
    w : array_like
    midi_pool : array_like

    """

    x,y,z = w.T

    # # variable length arrays
    # phi_cross = np.array([argrelmin(pphi)[0] for pphi in phi.T])
    # z_cross = np.array([argrelmin(zz**2)[0] for zz in z.T])

    # # quantize R orbit
    # nbins_hi = len(midi_pool_hi)
    # q_R_hi = quantize(R, nbins=nbins_hi, min=R.max(), max=R.min())
    # nbins_lo = len(midi_pool_lo)
    # q_R_lo = quantize(R, nbins=nbins_lo, min=R.max(), max=R.min())

    # delays = []
    # notes = []
    # for j in range(w.shape[0]):
    #     _no = []
    #     for i in range(w.shape[1]):
    #         if j in z_cross[i]:
    #             _no.append(midi_pool_hi[q_R_hi[j,i]])

    #         if j in phi_cross[i]:
    #             _no.append(midi_pool_lo[q_R_lo[j,i]])

    #     if len(_no) > 0:
    #         delays.append(t[j])
    #         notes.append(np.unique(_no).tolist())

    # return delays, notes
