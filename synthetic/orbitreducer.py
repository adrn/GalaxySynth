# coding: utf-8

""" Turn a collection of orbits into something we can make into music. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import gary.dynamics as gd
import numpy as np
from scipy.signal import argrelmin, argrelmax

__all__ = ['cyl_orbit_to_events', 'cyl_orbit_to_events2', 'xyz_orbit_to_events',
           'halo_orbit_to_events']

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

def cyl_orbit_to_events(t, w, midi_pool_hi, midi_pool_lo, time_resolution=None):
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

    # set amplitudes from size of z oscillations
    all_amps = np.abs(z).max(axis=0) / 10.

    # variable length arrays
    phi_cross = np.array([argrelmin(pphi)[0] for pphi in phi.T])
    z_cross = np.array([argrelmin(zz**2)[0] for zz in z.T])

    # quantize R orbit
    RR = np.sqrt(R)
    nbins_hi = len(midi_pool_hi)
    q_R_hi = quantize(RR, nbins=nbins_hi, min=RR.max(), max=RR.min())
    nbins_lo = len(midi_pool_lo)
    q_R_lo = quantize(RR, nbins=nbins_lo, min=RR.max(), max=RR.min())

    delays = []
    notes = []
    amps = []
    for j in range(w.shape[0]):
        _no = []
        _amps = []
        for i in range(w.shape[1]):
            if j in z_cross[i]:
                _no.append(midi_pool_hi[q_R_hi[j,i]])
                _amps.append(all_amps[i])

            if j in phi_cross[i]:
                _no.append(midi_pool_lo[q_R_lo[j,i]])
                _amps.append(all_amps[i])

        if len(_no) > 0:
            delays.append(t[j])
            notes.append(np.unique(_no).tolist())
            amps.append(_amps)

    delays = np.array(delays)
    notes = np.array(notes)
    amps = np.array(amps)

    return delays, notes, amps

    # if time_resolution is None:
    #     return delays, notes

    # new_delays = []
    # new_notes = []
    # q_delays = quantize(delays, nbins=int(delays.max()/time_resolution))
    # for xx in np.unique(q_delays):
    #     ix = q_delays == xx
    #     new_delays.append(delays[ix][0])
    #     new_notes.append([item for sublist in notes[ix] for item in sublist])

    # return np.array(new_delays), np.array(new_notes)

def cyl_orbit_to_events2(t, w, midi_pool_hi, midi_pool_lo):
    """
    Convert an orbit to MIDI events using cylindrical coordinates and rules.

    For cylindrical orbits, crossing the disk midplane (x-y plane) triggers a
    high note with pitch set by the vertical oscillation frequency. Crossing
    the x-z plane triggers a low note with pitch set by the azimuthal frequency.
    The radial oscillations modulate the volume of the note.

    Parameters
    ----------
    t : array_like
    w : array_like

    """
    ntimes,norbits,_ = w.shape

    R = np.sqrt(w[:,:,0]**2 + w[:,:,1]**2)
    phi = np.arctan2(w[:,:,1], w[:,:,0]) % (2*np.pi)
    z = w[:,:,2]

    # normalized R for oscillations
    normed_R = (R - R.min()) / (R.max() - R.min())

    # variable length arrays
    phi_cross = np.array([argrelmin(pphi)[0] for pphi in phi.T])
    z_cross = np.array([argrelmin(zz**2)[0] for zz in z.T])

    # estimate periods
    T_z = np.array([gd.peak_to_peak_period(t, z[:,i]) for i in range(norbits)])
    T_phi = np.array([gd.peak_to_peak_period(t, phi[:,i]) for i in range(norbits)])

    # quantize the periods and map on to notes
    q_z = quantize(T_z, nbins=len(midi_pool_hi), min=T_z.max(), max=T_z.min())
    q_phi = quantize(T_phi, nbins=len(midi_pool_lo), min=T_phi.max(), max=T_phi.min())

    delays = []
    notes = []
    Rphase = []
    for j in range(w.shape[0]):
        _no = []
        _ph = []
        for i in range(w.shape[1]):
            if j in z_cross[i]:
                _no.append(midi_pool_hi[q_z[i]])
                _ph.append(normed_R[j,i])

            if j in phi_cross[i]:
                _no.append(midi_pool_lo[q_phi[i]])
                _ph.append(normed_R[j,i])

        if len(_no) > 0:
            delays.append(t[j])
            notes.append(np.unique(_no).tolist())
            Rphase.append(_ph)

    delays = np.array(delays)
    notes = np.array(notes)
    Rphase = np.array(Rphase)

    return delays, notes, Rphase

def xyz_orbit_to_events(t, w, midi_pool):
    """
    Convert an orbit to MIDI events using Cartesian coordinates and rules.

    For Cartesian orbits...

    Parameters
    ----------
    t : array_like
    w : array_like
    midi_pool : array_like

    """

    x,y,z = w.T[:3]
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y,x)
    theta = np.arccos(z/r)

    # variable length arrays
    per = np.array([argrelmin(rr)[0] for rr in r])
    apo = np.array([argrelmax(rr)[0] for rr in r])

    # quantize the periods and map on to notes
    q_theta = quantize(theta, nbins=len(midi_pool))
    q_phi = quantize(phi, nbins=len(midi_pool))

    delays = []
    notes = []
    for j in range(w.shape[0]):
        _no = []
        for i in range(w.shape[1]):
            if j in per[i]:
                _no.append(midi_pool[q_theta[i,j]])

            if j in apo[i]:
                _no.append(midi_pool[q_phi[i,j]])

        if len(_no) > 0:
            delays.append(t[j])
            notes.append(np.unique(_no).tolist())

    delays = np.array(delays)
    notes = np.array(notes)

    return delays, notes

def halo_orbit_to_events(t, w, midi_pool):
    """
    Convert an orbit to MIDI events using Cartesian coordinates and rules.

    For Cartesian orbits...

    Parameters
    ----------
    t : array_like
    w : array_like
    midi_pool : array_like

    """

    x,y,z = w.T[:3]
    r = np.sqrt(x**2 + y**2 + z**2)

    # quantize the periods and map on to notes
    x_cross = np.array([argrelmin(xx**2)[0] for xx in x])
    y_cross = np.array([argrelmin(yy**2)[0] for yy in y])
    z_cross = np.array([argrelmin(zz**2)[0] for zz in z])

    q_r = quantize(r, nbins=len(midi_pool))

    delays = []
    notes = []
    for j in range(w.shape[0]):
        _no = []
        for i in range(w.shape[1]):
            if j in x_cross[i] or j in y_cross[i] or j in z_cross[i]:
                _no.append(midi_pool[q_r[i,j]])

        if len(_no) > 0:
            delays.append(t[j])
            notes.append(np.unique(_no).tolist())

    delays = np.array(delays)
    notes = np.array(notes)

    return delays, notes
