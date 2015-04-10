# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import matplotlib.pyplot as plt
import numpy as np

# Project
from ..orbitreducer import cyl_orbit_to_events

dt = 0.25
nsteps = 100

def test_cyl():
    print()
    hipool = [60, 62, 64, 65, 67, 69, 71]
    lopool = [36, 38, 40, 41, 43, 45, 47]

    import gary.potential as gp
    from gary.units import galactic
    w0 = np.array([[8.,0,0,0.,0.2,0.02],
                   [3.5,0,0,0.,0.22,-0.075]])

    pot = gp.MiyamotoNagaiPotential(m=2E12, a=6.5, b=.26, units=galactic)
    t,w = pot.integrate_orbit(w0, dt=dt, nsteps=nsteps)

    delays,notes = cyl_orbit_to_events(t, w, hipool, lopool)
    print(delays / dt)
    print(notes)
    print()

def test_cyl_time_resolution():
    print()
    hipool = [60, 62, 64, 65, 67, 69, 71]
    lopool = [36, 38, 40, 41, 43, 45, 47]

    import gary.potential as gp
    from gary.units import galactic
    w0 = np.array([[8.,0,0,0.,0.2,0.02],
                   [3.5,0,0,0.,0.22,-0.075]])

    pot = gp.MiyamotoNagaiPotential(m=2E12, a=6.5, b=.26, units=galactic)
    t,w = pot.integrate_orbit(w0, dt=dt, nsteps=nsteps)

    delays,notes = cyl_orbit_to_events(t, w, hipool, lopool, time_resolution=1. / dt)
    print(delays / dt)
    print(notes)
    print()
