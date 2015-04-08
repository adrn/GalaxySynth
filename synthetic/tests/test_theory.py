# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np

# Project
from ..theory import MasterKey

def test_masterkey():
    mk = MasterKey(key='C', mode='dorian')
    print(mk.midi_notes)
    assert np.all(np.array(mk.notes) == np.array(['c', 'd', 'd#', 'f', 'g', 'a', 'a#']))

    mk = MasterKey(key='C', mode='dorian', octave=[4,5,6])
    print(mk.midi_notes)
    print(mk.notes)
