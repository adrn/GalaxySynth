# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from astropy import log as logger
import astropy.units as u

__all__ = ['Scale']

_all_notes = np.array(['c','c#','d','d#','e','f','f#','g','g#','a','a#','b'])

# key to frequency
key_map4 = {'c': 261.63, 'c#': 277.18,
            'd': 293.66, 'd#': 311.13,
            'e': 329.63,
            'f': 349.23, 'f#':369.99,
            'g': 392.00, 'g#': 415.3,
            'a': 440.00, 'a#': 466.16,
            'b': 493.88}
key_map4['db'] = key_map4['c#']
key_map4['eb'] = key_map4['d#']
key_map4['gb'] = key_map4['f#']
key_map4['ab'] = key_map4['g#']
key_map4['bb'] = key_map4['a#']

# mode name to index sequence
mode_map = {'ionian': np.array([0,2,4,5,7,9,11]),
            'aeolian': np.array([0,2,3,5,7,8,10]),
            'mixolydian': np.array([0,2,4,5,7,9,10])}

class Scale(object):

    def __init__(self, notes=None, key=None, mode=None, octave=4):

        if notes is None and key is not None and mode is not None:

            if mode.lower() not in mode_map.keys():  # TODO: right now only support natural major/minor
                raise NotImplementedError("Currently only support {0}".format(",".join(mode_map.keys())))

            self.mode = mode.lower()

            # parse key - note
            self.root_note = key.lower()
            try:
                key_map4[self.root_note]
            except KeyError:
                raise ValueError("Invalid input key '{0}'".format(key))

            # get valid notes
            self.notes = self.get_scale(self.root_note, self.mode)

        elif notes is not None and key is None:
            self.notes = [n.lower() for n in notes]

            # assumes notes[0] is the root
            self.root_note = self.notes[0]

        else:
            # shouldn't get here if valid input
            raise ValueError("Either specify a sequency of (string) notes, or "
                             "an input key and scale name.")

        # convert note names to frequencies
        fdiv = 2**(octave - 4)
        self.freqs = np.array([key_map4[note] for note in self.notes]) * fdiv

        # for iterating
        self._i = 0

    @classmethod
    def get_scale(cls, note, mode):
        note_offset = np.where(_all_notes == note)[0][0]
        notes_rpt = np.append(_all_notes,_all_notes)
        return notes_rpt[mode_map[mode] + note_offset]

    def chord(self, xxx):
        # TODO: pass in, e.g., 'maj' or 'aug4' and get chord from scale?
        pass

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._i < len(self.freqs):
            f = self.freqs[self._i]
            self._i += 1
            return f
        else:
            raise StopIteration()
