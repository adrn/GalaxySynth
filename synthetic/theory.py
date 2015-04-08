# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
import pyo

__all__ = []

_all_notes_sharp = ['c','c#','d','d#','e','f','f#','g','g#','a','a#','b']
_all_notes_flat = ['c','db','d','eb','e','f','gb','g','ab','a','bb','b']

# mode name to index sequence
mode_map = dict(ionian=[0,2,4,5,7,9,11],
                dorian=[0,2,3,5,7,9,10],
                phrygian=[0,1,3,5,7,8,10],
                lydian=[0,2,4,6,7,9,11],
                mixolydian=[0,2,4,5,7,9,10],
                aeolian=[0,2,3,5,7,8,10],
                locrian=[0,1,3,5,6,8,10])

def _note_to_midi(note, octave):
    try:
        base_ix = _all_notes_sharp.index(note.lower())
    except ValueError:
        try:
            base_ix = _all_notes_flat.index(note.lower())
        except ValueError:
            raise ValueError("Note '{0}' doesn't exist.".format(note))

    # convert to a MIDI note number
    note_num = base_ix + octave*12
    return note_num

def _midi_to_note(midi):
    return _all_notes_sharp[(midi % 12)]

class MasterKey(object):

    def __init__(self, key, mode, octave=4):

        if mode.lower() not in mode_map.keys():
            raise NotImplementedError("Currently only support the modes: {0}"
                                      .format(",".join(mode_map.keys())))
        self.mode = mode.lower()

        # parse key - note
        self.root_note = key.lower()
        self._root_midi = _note_to_midi(self.root_note, octave)
        self.octave = int(octave)

        # get valid notes
        self.midi_notes = self.midi_scale(self.root_note, self.mode, self.octave)
        self.notes = [_midi_to_note(x) for x in self.midi_notes]

    @classmethod
    def midi_scale(cls, note, mode, octave):
        offset = _note_to_midi(note, octave)
        return offset + np.array(mode_map[mode])

    def chord(self, xxx):
        # TODO: pass in, e.g., 'maj' or 'aug4' and get chord from scale?
        pass
