import sys
import os

path = sys.argv[1]

soundfont_set = False
soundfont = None

if len(sys.argv) == 3:
    soundfont_set = True
    soundfont = sys.argv[2]

if not os.path.isfile(path):
    raise RuntimeError(f"{sys.argv[1]} is not a valid path")

import pretty_midi
import fluidsynth
import sounddevice

midi_data = pretty_midi.PrettyMIDI(path)

audio_data = None

if soundfont_set:
    audio_data = midi_data.fluidsynth(sf2_path=soundfont)
else:
    audio_data = midi_data.synthesize()

sounddevice.play(audio_data)
sounddevice.wait()
