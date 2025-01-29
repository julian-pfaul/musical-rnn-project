import io
import numpy as np
import sounddevice as sd

VOLUME_SCALER = 0.5 # half max volume

def load_data(path):
    lines = io.open(path, encoding='utf8').read().strip().split()
    notes = []

    for hex_string in lines:
        notes.append(int(hex_string.upper(), 16))

    return notes


def midi_to_frequency(midi_note):
    return 440.0 * 2 ** ((midi_note - 69) / 12.0)


def generate_sine_wave(frequency, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t) * VOLUME_SCALER
    return wave


def play_midi_note(midi_note, duration=1.0):
    print(midi_note, end=' ', flush=True)
    frequency = midi_to_frequency(midi_note)
    wave = generate_sine_wave(frequency, duration)
    sd.play(wave, samplerate=44100)
    sd.wait()

notes = load_data("data/input.txt")

for note in notes:
    play_midi_note(note, 1.0f)