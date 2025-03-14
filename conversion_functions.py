import torch
import pretty_midi

def convert_midi_to_tensor(midi_data):
    notes = []

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append([float(note.pitch), float(note.start), float(note.get_duration())])

    notes = sorted(notes, key=lambda x: x[1])

    data_tensor = torch.tensor(notes)           # shape [L, 3] <- L = len(notes)
    data_tensor = data_tensor.permute((0, 1))   # shape [3, L]

    return data_tensor

def convert_tensor_to_midi(input_tensor):
    midi_data = pretty_midi.PrettyMIDI()

    tensor = input_tensor.permute((0, 1))

    instrument = pretty_midi.Instrument(program=0)

    for tensor_note in tensor:
        midi_note = pretty_midi.Note(pitch=tensor_note[0], start=tensor_note[1], end=tensor_note[1] + tensor_note[2])
        instrument.append(midi_note)

    midi_data.instruments.append(instrument)
    
    return midi_data            
