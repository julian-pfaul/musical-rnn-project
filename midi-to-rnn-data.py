import sys
import os

source_directory = sys.argv[1]
destination_directory = sys.argv[2]    

import pretty_midi

def convert_midi_to_rnn_data(midi_path, data_path):
    midi_data = pretty_midi.PrettyMIDI(midi_path)

    piece = []

    for instrument in midi_data.instruments:
        voice = []

        for note in instrument.notes:
            voice.append((note.pitch, note.start, note.get_duration()))

        piece.append((voice, instrument.program))

    for voice, instrument in piece:
        voice = sorted(voice, key=lambda x: x[1])

    file = open(data_path, "w")

    for voice, instrument in piece:
        file.write(f"{instrument}:{voice}\n")

source_directory_files = os.listdir(source_directory)

os.makedirs(destination_directory, exist_ok=True)

for index, source_file in enumerate(source_directory_files):
    if source_file.endswith(".mid"):
        destination_file = source_file[:-4] + ".dat"

        source_file_path = os.path.join(source_directory, source_file)
        destination_file_path = os.path.join(destination_directory, destination_file)

        try:
            convert_midi_to_rnn_data(source_file_path, destination_file_path)
            
            if index % 50 == 0:
                print(f"[{index}/{len(source_directory_files)}]", end="\n")
        except Exception as e:
            print(f"[{index}/{len(source_directory_files)}] failure on {source_file_path}", end="\n")
            
