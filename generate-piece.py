import torch
import torch.nn as nn
import torch.nn.functional as nnf

class VoiceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        input_size = 4
        hidden_size = 4
        output_size = 4

        self.rnn = nn.RNN(input_size, hidden_size, num_layers=2, dropout=1e-6)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hx=None):
        x, hx = self.rnn(x, hx)
        # x : Tensor (seq_len, hidden_size)

        x = self.tanh(x)

        out = self.fc1(x)
        # x : Tensor (1, output_size)


        return out, hx

import re

def voice_tensor_from_line(line):
    match = re.match(r'(\w):\[(.*?)\]', line)
    if match:
        instrument = match.group(1)
        tuples_part = match.group(2)

        tuples = tuples_part.split('), (')
        timestamp_tensors = []

        for t in tuples:
            t = t.strip('()')
            pitch, start, duration = t.split(', ')

            pitch = float(pitch)
            start = float(start)
            duration = float(duration)
            instrument = float(instrument)

            timestamp_tensors.append(torch.tensor([pitch, start, duration, instrument]))

        return torch.stack(timestamp_tensors)
    else:
        raise RuntimeError("not a voice data line")


def load_from_file(path):
    contents = None

    with open(path, "r") as file:
        contents = file.read()
    
    lines = []
    for line in contents.split("\n"):
        if line.strip():
            lines.append(line)

    voice_tensors = []
    
    for line in lines:
        voice_tensors.append(voice_tensor_from_line(line))

    shapes = []

    for t in voice_tensors:
        shapes.append(t.shape)

    max_seq_len = 0

    for shape in shapes:
        if shape[0] > max_seq_len:
            max_seq_len = shape[0]

    print(max_seq_len)

    padded_tensors = []

    for tensor in voice_tensors:
        pad = (0, 0, 0, max_seq_len - tensor.shape[0])
        print(pad)
        padded_tensor = nnf.pad(tensor, pad, value=-1.)
        padded_tensors.append(padded_tensor)

    piece_tensor = torch.stack(padded_tensors)

    return piece_tensor




import sys
import os

data_directory = sys.argv[1]
data_directory_files = os.listdir(data_directory)

data_file_paths = []

for file in data_directory_files:
    if file.endswith(".dat"):
        path = os.path.join(data_directory, file)

        data_file_paths.append(path)

print(data_file_paths)

for path in data_file_paths:
    data_tensor = load_from_file(path)

    print(data_tensor, data_tensor.shape)

    model = VoiceModel()

    out, hidden = model(data_tensor)
    print(out, out.shape)
    print(hidden, hidden.shape)





