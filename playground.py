import torch
import torch.nn as nn
import torch.optim as optim

class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.rnn = nn.RNN(1, 4)
        self.fc = nn.Linear(4, 1)

    def forward(self, x, h0):
        out, hn = self.rnn(x, h0)
        print(f"in forward: out: {out} hn: {hn}")

        out = self.fc(out)
        print(f"in forward: out: {out}")

        return out[-1], hn

import os

file_path = "playground/model.dat"

model = Model()

if os.path.exists(file_path):
    model.load_state_dict(torch.load(file_path, weights_only=True))

import numpy

num_epochs = 1000

numpy_array = numpy.arange(0, 200) * numpy.pi / 32
numpy_array = numpy.sin(numpy_array)

training_split = numpy_array.size - 50
training_data   = torch.from_numpy(numpy_array[:-training_split]).clone().unsqueeze(dim=1).type(torch.float32).detach()
validation_data = torch.from_numpy(numpy_array[0:training_split]).clone().unsqueeze(dim=1).type(torch.float32).detach()

hn = None # hidden state

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

import random

training_seq_len = 4

for epoch in range(num_epochs):
    print("EPOCH")
    start_index = random.randint(0, training_data.shape[0] - 5)

    inputs = training_data[start_index:start_index + 4].clone().detach()
    labels = training_data[start_index + 4].clone().detach()

    inputs.detach()
    labels.detach()

    model.train()

    with torch.autograd.set_detect_anomaly(True):
        optimizer.zero_grad()

        outputs, hn = model(inputs, hn)
        hn.detach()

     #   print(f"inputs: {inputs}\noutputs: {outputs}\nhn: {hn}\nlabels: {labels}\n")

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
    
    model.eval()

    #if epoch % 10:
    #print(f"[{epoch}/{num_epochs}] {loss.item()}")

torch.save(model.state_dict(), file_path)
