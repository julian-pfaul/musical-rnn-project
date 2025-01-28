import io
import numpy as np
import sounddevice as sd

VOLUME_SCALER = 0.5 # half max volume

def load_data(path):
    lines = io.open(path, encoding='utf8').read().strip().split('\n')
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


#for note in notes:
    #play_midi_note(note, 0.15)

###
###
###

import torch
import torch.nn as nn
import torch.optim as optim

# Generate a simple dataset of musical notes (e.g., numbers 0-11 representing notes)
# For example, a sequence of notes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
sample_notes = load_data("data/out0")

#for note in notes:
#    print(note, end=' ')
#    play_midi_note(note, 0.5)


def create_sequences(notes, seq_length):
    input_sequences = []
    target_notes = []

    for i in range(len(notes) - seq_length):
        input_sequences.append(notes[i:i + seq_length])
        target_notes.append(notes[i + seq_length])

    return np.array(input_sequences), np.array(target_notes)


seq_length = 3
X, y = create_sequences(sample_notes, seq_length)

X = torch.FloatTensor(X).view(-1, seq_length, 1)
y = torch.LongTensor(y)

hidden_size = 20
num_layers = 3

class MusicalRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=hidden_size, output_size=255, num_layers=num_layers):
        super(MusicalRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.relu(out)
        out = self.fc(out[:, -1, :])  # Get the last time step

        return out, hidden


# Initialize the model, loss function, and optimizer
model = MusicalRNN()

###
###
###

# Initialize the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), eps=1e-5, lr=1e-4, weight_decay=1e-8)

# Training parameters
num_epochs = 5000
# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Zero the gradients

    # Get the batch size from the input tensor
    batch_size = seq_length

    # Initialize the hidden state with the correct batch size
    hidden_state = torch.zeros(num_layers, batch_size, hidden_size)  # Shape: (num_layers, batch_size, hidden_size)

    # Forward pass with the hidden state
    outputs, hidden_state = model(X, hidden_state)
    loss = criterion(outputs, y)  # Calculate the loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update the weights

    # Detach the hidden state to prevent backpropagation through the entire history
    hidden_state = hidden_state.detach()

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation phase
model.eval()  # Set the model to evaluation mode

# Generate a sequence of notes
generated_notes = sample_notes[0:seq_length]  # Start with the first sequence of notes
current_input = torch.FloatTensor([generated_notes]).view(-1, seq_length, 1)  # Shape: (batch_size, seq_length, input_size)

batch_size = seq_length

# Reset hidden state for generation
hidden_state = torch.zeros(num_layers, batch_size, hidden_size)  # Reset for batch size of 1

# Generate notes
for _ in range(100):  # Generate 50 new notes
    prediction, hidden_state = model(current_input, hidden_state)  # Get the model's prediction
    predicted_note = torch.argmax(prediction).item()  # Get the index of the highest probability

    if predicted_note == generated_notes[-1]:
        continue

    generated_notes.append(predicted_note)  # Append the predicted note to the generated notes

    # Prepare the next input for the model
    current_input = torch.cat((current_input[:, 1:, :], torch.FloatTensor([[predicted_note]]).view(-1, 1, 1)), dim=1)

# Print the generated notes
print("Generated Notes:", generated_notes)

print("\nSameple Notes:", end="\n\n")

for note in sample_notes[:32]:
    play_midi_note(note, duration=0.15)

print("\nGenerated Notes:", end="\n\n")

for note in generated_notes:
    play_midi_note(note, duration=0.2)
