from model import RaagRecog
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.optim as optim


def load_pitch_units(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def pad_sequences(sequences, max_len, padding_value=-100):
    padded_sequences = []
    for seq in sequences:
        # Pad sequence to max_len
        padded_seq = np.pad(seq, (0, max_len - len(seq)), constant_values=padding_value)
        padded_sequences.append(padded_seq)
    return np.array(padded_sequences)


def one_hot_encode(labels, num_classes=4):
    return np.eye(num_classes)[labels]


# Example usage:
yaman_units = load_pitch_units('./Training_Data/Yaman/segments/units.pkl')
darbari_units = load_pitch_units('./Training_Data/Darbari/segments/units.pkl')
marwa_units = load_pitch_units('./Training_Data/Marwa/segments/units.pkl')
kalavati_units = load_pitch_units('./Training_Data/Kalavati/segments/units.pkl')


# Constants
PAD_VALUE = -100
MIN_VALUE = -99
MAX_VALUE = 127

# Calculate the range
vocab_size = MAX_VALUE - MIN_VALUE + 1  # Total possible MIDI values
EMBEDDING_DIM = 64
padding_idx = 0
offset = 100

sequences = []
labels = []

sequences.extend(yaman_units)
labels.extend([0] * len(yaman_units))

sequences.extend(darbari_units)
labels.extend([1] * len(darbari_units))

sequences.extend(marwa_units)
labels.extend([2] * len(marwa_units))

sequences.extend(kalavati_units)
labels.extend([3] * len(kalavati_units))

max_len = max(len(seq) for seq in sequences)

padded_seqs = pad_sequences(sequences, max_len)
one_hot_labels = one_hot_encode(labels)

seq_tensor = torch.tensor(padded_seqs, dtype=torch.long) + offset
label_tensor = torch.tensor(one_hot_labels, dtype=torch.float32)

dataset = TensorDataset(seq_tensor, label_tensor)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

recognizer = RaagRecog(vocab_size, EMBEDDING_DIM, hidden_dim=64, fc_dim=64,
                       num_classes=4, padding_idx=0)

loss_func = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(recognizer.parameters(), lr=0.001)

num_epochs = 1000
for epoch in range(num_epochs):
    recognizer.train()
    curr_loss = 0.0

    optimizer.zero_grad()
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        outputs = recognizer(inputs)

        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()
        curr_loss += loss.item()

    avg_loss = curr_loss/len(data_loader)

    if epoch % 50 == 0:
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
    if (epoch + 1) % 100 == 0:
        torch.save(recognizer.state_dict(), f"model_weights{epoch}.pth")


torch.save(recognizer.state_dict(), "model_weights.pth")
