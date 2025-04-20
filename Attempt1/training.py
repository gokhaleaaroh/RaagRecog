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

total_samples = len(padded_seqs)
window_size = int(0.2 * total_samples)
results = []
val_iter = 1

for start in range(0, total_samples - window_size + 1, window_size):  # non-overlapping windows
    end = start + window_size

    perm = np.random.permutation(total_samples)

    padded_seqs_shuffled = padded_seqs[perm]
    labels_shuffled = one_hot_labels[perm]

    seq_tensor = torch.tensor(padded_seqs_shuffled, dtype=torch.long) + offset
    label_tensor = torch.tensor(labels_shuffled, dtype=torch.float32)

    # Split validation window
    s_val = seq_tensor[start:end]
    l_val = label_tensor[start:end]

    # Remaining is training
    s_train = torch.cat([seq_tensor[:start], seq_tensor[end:]], dim=0)
    l_train = torch.cat([label_tensor[:start], label_tensor[end:]], dim=0)

    train_loader = DataLoader(TensorDataset(s_train, l_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(s_val, l_val), batch_size=32)

    # Reinitialize model, optimizer for each run
    recognizer = RaagRecog(vocab_size, EMBEDDING_DIM, hidden_dim=64, fc_dim=64,
                           num_classes=4, padding_idx=0)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(recognizer.parameters(), lr=0.001)

    num_epochs = 300
    for epoch in range(num_epochs):
        recognizer.train()
        curr_loss = 0.0

        optimizer.zero_grad()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = recognizer(inputs)

            loss = loss_func(outputs, targets)
            loss.backward()
            optimizer.step()
            curr_loss += loss.item()
            avg_loss = curr_loss/len(train_loader)
        if ((epoch + 1) % 100 == 0):
            print(f"Validation Iteration: {val_iter}, Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    val_iter += 1
    recognizer.eval()
    correct, total = 0, 0
    yaman_correct, yaman_total = 0, 0
    darbari_correct, darbari_total = 0, 0
    marwa_correct, marwa_total = 0, 0
    kalavati_correct, kalavati_total = 0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = recognizer(inputs)
            preds = torch.argmax(outputs, dim=1)
            labels = torch.argmax(targets, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            yaman_correct += ((preds == labels) & (preds == 0)).sum().item()
            yaman_total += (preds == 0).sum().item()

            darbari_correct += ((preds == labels) & (preds == 1)).sum().item()
            darbari_total += (preds == 1).sum().item()

            marwa_correct += ((preds == labels) & (preds == 2)).sum().item()
            marwa_total += (preds == 2).sum().item()

            kalavati_correct += ((preds == labels) & (preds == 3)).sum().item()
            kalavati_total += (preds == 3).sum().item()

        acc = correct / total
        yaman_acc = yaman_correct / yaman_total
        darbari_acc = darbari_correct / darbari_total
        marwa_acc = marwa_correct / marwa_total
        kalavati_acc = kalavati_correct / kalavati_total
        results.append(acc)
        print(f"Validation window {start}:{end}, Accuracy = {acc:.4f}")
        print("Printing Individual Raag Accuracies...")
        print(f"Validation window {start}:{end}, Yaman Accuracy = {yaman_acc:.4f}")
        print(f"Validation window {start}:{end}, Darbari Accuracy = {darbari_acc:.4f}")
        print(f"Validation window {start}:{end}, Marwa Accuracy = {marwa_acc:.4f}")
        print(f"Validation window {start}:{end}, Kalavati Accuracy = {kalavati_acc:.4f}")

avg_acc = sum(results) / len(results)
print(f"\nAverage cross-validation accuracy = {avg_acc:.4f}")
