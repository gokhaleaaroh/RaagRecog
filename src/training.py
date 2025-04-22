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


def pad_mels(sequences, max_width, max_height):
    padded_mels = []
    for s in sequences:
        pad_height = (max_height - len(s))
        pad_width = (max_width - len(s[0]))
        padded_s = np.pad(np.array(s), ((0, pad_height), (0, pad_width), ), mode='constant', constant_values=0)
        padded_mels.append(padded_s)

    return np.stack(padded_mels)


def one_hot_encode(labels, num_classes=4):
    return np.eye(num_classes)[labels]


# Example usage:
yaman_units = load_pitch_units('../Training_Data/Yaman/segments/mels.pkl')
darbari_units = load_pitch_units('../Training_Data/Darbari/segments/mels.pkl')
marwa_units = load_pitch_units('../Training_Data/Marwa/segments/mels.pkl')
kalavati_units = load_pitch_units('../Training_Data/Kalavati/segments/mels.pkl')

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

max_height = max([len(seq) for seq in sequences])
max_width = max([len(seq[0]) for seq in sequences])

padded_mels = pad_mels(sequences, max_width, max_height)
one_hot_labels = one_hot_encode(labels)

total_samples = len(padded_mels)
window_size = int(0.2 * total_samples)
results = []
val_iter = 1

perm = np.random.permutation(total_samples)

padded_mels_shuffled = padded_mels[perm]
labels_shuffled = one_hot_labels[perm]

seq_tensor = torch.tensor(padded_mels_shuffled, dtype=torch.float32)
seq_tensor = seq_tensor.unsqueeze(1)
label_tensor = torch.tensor(labels_shuffled, dtype=torch.float32)


for start in range(0, total_samples - window_size + 1, window_size):  # non-overlapping windows
    end = start + window_size

    # Split validation window
    s_val = seq_tensor[start:end]
    l_val = label_tensor[start:end]

    # Remaining is training
    s_train = torch.cat([seq_tensor[:start], seq_tensor[end:]], dim=0)
    l_train = torch.cat([label_tensor[:start], label_tensor[end:]], dim=0)

    train_loader = DataLoader(TensorDataset(s_train, l_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(s_val, l_val), batch_size=16)

    # Reinitialize model, optimizer for each run
    recognizer = RaagRecog(32, 4)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(recognizer.parameters(), lr=0.0001)
    recognizer.train()

    num_epochs = 1000
    for epoch in range(num_epochs):
        curr_loss = 0.0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = recognizer(inputs)
            loss = loss_func(outputs, targets)
            loss.backward()
            optimizer.step()
            curr_loss += loss.item()

        avg_loss = curr_loss/len(train_loader)


        if (epoch + 1) % 50 == 0:
            print(f"Validation Iteration: {val_iter}, Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

            val_iter += 1
            recognizer.eval()
            correct, total = 0, 0
            yaman_correct, yaman_total = 0, 0
            darbari_correct, darbari_total = 0, 0
            marwa_correct, marwa_total = 0, 0
            kalavati_correct, kalavati_total = 0, 0
            with torch.no_grad():
                for inputs_val, targets_val in val_loader:
                    outputs_val = recognizer(inputs_val)
                    preds = torch.argmax(outputs_val, dim=1)
                    labels = torch.argmax(targets_val, dim=1)
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
                yaman_acc = yaman_correct / yaman_total if yaman_total != 0 else 0
                darbari_acc = darbari_correct / darbari_total if darbari_total != 0 else 0
                marwa_acc = marwa_correct / marwa_total if marwa_total != 0 else 0
                kalavati_acc = kalavati_correct / kalavati_total if kalavati_total != 0 else 0
                results.append(acc)
                print(f"Validation window {start}:{end}, Accuracy = {acc:.4f}")
                print("Printing Individual Raag Accuracies...")
                print(f"Validation window {start}:{end}, Yaman Accuracy = {yaman_acc:.4f}")
                print(f"Validation window {start}:{end}, Darbari Accuracy = {darbari_acc:.4f}")
                print(f"Validation window {start}:{end}, Marwa Accuracy = {marwa_acc:.4f}")
                print(f"Validation window {start}:{end}, Kalavati Accuracy = {kalavati_acc:.4f}")

                torch.save(recognizer.state_dict(), f'model_weights{epoch + 1}.pth')

            recognizer.train()


avg_acc = sum(results) / len(results)
print(f"\nAverage cross-validation accuracy = {avg_acc:.4f}")
