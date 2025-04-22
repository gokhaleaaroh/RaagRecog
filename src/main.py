import os
from model import RaagRecog
import torch
import pickle
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


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


if os.path.exists("./model_weights200.pth"):
    # Constants
    PAD_VALUE = -100
    MIN_VALUE = -99
    MAX_VALUE = 127

    # Calculate the range
    vocab_size = MAX_VALUE - MIN_VALUE + 1  # Total possible MIDI values
    EMBEDDING_DIM = 64
    padding_idx = 0
    offset = 100

    print("Set value!")

    recognizer = RaagRecog(32, 4)
    print("Created Model!")
    recognizer.load_state_dict(torch.load("model_weights250.pth"))
    recognizer.eval()
#     yaman_units = load_pitch_units('./Training_Data/Yaman/segments/units.pkl')
#     padded_yaman = pad_sequences(yaman_units, 1729)
# 
#     darbari_units = load_pitch_units('./Training_Data/Darbari/segments/units.pkl')
#     padded_darbari = pad_sequences(darbari_units, 1729)
# 
#     marwa_units = load_pitch_units('./Training_Data/Marwa/segments/units.pkl')
#     padded_marwa = pad_sequences(marwa_units, 1729)
# 
#     kalavati_units = load_pitch_units('./Training_Data/Kalavati/segments/units.pkl')
#     padded_kalavati = pad_sequences(kalavati_units, 1729)

    yaman_discard_units = load_pitch_units('../Training_Data/Yaman-discard/segments/mels.pkl')
    max_height = max([len(seq) for seq in yaman_discard_units])
    max_width = max([len(seq[0]) for seq in yaman_discard_units])
    padded_discard_mels = pad_mels(yaman_discard_units, max_width, max_height)

    input_tensor = torch.tensor(padded_discard_mels, dtype=torch.float32)
    input_tensor = input_tensor.unsqueeze(1)
    raag_list = ["Yaman", "Darbari", "Marwa", "Kalavati"]

    train_loader = DataLoader(input_tensor, batch_size=16, shuffle=True)

    with torch.no_grad():
        total_correct = 0
        for inputs in train_loader:
            outputs = recognizer(inputs)
            percent_array = np.array([0.0, 0.0, 0.0, 0.0])
            for ind in range(len(outputs)):
                if (torch.argmax(outputs[ind]) != 0):
                    answer = torch.argmax(outputs[ind])
                    # print(ind,
                     #     " ", raag_list[answer], "confidence: ", outputs[ind])
                else:
                    total_correct+=1

        print(f"Accuracy: {total_correct/len(input_tensor)}")

            
