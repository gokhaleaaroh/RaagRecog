import os
from model import RaagRecog
import torch
import pickle
import numpy as np


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


if os.path.exists("./model_weights.pth"):
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

    recognizer = RaagRecog(vocab_size, EMBEDDING_DIM, hidden_dim=64,
                           fc_dim=64, num_classes=4, padding_idx=0)

    print("Created Model!")
    recognizer.load_state_dict(torch.load("model_weights.pth"))
    recognizer.eval()

    yaman_units = load_pitch_units('./Training_Data/Yaman/segments/units.pkl')
    padded_yaman = pad_sequences(yaman_units, 1729)

    darbari_units = load_pitch_units('./Training_Data/Darbari/segments/units.pkl')
    padded_darbari = pad_sequences(darbari_units, 1729)

    marwa_units = load_pitch_units('./Training_Data/Marwa/segments/units.pkl')
    padded_marwa = pad_sequences(marwa_units, 1729)

    kalavati_units = load_pitch_units('./Training_Data/Kalavati/segments/units.pkl')
    padded_kalavati = pad_sequences(kalavati_units, 1729)

    input_tensor = torch.tensor(padded_marwa, dtype=torch.long) + offset

    with torch.no_grad():
        output = recognizer(input_tensor)
        total_correct = 0
        for out in output:
            if torch.argmax(out) == 2:
                total_correct += 1
        print(f"Percent Correct: {total_correct/len(output)}")
