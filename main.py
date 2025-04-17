import os
from model import RaagRecog
import torch

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

    recognizer = RaagRecog(vocab_size, EMBEDDING_DIM, hidden_dim=64,
                           fc_dim=64, num_classes=4, padding_idx=0)

    recognizer.load_state_dict(torch.load("model_weights.pth"))
    recognizer.eval()
