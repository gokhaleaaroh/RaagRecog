import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from model import RaagRecog
import torch
import torch.nn

# Load audio file
y, sr = librosa.load('../Training_Data/Yaman/segments/yaman-rajurkar_part35.wav')  # sr is the sampling rate

# Compute Mel spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

# Convert to dB scale (log scale)
S_dB = librosa.power_to_db(S, ref=np.max)

# Display
# librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel-frequency spectrogram')
# plt.tight_layout()
# plt.show()

recog = RaagRecog(128, 4)
recog.eval()

with torch.no_grad():
    output = recog(torch.tensor(np.array([S_dB])))
    print(output)
