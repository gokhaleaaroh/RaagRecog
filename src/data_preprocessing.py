import librosa
import os
import numpy as np
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

dir_list = [
    # "../Training_Data/Yaman/segments",
    # "../Training_Data/Darbari/segments",
    # "../Training_Data/Kalavati/segments",
    # "../Training_Data/Marwa/segments",
    "../Training_Data/Yaman-discard/segments",
]


def hz_to_midi_token(f0):
    midi = librosa.hz_to_midi(f0)
    midi[~np.isfinite(midi)] = -1  # Use -1 or some special value for silence
    return midi.astype(int)


def relative_pitch(tokens):
    valid = tokens[tokens >= 0]
    reference = np.median(valid)
    rel = tokens - reference
    rel[tokens == -1] = -100  # special value for silence
    return rel


# for dir_name in dir_dict.keys():
#     files = [f for f in os.listdir(dir_name)
#              if os.path.isfile(os.path.join(dir_name, f))]
# 
#     num = 0
#     for file_name in files:
#         audio, sr = librosa.load(f"{dir_name}/{file_name}", sr=None)
#         f0, voiced_flag, voiced_prob = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
#         tokens = hz_to_midi_token(f0)
#         normalized_tokens = relative_pitch(tokens)
#         dir_dict[dir_name].append(normalized_tokens)
#         if num % 10 == 0:
#             print(f"Done with {file_name}")
#         num += 1
# 
#     output_file = f"{dir_name}/units.pkl"
#     with open(output_file, "wb") as f:
#         pickle.dump(dir_dict[dir_name], f)


def gen_mel(path):
    audio, sr = librosa.load(path)  # sr is the sampling rate
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB


def parallel_preprocess(directory, num_workers):
    files = list(Path(directory).glob("*.wav"))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(gen_mel, files)

    return list(results)


for dir_name in dir_list:
    data = parallel_preprocess(dir_name, 8)
    output_file = f"{dir_name}/mels.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(data, f)
