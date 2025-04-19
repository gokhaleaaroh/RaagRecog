import librosa
import os
import numpy as np
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

dir_list = [
    # "./Training_Data/Yaman/segments",
    # "./Training_Data/Darbari/segments",
    # "./Training_Data/Kalavati/segments",
    # "./Training_Data/Marwa/segments",
    "./Training_Data/Yaman-discard/segments",
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


def process_clip(path):
    audio, sr = librosa.load(path, sr=None)
    f0, voiced_flag, voiced_prob = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    tokens = hz_to_midi_token(f0)
    normalized_tokens = relative_pitch(tokens)
    return normalized_tokens


def parallel_preprocess(directory, num_workers):
    files = list(Path(directory).glob("*.wav"))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(process_clip, files)

    return list(results)


for dir_name in dir_list:
    data = parallel_preprocess(dir_name, 8)
    output_file = f"{dir_name}/units.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(data, f)


'''
import pygame.midi
import time

pygame.midi.init()
player = pygame.midi.Output(0)
player.set_instrument(0)

for note in tokens:
    # player.note_on(note, 100)
    # time.sleep(0.2)
    # player.note_off(note, 100)
    print(note)
'''

# audio, sr = librosa.load("./Training_Data/Yaman/segments/yaman-rajurkar_part35.wav", sr=None)
# f0, voiced_flag, voiced_prob = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
# tokens = hz_to_midi_token(f0)
# 
# print("Created midi!")
# 
# import time
# import fluidsynth
# 
# fs = fluidsynth.Synth()
# fs.start()  # defaults to pulseaudio or alsa depending on your system
# 
# sfid = fs.sfload("/usr/share/soundfonts/FluidR3_GM.sf2")  # adjust path if needed
# fs.program_select(0, sfid, 0, 0)
# 
# for note in tokens:
#     fs.noteon(0, note, 100)
#     time.sleep(0.2)
#     fs.noteoff(0, note)
# 
# fs.delete()
