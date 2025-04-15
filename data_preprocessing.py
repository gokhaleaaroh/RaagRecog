import librosa
import os


yaman_units = []
darbari_units = []
kalavati_units = []
marwa_units = []

dir_dict = {
    "./Training_Data/Yaman/segments": yaman_units,
    "./Training_Data/Darbari/segments": darbari_units,
    "./Training_Data/Kalavati/segments": kalavati_units,
    "./Training_Data/Marwa/segments": marwa_units,
}

for dir_name in dir_dict.keys():
    files = [f for f in os.listdir(dir_name)
             if os.path.isfile(os.path.join(dir_name, f))]

    for file in files:
        audio, sr = librosa.load(file, sr=None)
