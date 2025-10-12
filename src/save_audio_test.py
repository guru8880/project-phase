import numpy as np
import os
import librosa

audio_folder = "data/test_audios"
audio_test_list = []

for audio_file in os.listdir(audio_folder):
    if not audio_file.endswith(".wav"):
        continue
    audio_path = os.path.join(audio_folder, audio_file)
    y, sr = librosa.load(audio_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    audio_test_list.append(mfccs)

audio_test = np.array(audio_test_list)
os.makedirs("data/test", exist_ok=True)
np.save("data/test/audio_test.npy", audio_test)
print("✅ audio_test.npy created successfully!")
