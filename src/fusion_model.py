import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate
import numpy as np

# ==========================
# Load Pre-trained Video Model (.keras)
# ==========================
video_model_path = "results/video_model.keras"  # correct file path
video_model = load_model(video_model_path)
video_model.trainable = False  # freeze video model weights

# Wrap Sequential model using Functional API
video_input = Input(shape=(150, 150, 3), name="video_input")
video_features = video_model(video_input)  # call the Sequential model on input

# ==========================
# Build Audio Model (Placeholder)
# ==========================
# Example: audio features shape = (20, 44, 1) (e.g., MFCC or spectrogram)
audio_input = Input(shape=(20, 44, 1), name="audio_input")
x = Flatten()(audio_input)
x = Dense(64, activation='relu')(x)

# ==========================
# Combine Video + Audio Features
# ==========================
combined = Concatenate()([video_features, x])
combined = Dense(128, activation='relu')(combined)
final_output = Dense(1, activation='sigmoid')(combined)

fusion_model = Model(inputs=[video_input, audio_input], outputs=final_output)
fusion_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
fusion_model.summary()

# ==========================
# Dummy Data for Testing
# ==========================
# Replace these with actual features from your dataset
video_data = np.random.rand(10, 150, 150, 3)
audio_data = np.random.rand(10, 20, 44, 1)
labels = np.random.randint(0, 2, 10)

# ==========================
# Train Fusion Model
# ==========================
fusion_model.fit([video_data, audio_data], labels, epochs=5, batch_size=2)

# ==========================
# Save Fusion Model
# ==========================
fusion_model.save("results/fusion_model.keras")
print("✅ Fusion model saved successfully!")
