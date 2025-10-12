import numpy as np
import os

# Example: 0 = real, 1 = deepfake
labels_test = np.array([0, 1, 0, 1])  # replace with your actual test labels
os.makedirs("data/test", exist_ok=True)
np.save("data/test/labels_test.npy", labels_test)
print("✅ labels_test.npy created successfully!")
