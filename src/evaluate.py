import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# ==========================
# Load Fusion Model
# ==========================
fusion_model = load_model("results/fusion_model.keras")
print("✅ Fusion model loaded successfully!")

# ==========================
# Load / Prepare Test Data
# ==========================
# Replace these lines with your real test data
# video_test: shape (num_samples, 150, 150, 3)
# audio_test: shape (num_samples, 20, 44, 1)
# labels_test: shape (num_samples,)
video_test = np.load("data/test/video_test.npy")  # example path
audio_test = np.load("data/test/audio_test.npy")
labels_test = np.load("data/test/labels_test.npy")

# ==========================
# Predict
# ==========================
y_pred = fusion_model.predict([video_test, audio_test])
y_pred_classes = (y_pred > 0.5).astype(int)

# ==========================
# Metrics
# ==========================
accuracy = accuracy_score(labels_test, y_pred_classes)
print(f"Test Accuracy: {accuracy*100:.2f}%")

cm = confusion_matrix(labels_test, y_pred_classes)
print("Confusion Matrix:\n", cm)

# ==========================
# ROC Curve
# ==========================
fpr, tpr, thresholds = roc_curve(labels_test, y_pred)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(labels_test, y_pred):.2f}')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
