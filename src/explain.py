import os
import numpy as np
import matplotlib.pyplot as plt
import shap
from lime import lime_image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from skimage.segmentation import mark_boundaries

# Paths
model_path = "models/video_model.h5"
test_folder = "data/test_images"
results_folder = "results/visualizations"

os.makedirs(results_folder, exist_ok=True)

# Load trained model
model = load_model(model_path)
print(f"✅ Model loaded: {model_path}")

# Load test images
image_files = [f for f in os.listdir(test_folder) if f.endswith(".jpeg") or f.endswith(".png")]

# SHAP Explainer
explainer_shap = shap.GradientExplainer(model, np.zeros((1, 150, 150, 3)))

# Loop through images
for img_file in image_files:
    img_path = os.path.join(test_folder, img_file)
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array_exp = np.expand_dims(img_array, axis=0)

    # SHAP explanation
    shap_values = explainer_shap.shap_values(img_array_exp)
    shap_image = shap_values[0][0]
    plt.imshow(img_array)
    plt.imshow(shap_image, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.title(f"SHAP Explanation: {img_file}")
    plt.savefig(os.path.join(results_folder, f"shap_{img_file}"))
    plt.close()

    # LIME explanation
    explainer_lime = lime_image.LimeImageExplainer()
    explanation = explainer_lime.explain_instance(
        img_array.astype('double'),
        classifier_fn=lambda x: model.predict(x),
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    plt.imshow(mark_boundaries(temp / 255.0, mask))
    plt.axis('off')
    plt.title(f"LIME Explanation: {img_file}")
    plt.savefig(os.path.join(results_folder, f"lime_{img_file}"))
    plt.close()

print("✅ Explanations generated for all test images!")
