import torch
import cv2
import numpy as np
import requests
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

# 1. SETUP: Download a sample X-ray image (Pneumonia case)
# We download this automatically so you don't have to search for one
img_url = "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/01E392EE-69F9-4E33-BFCE-E5C968654078.jpeg"
img_path = "sample_xray.jpg"

print(f"Downloading sample X-ray from {img_url}...")
with open(img_path, 'wb') as f:
    f.write(requests.get(img_url).content)
print("✅ Download complete.")

# 2. MODEL: Load a standard ResNet50
# In a real project, you would add: model.load_state_dict(torch.load("your_trained_medical_model.pth"))
print("Loading AI Model (ResNet50)...")
model = models.resnet50(pretrained=True)
model.eval()

# 3. PRE-PROCESSING: Prepare image for the AI
# Standard ImageNet normalization
rgb_img = cv2.imread(img_path, 1)[:, :, ::-1] # Convert BGR to RGB
rgb_img = np.float32(rgb_img) / 255
input_tensor = transforms.ToTensor()(rgb_img).unsqueeze(0)

# 4. XAI ENGINE: The "Grad-CAM" Magic
# We focus on the last convolutional layer (where the complex features are)
target_layers = [model.layer4[-1]]

# Construct the CAM object
cam = GradCAM(model=model, target_layers=target_layers)

# We want to explain the model's top prediction.
# If you wanted to see why it thinks "Pneumonia" specifically, you'd target that class index.
targets = None # None means "Explain the highest confidence class"

print("Generating Explainability Heatmap...")
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
grayscale_cam = grayscale_cam[0, :] # Take the first image in batch

# 5. VISUALIZATION: Overlay Heatmap on X-ray
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# Save the results
cv2.imwrite('1_original_xray.jpg', cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
cv2.imwrite('2_heatmap_overlay.jpg', cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

print("\nSUCCESS! 🚀")
print("Check your folder for:")
print("1. '1_original_xray.jpg' (The input)")
print("2. '2_heatmap_overlay.jpg' (The AI's focus area)")
print("\nIf the heatmap highlights the lung area, the model is looking at the right place!")