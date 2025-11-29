import torch
import cv2
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import os

# 1. SETUP
# Point this to an image you want to test (e.g., a TB or Pneumonia image from your dataset)
# Example: r"D:\datasets\chest x-ray\test\pneumonia\person1_virus_6.jpeg"
IMG_PATH = r"D:\datasets\chest x-ray\test\COVID19\COVID19(479).jpg" 

# Your saved model file
MODEL_PATH = "medical_resnet_4class.pth"

# These must match your folder names alphabetically!
# The user provided: covid19, normal, pneumonia, tubercuosis
CLASSES = ['covid19', 'normal', 'pneumonia', 'tubercuosis']

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. LOAD THE 4-CLASS MODEL
print(f"Loading {MODEL_PATH}...")
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4) # <--- UPDATED for 4 Classes
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model = model.to(device)

# 3. PREPARE IMAGE
# We need to grab a sample image if you haven't set one
if IMG_PATH == "sample_image.jpg" and not os.path.exists(IMG_PATH):
    print("⚠️ Please set IMG_PATH to a real image from your dataset!")
    print("For now, I'll try to download a sample Pneumonia X-ray...")
    import requests
    url = "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/01E392EE-69F9-4E33-BFCE-E5C968654078.jpeg"
    with open("sample_image.jpg", 'wb') as f:
        f.write(requests.get(url).content)
    print("✅ Downloaded sample_image.jpg")

img = cv2.imread(IMG_PATH)[:, :, ::-1] # BGR -> RGB
img = cv2.resize(img, (224, 224))
img = np.float32(img) / 255
input_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

# 4. GET PREDICTION (What does the AI think it is?)
output = model(input_tensor)
prob = torch.nn.functional.softmax(output, dim=1)
score, index = torch.max(prob, 1)

prediction = CLASSES[index.item()]
confidence = score.item() * 100

print(f"\n🔍 AI Diagnosis: {prediction.upper()}")
print(f"📊 Confidence: {confidence:.2f}%")

# 5. GENERATE HEATMAP (Why does it think that?)
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# Generate heatmap for the predicted class
grayscale_cam = cam(input_tensor=input_tensor, targets=None) 
grayscale_cam = grayscale_cam[0, :]

# Overlay
visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)

# 6. SAVE RESULTS
output_name = f"result_{prediction}.jpg"
# Convert back to BGR for OpenCV saving
final_image = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_name, final_image)

print(f"\n✅ Explanation Saved: {output_name}")
print("Open this file to see where the AI detected the disease!")