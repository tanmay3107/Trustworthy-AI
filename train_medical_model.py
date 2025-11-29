import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.cuda.amp import GradScaler, autocast
import time
import os

# 1. SETUP & CONFIGURATION
# We use 'r' before the string to handle Windows backslashes correctly
DATA_DIR = r"D:\datasets\chest x-ray" 
BATCH_SIZE = 32 
EPOCHS = 5      

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# 2. DATA AUGMENTATION
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # ResNet requires 224x224
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. SMART DATA LOADER
try:
    train_dir = os.path.join(DATA_DIR, 'train')
    
    # This automatically finds your 4 folders: covid19, normal, pneumonia, tubercuosis
    train_dataset = datasets.ImageFolder(train_dir, train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    class_names = train_dataset.classes
    num_classes = len(class_names) # <--- Will automatically set to 4
    
    print(f"✅ Data Loaded Successfully!")
    print(f"   Classes Found ({num_classes}): {class_names}")
    print(f"   Total Images: {len(train_dataset)}")
    
except FileNotFoundError:
    print(f"❌ Error: Could not find folder '{train_dir}'")
    print("Please check your path.")
    exit()

# 4. MODEL SETUP
print("Loading ResNet50...")
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features

# Adjust the final layer to match your 4 classes
model.fc = nn.Linear(num_ftrs, num_classes) 
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scaler = GradScaler()

# 5. TRAINING LOOP
print(f"\n💪 Starting Training for {EPOCHS} Epochs...")
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed Precision (Faster on RTX 3050Ti)
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct / total
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc*100:.2f}%")

total_time = time.time() - start_time
print(f"\n✅ Training Complete in {total_time/60:.1f} minutes.")

# 6. SAVE
save_name = "medical_resnet_4class.pth"
torch.save(model.state_dict(), save_name)
print(f"💾 Model saved as '{save_name}'")