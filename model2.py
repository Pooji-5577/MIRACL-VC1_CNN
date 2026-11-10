#part1-imports
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

#part2-data(5 steps)
#step-1 Speaker folders	Recording folder locations
clip_paths = []
clip_labels = []
clips = []
DATA_DIR = "dataset"
speakers=sorted(os.listdir(DATA_DIR))
# mouth centre (x, y) of each speaker in the 640x480 picture
mouth_center = {
    "F01": (242, 368), "F02": (287, 373), "F04": (268, 285), "F05": (258, 315),
    "F06": (267, 300), "F07": (245, 280), "F08": (255, 328), "F09": (257, 318),
    "F10": (298, 345), "F11": (290, 350), "M01": (278, 275), "M02": (355, 252),
    "M04": (345, 302), "M07": (262, 332), "M08": (302, 342)}
CROP_W = 200
CROP_H = 125
for speaker in speakers:
    if speaker.endswith(".txt"):
        continue
    for category in ["words", "phrases"]:
        category_path = os.path.join(DATA_DIR, speaker, category)
        labels = sorted(os.listdir(category_path))
        for label in labels:
            label_path = os.path.join(category_path, label)
            repetitions = sorted(os.listdir(label_path))
            for repetition in repetitions:
                repetition_path = os.path.join(label_path, repetition) 
                frames = sorted([f for f in os.listdir(repetition_path)if f.startswith("color")])
                frames_data = []
                for frame in frames:
                    frame_path = os.path.join(repetition_path, frame)
#step2 Reading images and converting them into grayscale
                    image = cv2.imread(frame_path)
                    cx, cy = mouth_center[speaker]
                    left = max(0, min(640 - CROP_W, cx - CROP_W // 2))
                    top = max(0, min(480 - CROP_H, cy - CROP_H // 2))
                    image = image[top:top + CROP_H, left:left + CROP_W]
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image = cv2.resize(image, (48, 30))
                    image = image.astype(np.float32)
                    image = image / 255.0
                    frames_data.append(image)
                clip = np.array(frames_data)
                 
#step 3 adding dummy for frames less than 15 
                if len(clip) == 0:
                    continue
                elif len(clip) > 15:
                    indices = np.linspace(0, len(clip)-1, 15).astype(int)
                    clip = clip[indices]

                elif len(clip) < 15:
                    padding = 15 - len(clip)
                    last_frame = clip[-1]
                    extra_frames = np.repeat(last_frame[np.newaxis, :, :], padding, axis=0)
                    clip = np.concatenate((clip, extra_frames), axis=0)
                clips.append(clip)
                clip_paths.append(repetition_path)
                clip_labels.append(category + "_" + label)   
#step4  convert word and phrases labels into numbers  
label_names = sorted(set(clip_labels))
label_to_number = {}

for i, label in enumerate(label_names):
    label_to_number[label] = i

numeric_labels = []

for label in clip_labels:
    numeric_labels.append(label_to_number[label])

#step5 split data into training , validation and testing
X = np.array(clips)
y = np.array(numeric_labels)
# First split: 70% training and 30% remaining data
X_train, X_temp, y_train, y_temp = train_test_split(X, y,test_size=0.30,random_state=42,stratify=y)

# Second split: 15% validation and 15% testing
X_val, X_test, y_val, y_test = train_test_split( X_temp, y_temp,test_size=0.50, random_state=42,stratify=y_temp)

#part 3: model building
num_classes = len(label_names)
class LipReadingCNN(nn.Module):
    def __init__(self):
        super(LipReadingCNN, self).__init__()
        # First 3D convolution layer
        self.conv1 = nn.Conv3d(in_channels=1,out_channels=32,kernel_size=3,padding=1)     
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        # Second 3D convolution layer
        self.conv2 = nn.Conv3d(in_channels=32,out_channels=64,kernel_size=3,padding=1)   
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=2)       
        # Third 3D convolution layer
        self.conv3 = nn.Conv3d(in_channels=64,out_channels=128,kernel_size=3,padding=1)   
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=2)        
        # Flatten layer
        self.flatten = nn.Flatten()
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3 * 6, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    # Forward propagation
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
# Create the model
model = LipReadingCNN()
print(model)# STEP 1: Convert NumPy arrays into PyTorch tensors

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# STEP 2: Create datasets

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)


# STEP 3: Create DataLoaders

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# STEP 5: Define loss function and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# STEP 6: Train and validate the model

epochs = 40

for epoch in range(epochs):

    # Training

    model.train()

    train_loss = 0.0
    correct = 0
    total = 0

    for videos, labels in train_loader:

        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(videos)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = correct / total
    train_loss = train_loss / len(train_loader)


    # Validation

    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for videos, labels in val_loader:

            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = correct / total
    val_loss = val_loss / len(val_loader)


    print(f"Epoch [{epoch+1}/{epochs}]")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")


# STEP 7: Test the final model

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for videos, labels in test_loader:

        videos = videos.to(device)
        labels = labels.to(device)

        outputs = model(videos)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total

print(f"Final Test Accuracy: {test_accuracy:.4f}")

