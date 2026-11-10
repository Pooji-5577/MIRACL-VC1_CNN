# Lip Reading CNN

A 3D CNN, written in PyTorch, that reads lip movements from short videos and predicts which word or phrase was spoken.

Reference: book chapter 7.3, "Visual Speech Recognition using 3D-CNN" (`Lip_Reading_CNN_Model.pdf`).

## Dataset

MIRACL-VC1, color images only.

- 15 speakers (`F01`..`F11`, `M01`..`M08`)
- 20 classes: 10 words + 10 phrases (labels such as `words_01`, `phrases_07`)
- 10 repetitions per class per speaker, so 15 x 20 x 10 = **3000 recordings**
- Each recording is a folder of frames: `dataset/<speaker>/<words|phrases>/<label>/<repetition>/color_001.jpg ...`
- Recordings have between 4 and 27 frames (most have about 10)
- `dataset/calib.txt` is a file, not a speaker, and is skipped

## Files

| File | What it is |
|---|---|
| `model2.py` | The model built step by step in this project. Run this one. |
| `model.py` | An older full script, kept only for comparison. Running it overwrites `best_model.pth`. |
| `Lip_Reading_CNN_Model.pdf` | The book chapter this project follows. |
| `PROGRESS.md` | Session log. |
| `plot_charts.py` | Draws the loss and accuracy charts from `training_log.csv`. Run: `python3 plot_charts.py` |
| `training_log.csv` | One row per epoch: training loss, training accuracy, validation accuracy (0 to 1). |
| `training_charts.png` | The charts drawn by `plot_charts.py`. |

## How `model2.py` works

It has 4 parts, in this order.

**Part 1: imports.** `os`, `cv2`, `numpy`, `torch`, `torch.nn`, plus `train_test_split` and the PyTorch data tools.

**Part 2: data.** For every recording:
1. Find the folder (speaker, words or phrases, label, repetition).
2. Read each `color_*.jpg` frame. Depth files are skipped.
3. Crop a 200 x 125 box around the speaker's mouth (`mouth_center` holds one position per speaker).
4. Make the crop gray, resize to 48 wide x 30 tall, and divide by 255 so values run from 0 to 1.
5. Make the clip exactly 15 frames. Longer clips: pick 15 evenly spaced frames. Shorter clips: repeat the last frame.
6. Turn the label names into numbers 0 to 19.
7. Split 70% / 15% / 15% into train / validation / test (2100 / 450 / 450). The split is stratified, so every class appears in each part.

**Part 3: model.** Input shape is `(batch, 1, 15, 30, 48)`.

| Layer | Output shape |
|---|---|
| Conv3d 1 to 32, kernel 3, padding 1 + ReLU + MaxPool (1,2,2) | (32, 15, 15, 24) |
| Conv3d 32 to 64 + ReLU + MaxPool 2 | (64, 7, 7, 12) |
| Conv3d 64 to 128 + ReLU + MaxPool 2 | (128, 3, 3, 6) |
| Flatten | 6912 |
| Linear 6912 to 256 + ReLU + Dropout 0.5 | 256 |
| Linear 256 to 20 | 20 |

Total parameters: 2,052,436.

**Part 4: train, validate, test.** CrossEntropyLoss, Adam with learning rate 0.001, batch size 16, 20 epochs. Training and validation accuracy print after every epoch. The test set is used once, at the end. All accuracies are numbers from 0 to 1.

## Run it

From the project folder:

```
python3 model2.py
```

Reading about 35,000 images takes a few minutes. After that, one epoch takes roughly 45 seconds on CPU.

Needs: Python 3, `torch`, `opencv-python`, `numpy`, `scikit-learn`.

## What we learned so far

Guessing among 20 classes gives accuracy 0.05 and a loss of 2.9957 (that is ln 20).

| Experiment | Result |
|---|---|
| Whole 640 x 480 frame shrunk to 48 x 30 | Stuck at 0.05. The mouth was only 1 or 2 pixels wide. |
| Crop around the mouth, nothing else changed | Still stuck at 0.05 (6 epochs). |
| Overfit test: train on 64 clips only | Loss fell from 3.2 to 1.7 and accuracy reached 0.59. The model and training loop work. |
| Lower learning rate (0.0003) | Still stuck at 0.05 after 3 epochs. |
| BatchNorm after each conv | 0.08 validation accuracy after 3 epochs. |
| **Scale pixels: (pixel - 0.457) / 0.273** | **0.1978 validation accuracy after 3 epochs.** The model starts learning. |

The numbers 0.457 and 0.273 are the mean and standard deviation of the training pixels.

## Results

Full run: cropped mouth region + scaled pixels, 20 epochs, batch size 16, learning rate 0.001. All accuracies are from 0 to 1. Guessing among 20 classes gives 0.05.

| Model | Training accuracy | Validation accuracy | **Test accuracy** |
|---|---|---|---|
| After epoch 20 (last) | 0.9795 | 0.8733 | **0.8978** |
| Epoch 16 (best validation accuracy) | 0.9790 | 0.8822 | **0.9156** |

- The epoch 16 model was picked using the validation set, not the test set, so its test accuracy is an honest number.
- The test set has 450 clips, so a difference of about 0.02 between two test accuracies is within normal noise. The two models are about equally good.
- Training accuracy in the table was measured with dropout off, on all 2100 training clips. The per-epoch training accuracy in the chart is lower (0.89 at epoch 20) because it is measured while training, with dropout on. Compared fairly, the model scores about 0.98 on clips it has seen and about 0.88 to 0.90 on clips it has not. That gap is overfitting.
- Validation accuracy stopped improving after epoch 16 (best 0.8822), while training accuracy kept rising.

## Training curves

![Training loss and accuracy per epoch](training_charts.png)

The dashed lines mark pure guessing among 20 classes (accuracy 0.05, loss 2.9957). Validation loss was not recorded, so the loss chart shows training loss only.

## Known issues and next steps

- The pixel scaling is **not yet in `model2.py`**. Without it, `model2.py` stays at about 0.05. Add it before training.
- The mouth box is fixed per speaker. A few recordings (for example F04 and F09) move enough that the box misses the mouth. A face or mouth detector would fix this.
- The model overfits after about epoch 16. Ideas to try: stop at the best validation epoch, add data augmentation, or add more dropout.
- Data loading is slow. Saving the processed arrays once with `np.savez` avoids re-reading the images each run.
- Padding repeats the last frame. The book pads with each speaker's `dummy.jpg` instead.

## Full code: model2.py

This is a copy of the file as it was when this README was written. If you edit `model2.py`, the real file is the one that counts.

```python
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
```
