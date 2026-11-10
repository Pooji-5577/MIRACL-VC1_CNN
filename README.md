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

## Known issues and next steps

- The pixel scaling is **not yet in `model2.py`**. Without it, `model2.py` stays at about 0.05. Add it before training.
- The mouth box is fixed per speaker. A few recordings (for example F04 and F09) move enough that the box misses the mouth. A face or mouth detector would fix this.
- Final training, validation and test accuracy for the full 20 epochs with scaling are being measured. Fill them in here when the run finishes.
- Data loading is slow. Saving the processed arrays once with `np.savez` avoids re-reading the images each run.
- Padding repeats the last frame. The book pads with each speaker's `dummy.jpg` instead.
