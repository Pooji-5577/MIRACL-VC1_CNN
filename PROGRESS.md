
## Session 2 (2026-11-10)
Goal: build model2.py from scratch, all 15 speakers, words + phrases (20 classes), then train, validate, test.
Done:
- Part 1 imports, Part 2 data (loops, grayscale 48x30, 15 frames per clip, label numbers, 70/15/15 split), Part 3 model, Part 4 train/validate/test. User wrote these.
- First full run stuck at 0.05 accuracy (loss 2.9957 = ln 20). Cause 1: whole 640x480 frame, mouth only 1-2 pixels. Cropped a 200x125 box per speaker around the mouth. Still stuck.
- Overfit test on 64 clips worked (loss 3.2 to 1.7), so the code is fine. Cause 2: unscaled pixels. Scaling (pixel - 0.457) / 0.273 fixed it. Lower learning rate and BatchNorm did not.
- Full 20 epochs with crop + scaling: last model train 0.9795, val 0.8733, test 0.8978. Best-val model (epoch 16): val 0.8822, test 0.9156.
- Accuracy printed as 0 to 1. Charts: plot_charts.py -> training_charts.png. README.md written with the code.
Open:
- model2.py has the crop and 0-1 accuracy edits but NOT the pixel scaling yet. Add it (user types it, or coach adds it).
- Overfitting after epoch 16. Next: save best-val model, try augmentation or more dropout.
- Ask user where model.py (project root) came from; never answered.
