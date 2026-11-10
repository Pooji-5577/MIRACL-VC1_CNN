import cv2, os, json, numpy as np, sys
root = sys.argv[1]; out = sys.argv[2]; mosaic_out = sys.argv[3]
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
boxes = {}; tiles = []
for sp in sorted(os.listdir(root)):
    wd = os.path.join(root, sp, "words")
    if not os.path.isdir(wd): continue
    faces_found = []; n = 0
    for w in sorted(os.listdir(wd)):
        for r in ["01", "04", "07", "10"]:
            d = os.path.join(wd, w, r)
            if not os.path.isdir(d): continue
            names = sorted(f for f in os.listdir(d) if f.startswith("color_"))
            if not names: continue
            img = cv2.imread(os.path.join(d, names[len(names)//2]), 0)
            n += 1
            f = cascade.detectMultiScale(img, 1.1, 5, minSize=(60, 60))
            if len(f):
                faces_found.append(max(f, key=lambda b: b[2]*b[3]))
    if len(faces_found) >= 5:
        x, y, w_, h = np.median(np.array(faces_found), axis=0)
        box = [max(0,int(x+0.05*w_)), max(0,int(y+0.5*h)), int(x+0.95*w_), int(y+1.0*h)]
    else:
        box = None
    boxes[sp] = box
    print(sp, "detections", len(faces_found), "of", n, "box", box)
json.dump(boxes, open(out, "w"), indent=1)
for sp, box in boxes.items():
    if box is None: continue
    img = cv2.imread(os.path.join(root, sp, "words", "03", "02", "color_006.jpg"), 0)
    x1, y1, x2, y2 = box
    t = cv2.resize(img[y1:y2, x1:x2], (192, 120)); cv2.putText(t, sp, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
    tiles.append(t)
while len(tiles) % 5: tiles.append(np.zeros((120, 192), np.uint8))
rows = [np.hstack(tiles[i:i+5]) for i in range(0, len(tiles), 5)]
cv2.imwrite(mosaic_out, np.vstack(rows))
