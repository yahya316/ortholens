import csv, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, cv2
from PIL import Image
from torchvision import models, transforms as T
from ultralytics import YOLO

RAW = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\fyp\real_photos")
CLEAN = Path(sys.argv[2] if len(sys.argv) > 2 else r"D:\fyp\real_restored")
OUT = Path(sys.argv[3] if len(sys.argv) > 3 else r"D:\fyp\real_results")
OUT.mkdir(parents=True, exist_ok=True)

cfg = json.load(open(r"D:\fyp\combo_config.json"))
ref_y, ref_r = np.array(cfg["ref_yolo"]), np.array(cfg["ref_resnet"])
cdf = lambda ref, x: np.searchsorted(ref, x, side="right") / len(ref)

dev = "cuda" if torch.cuda.is_available() else "cpu"
res = models.resnet50(); res.fc = nn.Linear(res.fc.in_features, 2)
res.load_state_dict(torch.load(r"D:\fyp\runs_cls\resnet50.pt", map_location=dev))
res.to(dev).eval()
tf = T.Compose([T.Resize((cfg["resnet_size"],) * 2), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
yolo = YOLO(r"D:\fyp\runs\frac_s_adult\weights\best.pt")

def read(p):  # safe for odd characters in WhatsApp file names
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)

rows = []
files = sorted(p for p in RAW.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
for f in files:
    c = CLEAN / f.name
    if not c.exists():
        print("no cleaned copy for", f.name); continue
    r = yolo.predict(str(f), imgsz=cfg["imgsz"], conf=0.01, verbose=False)[0]
    ys = float(r.boxes.conf.max()) if len(r.boxes) else 0.0
    with torch.no_grad():
        x = tf(Image.open(c).convert("RGB")).unsqueeze(0).to(dev)
        rs = float(torch.softmax(res(x), 1)[0, 1])
    score = cfg["w_yolo"] * cdf(ref_y, ys) + cfg["w_resnet"] * cdf(ref_r, rs)
    bal, scr = score >= cfg["thr_balanced"], score >= cfg["thr_screening"]

    img = read(f); H, W = img.shape[:2]
    th, fs = max(2, W // 300), max(0.6, W / 1100)
    nb = 0
    for b in r.boxes:
        cf = float(b.conf)
        if cf < cfg["yolo_box_conf"]: continue
        nb += 1
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 80, 0), th)
        cv2.putText(img, f"fracture {cf:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 80, 0), th)
    txt = f"score {score:.2f} | balanced: {'FRACTURE' if bal else 'no fracture'} | screening: {'FRACTURE' if scr else 'no fracture'}"
    cv2.rectangle(img, (0, 0), (W, int(50 * fs) + 10), (0, 0, 0), -1)
    cv2.putText(img, txt, (10, int(36 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, th - 1))
    cv2.imencode(".jpg", img)[1].tofile(str(OUT / (f.stem + ".jpg")))
    rows.append([f.name, round(ys, 3), round(rs, 3), round(score, 3), int(bal), int(scr), nb, ""])

with open(OUT / "results.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["file", "yolo_top", "resnet_prob", "combined", "balanced", "screening", "boxes_shown", "doctor_label"])
    w.writerows(rows)
    
print(f"{'file':52s} yolo  resnet  comb  bal scr boxes")
for r in rows:
    print(f"{r[0][:50]:52s} {r[1]:.2f}  {r[2]:.2f}    {r[3]:.2f}   {r[4]}   {r[5]}   {r[6]}")
print("saved:", OUT)