import csv, json, numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

D = r"D:\fyp"
W_YOLO, W_RES = 0.6, 0.4

def read(folder):
    with open(rf"{D}\{folder}\scores.csv") as fh:
        rows = list(csv.DictReader(fh))
    return sorted(rows, key=lambda r: r["file"])

col = lambda rows, k: np.array([float(r[k]) for r in rows])
lab = lambda rows: np.array([int(r["label"]) for r in rows])

va, vb = read("phone_val"), read("phone_val_restored")
ta, tb = read("phone_test"), read("phone_test_restored")
yv, yt = lab(va), lab(ta)
ref_y = np.sort(col(va, "yolo"))          # YOLO on the raw photo
ref_r = np.sort(col(vb, "resnet50"))      # ResNet50 on the cleaned photo

cdf = lambda ref, x: np.searchsorted(ref, x, side="right") / len(ref)
combo = lambda y, r: W_YOLO * cdf(ref_y, y) + W_RES * cdf(ref_r, r)
sv = combo(col(va, "yolo"), col(vb, "resnet50"))
st = combo(col(ta, "yolo"), col(tb, "resnet50"))

fpr, tpr, th = roc_curve(yv, sv)
t_bal = float(th[np.argmax(tpr - fpr)])
t_scr = float(th[np.where(tpr >= 0.95)[0][0]])

for name, t in (("balanced", t_bal), ("screening", t_scr)):
    p = st >= t
    tp, fn = (p & (yt == 1)).sum(), (~p & (yt == 1)).sum()
    tn, fp = (~p & (yt == 0)).sum(), (p & (yt == 0)).sum()
    print(f"{name:10s} thr {t:.3f} | test sens {tp/(tp+fn):.3f} spec {tn/(tn+fp):.3f}")
print("test AUC:", round(roc_auc_score(yt, st), 3))

json.dump({"w_yolo": W_YOLO, "w_resnet": W_RES,
           "ref_yolo": ref_y.tolist(), "ref_resnet": ref_r.tolist(),
           "thr_balanced": t_bal, "thr_screening": t_scr,
           "yolo_score_conf": 0.01, "yolo_box_conf": 0.33, "imgsz": 1024, "resnet_size": 384},
          open(rf"{D}\combo_config.json", "w"))
print("saved D:\\fyp\\combo_config.json")