import cv2, numpy as np, pathlib, shutil, csv, sys

SRC = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(r"D:\fyp\phone_test\images")
DST = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path(r"D:\fyp\phone_test_restored\images")
DST.mkdir(parents=True, exist_ok=True)
if (SRC.parent / "labels.csv").exists():
    shutil.copy(SRC.parent / "labels.csv", DST.parent / "labels.csv")

def order_pts(p):
    p = p.reshape(4, 2).astype(np.float32)
    s, d = p.sum(1), np.diff(p, axis=1).ravel()
    return np.float32([p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]])

def find_film(img):
    """Return 4 corner points of the film, or None."""
    H, W = img.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    # background level = median of the photo border
    border = np.concatenate([g[:15].ravel(), g[-15:].ravel(), g[:, :15].ravel(), g[:, -15:].ravel()])
    bgv = np.median(border)
    
    diff = cv2.absdiff(g, np.full_like(g, int(bgv)))
    edges = cv2.Canny(g, 20, 60)
    mask = ((diff > 12).astype(np.uint8) * 255) | edges
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.15 * H * W:
        return None
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.05, 0.08):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            return order_pts(ap)
    return order_pts(cv2.boxPoints(cv2.minAreaRect(hull)))

def straighten(img, pts):
    tl, tr, br, bl = pts
    w = int((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2)
    h = int((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2)
    if w < 100 or h < 100:
        return None
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(pts, dst), (w, h))

def fix_light(gray):
    """Flatten uneven light / soft glare, then boost contrast."""
    g = gray.astype(np.float32)
    sigma = max(gray.shape) / 6
    illum = cv2.GaussianBlur(g, (0, 0), sigma)
    flat = g / (illum + 10) * float(illum.mean())
    flat = 0.7 * flat + 0.3 * g                      # keep some original look
    flat = np.clip(flat, 0, 255).astype(np.uint8)
    # inpaint any still-saturated pixels
    sat = (flat > 245).astype(np.uint8) * 255
    sat = cv2.dilate(sat, np.ones((5, 5), np.uint8))
    if sat.any():
        flat = cv2.inpaint(flat, sat, 5, cv2.INPAINT_TELEA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(flat)

def restore(img):
    pts = find_film(img)
    ok = pts is not None
    film = straighten(img, pts) if ok else None
    if film is None:
        film, ok = img, False
    h, w = film.shape[:2]
    m = int(0.03 * min(h, w))                        # cut the film edge / background
    film = film[m:h - m, m:w - m]
    gray = cv2.cvtColor(film, cv2.COLOR_BGR2GRAY)
    out = fix_light(gray)
    s = 1024 / max(out.shape)
    if s < 1:
        out = cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR), ok

files = sorted(q for q in SRC.iterdir() if q.suffix.lower() in {".jpg", ".jpeg", ".png"})
fails = 0
for i, f in enumerate(files):
    out, ok = restore(cv2.imread(str(f)))
    fails += (not ok)
    cv2.imwrite(str(DST / f.name), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if i % 300 == 0:
        print(i, "/", len(files))
print("done:", len(files), "images | film not found in:", fails)