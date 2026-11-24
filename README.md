# OrthoLens

Prototype web app that finds **suspected** fractures in phone photos of wrist/forearm X-ray films and draws bounding boxes.

> **OrthoLens is a screening and teaching aid, not a diagnosis. A qualified doctor must confirm every finding.**
> It gives no treatment, first-aid or referral advice.

Flow: upload/capture photo -> validate and strip metadata (in memory) -> quality check -> **CLEAN** (film crop + perspective fix, light flattening, CLAHE) -> **YOLOv8s on the RAW photo** (boxes) + **ResNet50 on the CLEANED photo** -> rank-based combined score -> decision (Screening or Balanced) -> original with boxes, plus the cleaned image ("What the AI saw").

## Requirements
- Python 3.11, Node.js 18+ (20+ recommended)
- Three model files (see below). `ultralytics==8.4.216` is pinned; `torch>=2.6` with a matching `torchvision` (the CPU build is fine).

## Where to put the model files
Copy these into `backend/models/` (or set `MODELS_DIR`; each file can also be set individually):

| File | Env override | What |
|---|---|---|
| `best.pt` | `MODEL_PATH` | YOLOv8s fracture detector (one class) |
| `resnet50.pt` | `RESNET_PATH` | torchvision resnet50 state_dict, `fc = Linear(2048, 2)`, class 1 = fracture |
| `combo_config.json` | `COMBO_CONFIG_PATH` | weights, reference distributions, thresholds, `imgsz`, `resnet_size` |

If any is missing the server still starts; `/api/health` shows which are loaded (`{"models_loaded": {"yolo", "resnet", "config"}, "device"}`) and `POST /api/analyses` returns **503** naming the missing file(s). `backend/reference/` holds your offline scripts for reference only; the app never imports them.

## The recipe (what the app computes)
1. **Clean** (`app/restore.py`, a port of `reference/restore.py`): film crop with perspective fix, 3% margin trim, grayscale, light flattening + saturated-pixel inpaint, CLAHE, max 1024 px.
2. **YOLO** on the raw photo: `conf=0.01`, `imgsz` from the config. `yolo_top` = highest box confidence (0 if none). Boxes shown only if confidence >= `yolo_box_conf` (0.33).
3. **ResNet50** on the cleaned photo (JPEG q95 round-trip, resize 384, ImageNet normalisation). `resnet_prob` = softmax probability of class 1.
4. **Combined score** = `w_yolo * cdf(ref_yolo, yolo_top) + w_resnet * cdf(ref_resnet, resnet_prob)`, where `cdf(ref, x) = searchsorted(ref, x, "right") / len(ref)`. This is a **rank-based score on a 0-1 scale, not a probability** and is never shown as a percentage.
5. **Decision**: combined >= `thr_screening` (mode `screening`, catches more) or >= `thr_balanced` (mode `balanced`). Default from `DECISION_MODE`; the upload accepts an optional form field `mode`.
6. Fracture: show the boxes from step 2; if none passes 0.33 the message is "Fracture suspected, location uncertain" with no boxes. No fracture: status `no_fracture_detected`, no boxes.

Response additions: `decision`, `decision_mode`, `scores {yolo_top, resnet_prob, combined, threshold}`, `cleaned_image` (when `SHOW_CLEANED`), `timings_ms {clean, yolo, resnet, total}`. Box coordinates are in the **raw photo**; the returned images are downscaled to `DISPLAY_MAX_SIDE` for transport only (analysis always uses the full photo).

### Checking parity with your offline script
```powershell
cd backend
.\.venv\Scripts\python.exe scripts\check_combo.py path\to\photos
```
Prints `file, yolo_top, resnet_prob, combined, balanced, screening, boxes` in the same format as `reference/predict_combo.py`, plus median ms per step. Values should agree within 0.01.

## Backend

Windows (PowerShell):
```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Windows without activating (works under any execution policy):
```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```
macOS/Linux:
```bash
cd backend && python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && uvicorn app.main:app --reload
```
API: http://localhost:8000 Â· docs: http://localhost:8000/docs Â· health: http://localhost:8000/api/health

Run tests (no weights needed; the models are mocked):
```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest
```

## Frontend
```powershell
cd frontend
copy .env.example .env      # sets VITE_API_URL (default http://localhost:8000)
npm install                 # in PowerShell with blocked scripts use: npm.cmd install
npm run dev                 # or: npm.cmd run dev   -> http://localhost:5173
```
Camera (`getUserMedia`) works on `localhost` or HTTPS only. On a phone over plain-HTTP LAN, use "Choose file" instead (or serve over HTTPS). If you use another frontend origin, add it to `CORS_ORIGINS`.

## Tuning (environment variables, defaults in `backend/app/config.py`)
Set before starting the server, e.g. PowerShell: `$env:DECISION_MODE="balanced"; uvicorn app.main:app --reload`.

| Variable | Default | Meaning |
|---|---|---|
| `MODELS_DIR` | backend/models | Folder with the three model files. |
| `MODEL_PATH`, `RESNET_PATH`, `COMBO_CONFIG_PATH` | `MODELS_DIR/...` | Individual file locations. |
| `DECISION_MODE` | screening | `screening` or `balanced` (the request's `mode` field overrides it). |
| `SHOW_CLEANED` | true | Include `cleaned_image` in the response. |
| `SPLIT_VIEWS` | false | Double-view splitting. **Not validated; keep off.** Code is kept but never runs unless true. |
| `LOW_CONF` | 0.45 | Boxes below this are flagged "low confidence" (dashed orange box). Display only. |
| `DEVICE` | auto | `auto` (GPU if available), `cpu`, `0`, ... |
| `DISPLAY_MAX_SIDE` | 1600 | Size of returned images (display only). |
| `MAX_SIDE` | 0 | 0 = never downscale the upload (needed for parity). |
| `SAVE_UPLOADS` | false | If true, the de-identified (metadata-free) image is saved to `UPLOAD_DIR` under a random name. |
| `QUALITY_MODE` | warn | `warn`: always run the models, add warnings for poor quality. `block`: return `retake` only for unusable images (below half a threshold). `off`: skip the check. |
| `MIN_SHARPNESS` | 15 | Variance of the Laplacian on the film region (resized to 1024 px wide). Placeholder: calibrate with `quality_stats.py`. |
| `MIN_CONTRAST` | 40 | 95th minus 5th percentile of grey values on the film region. Placeholder: calibrate. |
| `DEBUG_QUALITY` | false | Log sharpness and contrast for every image. |
| `MAX_UPLOAD_BYTES`, `JPEG_QUALITY`, `CORS_ORIGINS` | 10 MB, 90, localhost:5173 | Upload limit, display JPEG quality, allowed origins. |

No longer used: `CONF` (the decision now comes from `combo_config.json`), `IOU`, `IMGSZ` (the config's `imgsz` is used), `ENABLE_FILM_CROP`, `ENABLE_GLARE_REDUCTION`, `ENABLE_CLAHE` (the cleanup is the fixed ported algorithm).

## Latency (CPU)
Measured on this Windows machine, CPU only, 2400x1800 photos, median of 3 (`check_combo.py`): clean about 1.5-2.0 s (median 1.85 s), YOLO about 0.3-0.6 s, ResNet50 about 0.3 s. A full API request takes about 2.2-3.1 s in total including decoding and JPEG encoding (the first request is the slowest). Your hardware will differ; run `check_combo.py` to measure.

See `config.py` for all settings.

## Quality check
Mean brightness is **not** used (X-ray photos have a large dark background). Sharpness and contrast are measured on the cropped film region (full frame if no film is found). In `warn` mode (default) poor quality only adds a `warnings` entry; the models still run. Responses always include `notice`, `warnings` (always containing the body-part warning), `quality` (`sharpness`, `contrast`, `film_found`, `views_found`) and `thresholds` (`conf` = box display threshold, `low_conf`, `imgsz`, `quality_mode`).

### Calibrating thresholds
```powershell
cd backend
.\.venv\Scripts\python.exe scripts\quality_stats.py path\to\photos
```
Prints filename, sharpness, contrast and film_found per image (same functions as the pipeline), then min/median/max. Pick `MIN_SHARPNESS` and `MIN_CONTRAST` from your good and bad photos, e.g. `$env:MIN_SHARPNESS="25"`.

## Frontend features
- **Privacy tool:** after choosing a photo, drag to draw black rectangles over names/IDs (Undo / Clear). Nothing is uploaded until you press "Upload and analyse"; the redacted pixels are what is sent.
- **Demo mode:** toggle in the header shows a gallery of images from `frontend/public/demo/` (`.jpg`, `.jpeg`, `.png`). Add files there (no restart needed in dev); one click analyses an image. With no images the gallery is hidden. Blur patient names in demo images first.
- **Sensitivity** selector (Screening / Balanced) sends the form field `mode`.
- Results show the decision line, "Suspicion score 0.xx (relative scale, not a probability)" with an explanatory tooltip, the original photo with boxes, a collapsible "What the AI saw (cleaned image)", findings with box confidence and "low confidence" tags, warnings and the body-part warning (yellow), the notice (red border), total time, and a collapsible Limitations panel. Works from 360 px width.

## Presentation checklist
- Pick demo photos in advance and test each one.
- Keep one weak photo (blurry or low contrast) to show the warning banner.
- Blur patient names before adding images to `public/demo`.
- Record a backup screen video in case of camera, network or model problems.

## Privacy
Uploads are processed in memory only, EXIF/metadata are dropped by re-encoding pixels, and the original file name is never used. Nothing is stored unless `SAVE_UPLOADS=true`. The UI asks users to blur names and patient IDs first.

## Notes and limits
- Film detection assumes a bright film/screen region; otherwise the full frame is used (`film_found: false`).
- Double-view split (dark vertical gap in the middle third) exists but is disabled (`SPLIT_VIEWS=false`) because it was not validated.
- Boxes are in coordinates of the *raw* photo (the `annotated_image`).
- `backend/app/scope.py` is a stub (always in scope); every response carries a warning that the body-part check is not installed. Plug in a classifier with `set_classifier(fn)` or edit `is_in_scope`.
- Structure leaves room to add auth and Postgres later.