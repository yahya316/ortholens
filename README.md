# OrthoLens

Prototype web app that finds **suspected** fractures in phone photos of wrist/forearm X-ray films and draws bounding boxes.

> **OrthoLens is a screening and teaching aid, not a diagnosis. A qualified doctor must confirm every finding.**
> It gives no treatment, first-aid or referral advice.

Flow: upload/capture photo â†’ validate and strip metadata (in memory) â†’ quality check â†’ restore (film crop, glare, CLAHE) â†’ split double views â†’ YOLOv8 detection â†’ original and annotated images side by side.

## Requirements
- Python 3.11, Node.js 18+ (20+ recommended)
- Your trained weights: a single-class (`fracture`) YOLOv8s `best.pt`

## Where to put the weights
Copy `best.pt` to `backend/models/best.pt` (or set `MODEL_PATH`). If it is missing the server still starts; `/api/health` shows `model_loaded: false` and `POST /api/analyses` returns **503** with a message telling you to add the weights.

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

Run tests (no `best.pt` needed; the model is mocked):
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
Set before starting the server, e.g. PowerShell: `$env:CONF="0.25"; uvicorn app.main:app --reload`.

| Variable | Default | Meaning |
|---|---|---|
| `CONF` | 0.33 | Minimum confidence to report a finding (validation F1 peak, F1 = 0.90). Lower (e.g. 0.25) = more sensitive, more false positives. |
| `LOW_CONF` | 0.45 | Findings below this are flagged "low confidence" (dashed orange box). |
| `IMGSZ` | 1024 | Inference image size. |
| `IOU` | 0.7 | NMS IoU threshold. |
| `DEVICE` | auto | `auto` (GPU if available), `cpu`, `0`, â€¦ |
| `MODEL_PATH` | backend/models/best.pt | Weights location. |
| `SAVE_UPLOADS` | false | If true, the de-identified (metadata-free) image is saved to `UPLOAD_DIR` under a random name. |
| `QUALITY_MODE` | warn | `warn`: always run the model, add warnings for poor quality. `block`: return `retake` only for unusable images (below half a threshold). `off`: skip the check. |
| `MIN_SHARPNESS` | 15 | Variance of the Laplacian on the film region (resized to 1024 px wide). Placeholder: calibrate with `quality_stats.py`. |
| `MIN_CONTRAST` | 40 | 95th minus 5th percentile of grey values on the film region. Placeholder: calibrate. |
| `DEBUG_QUALITY` | false | Log sharpness and contrast for every image. |
| `ENABLE_FILM_CROP`, `ENABLE_GLARE_REDUCTION`, `ENABLE_CLAHE`, `ENABLE_SPLIT` | true | Toggle restore steps. |
| `MAX_SIDE` | 2000 | Larger photos are downscaled. |

See `config.py` for all settings.

## Quality check
Mean brightness is **not** used (X-ray photos have a large dark background). Sharpness and contrast are measured on the cropped film region (full frame if no film is found). In `warn` mode (default) poor quality only adds a `warnings` entry; the model still runs and confidences are always returned. Responses always include `notice`, `warnings`, `quality` (`sharpness`, `contrast`, `film_found`, `views_found`) and `thresholds` (`conf`, `low_conf`, `imgsz`, `quality_mode`).

### Calibrating thresholds
```powershell
cd backend
.\.venv\Scripts\python.exe scripts\quality_stats.py path\to\photos
```
Prints filename, sharpness, contrast and film_found per image (same functions as the pipeline), then min/median/max. Pick `MIN_SHARPNESS` and `MIN_CONTRAST` from your good and bad photos, e.g. `$env:MIN_SHARPNESS="25"`.

## Frontend features
- **Privacy tool:** after choosing a photo, drag to draw black rectangles over names/IDs (Undo / Clear). Nothing is uploaded until you press "Upload and analyse"; the redacted pixels are what is sent.
- **Demo mode:** toggle in the header shows a gallery of images from `frontend/public/demo/` (`.jpg`, `.jpeg`, `.png`). Add files there (no restart needed in dev); one click analyses an image. With no images the gallery is hidden. Blur patient names in demo images first.
- Results show warnings (yellow), the notice (red border), per-finding confidence, "low confidence" tags, elapsed time, model version, and a collapsible Limitations panel.

## Presentation checklist
- Pick demo photos in advance and test each one.
- Keep one weak photo (blurry or low contrast) to show the warning banner.
- Blur patient names before adding images to `public/demo`.
- Record a backup screen video in case of camera, network or model problems.

## Privacy
Uploads are processed in memory only, EXIF/metadata are dropped by re-encoding pixels, and the original file name is never used. Nothing is stored unless `SAVE_UPLOADS=true`. The UI asks users to blur names and patient IDs first.

## Notes and limits
- Film detection assumes a bright film/screen region; otherwise the full frame is used (`film_found: false`).
- Double-view split looks for a dark vertical gap in the middle third; otherwise one image is used.
- Boxes are in coordinates of the *restored* image (the `annotated_image`).
- `backend/app/scope.py` is a stub (always in scope). Plug in a classifier with `set_classifier(fn)` or edit `is_in_scope`.
- Structure leaves room to add auth and Postgres later.