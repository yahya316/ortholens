import { useCallback, useEffect, useRef, useState } from "react";

const MAX_SIDE = 2000;

/**
 * Client-side privacy tool. Draw filled black rectangles over names/IDs.
 * Nothing is uploaded here: onConfirm(File) is called only when the user confirms,
 * with the redacted pixels re-encoded from the canvas.
 */
export default function PrivacyEditor({ file, onConfirm, onCancel, cancelLabel }) {
  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  const dragRef = useRef(null);
  const [rects, setRects] = useState([]); // in canvas pixel coordinates
  const [draft, setDraft] = useState(null);
  const [ready, setReady] = useState(false);
  const [loadError, setLoadError] = useState(null);

  useEffect(() => {
    let revoked = false;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      if (revoked) return;
      const scale = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
      const c = canvasRef.current;
      c.width = Math.round(img.naturalWidth * scale);
      c.height = Math.round(img.naturalHeight * scale);
      imgRef.current = img;
      setReady(true);
    };
    img.onerror = () => !revoked && setLoadError("This image could not be opened. Please choose another file.");
    img.src = url;
    return () => { revoked = true; URL.revokeObjectURL(url); };
  }, [file]);

  const draw = useCallback(() => {
    const c = canvasRef.current;
    if (!c || !imgRef.current) return;
    const ctx = c.getContext("2d");
    ctx.drawImage(imgRef.current, 0, 0, c.width, c.height);
    ctx.fillStyle = "#000";
    [...rects, ...(draft ? [draft] : [])].forEach((r) => ctx.fillRect(r.x, r.y, r.w, r.h));
  }, [rects, draft]);

  useEffect(() => { if (ready) draw(); }, [ready, draw]);

  function pos(e) {
    const c = canvasRef.current;
    const b = c.getBoundingClientRect();
    return {
      x: Math.min(Math.max(0, ((e.clientX - b.left) * c.width) / b.width), c.width),
      y: Math.min(Math.max(0, ((e.clientY - b.top) * c.height) / b.height), c.height),
    };
  }
  const norm = (a, b) => ({ x: Math.min(a.x, b.x), y: Math.min(a.y, b.y), w: Math.abs(a.x - b.x), h: Math.abs(a.y - b.y) });

  function down(e) {
    e.currentTarget.setPointerCapture(e.pointerId);
    dragRef.current = pos(e);
  }
  function move(e) {
    if (dragRef.current) setDraft(norm(dragRef.current, pos(e)));
  }
  function up(e) {
    if (!dragRef.current) return;
    const r = norm(dragRef.current, pos(e));
    dragRef.current = null;
    setDraft(null);
    if (r.w > 4 && r.h > 4) setRects((prev) => [...prev, r]);
  }

  function confirm() {
    const c = canvasRef.current;
    c.toBlob(
      (blob) => blob && onConfirm(new File([blob], "upload.jpg", { type: "image/jpeg" })),
      "image/jpeg",
      0.92
    );
  }

  if (loadError) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">
        <p>{loadError}</p>
        <button onClick={onCancel} className="mt-3 btn-secondary">Back</button>
      </div>
    );
  }

  return (
    <div className="card space-y-4">
      <div className="border-b border-slate-100 pb-3">
        <h3 className="text-base font-semibold text-slate-900">Privacy tool: redact patient info</h3>
        <p className="mt-1 text-xs text-slate-500">
          Optional: drag on the image to draw black boxes over names, dates, or IDs. Only the redacted image is uploaded.
        </p>
      </div>
      <div className="overflow-hidden rounded-lg border border-slate-200 bg-slate-900">
        <canvas
          ref={canvasRef}
          onPointerDown={down}
          onPointerMove={move}
          onPointerUp={up}
          onPointerCancel={up}
          aria-label="Image preview. Drag to draw black boxes over private information."
          className="mx-auto max-h-[60vh] w-full cursor-crosshair object-contain"
          style={{ touchAction: "none" }}
        />
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex gap-2">
          <button onClick={() => setRects((r) => r.slice(0, -1))} disabled={!rects.length} className="btn-secondary !py-1.5 !px-3 text-xs">
            Undo
          </button>
          <button onClick={() => setRects([])} disabled={!rects.length} className="btn-secondary !py-1.5 !px-3 text-xs">
            Clear all
          </button>
        </div>
        <span className="text-xs text-slate-500">{rects.length} area{rects.length === 1 ? "" : "s"} hidden</span>
      </div>
      <div className="flex flex-col gap-3 pt-2 sm:flex-row">
        <button onClick={confirm} disabled={!ready} className="btn-primary">
          Upload and analyse
        </button>
        <button onClick={onCancel} className="btn-secondary">{cancelLabel}</button>
      </div>
    </div>
  );
}
