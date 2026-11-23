import { useEffect, useRef, useState } from "react";
import { analyze, validateFile } from "./api.js";
import CameraCapture from "./components/CameraCapture.jsx";
import DemoGallery from "./components/DemoGallery.jsx";
import Limitations from "./components/Limitations.jsx";
import PrivacyEditor from "./components/PrivacyEditor.jsx";
import Results from "./components/Results.jsx";

export default function App() {
  const [mode, setMode] = useState("idle"); // idle | camera | edit | loading | result
  const [file, setFile] = useState(null); // chosen file (before redaction)
  const [sentUrl, setSentUrl] = useState(null); // preview of what is being uploaded
  const [result, setResult] = useState(null);
  const [inputError, setInputError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [demoMode, setDemoMode] = useState(false);
  const [sens, setSens] = useState("screening"); // decision mode sent as form field "mode"
  const inputRef = useRef(null);
  const abortRef = useRef(null);

  useEffect(() => () => sentUrl && URL.revokeObjectURL(sentUrl), [sentUrl]);

  function choose(f) {
    if (!f) return;
    const err = validateFile(f);
    if (err) {
      setInputError(err);
      return;
    }
    setInputError(null);
    setFile(f);
    setMode("edit");
  }

  function reset() {
    abortRef.current?.abort();
    setFile(null);
    setSentUrl(null);
    setResult(null);
    setInputError(null);
    setMode("idle");
  }

  async function upload(f) {
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setSentUrl(URL.createObjectURL(f));
    setMode("loading");
    try {
      setResult(await analyze(f, ctrl.signal, sens));
      setMode("result");
    } catch (e) {
      if (e.name !== "AbortError") {
        setResult({ status: "error", message: "The request failed. Please try again.", findings: [], warnings: [] });
        setMode("result");
      }
    }
  }

  function pickDemo(f) {
    const err = validateFile(f);
    if (err) setInputError(err);
    else upload(f); // demo images are pre-vetted: one click straight to analysis
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-6">
      <header className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold">OrthoLens</h1>
          <p className="text-sm text-slate-600">Wrist and forearm X-ray screening and teaching aid. Not a diagnostic tool.</p>
        </div>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
          <label className="flex items-center gap-2">
            Sensitivity:
            <select value={sens} onChange={(e) => setSens(e.target.value)} className="rounded border border-slate-300 bg-white px-2 py-1">
              <option value="screening">Screening (catch more)</option>
              <option value="balanced">Balanced</option>
            </select>
          </label>
          <label className="flex cursor-pointer items-center gap-2">
            <input type="checkbox" checked={demoMode} onChange={(e) => setDemoMode(e.target.checked)} className="h-4 w-4" />
            Demo mode
          </label>
        </div>
      </header>

      <div className="space-y-4">
        {mode === "result" && result ? (
          <Results result={result} onReset={reset} />
        ) : (
          <>
            <div className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm font-medium text-amber-900" role="alert">
              Blur or crop names and patient IDs before uploading.
            </div>

            {mode === "idle" && (
              <>
                {demoMode && <DemoGallery onPick={pickDemo} disabled={false} />}
                <div
                  onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={(e) => { e.preventDefault(); setDragging(false); choose(e.dataTransfer.files?.[0]); }}
                  className={`rounded-lg border-2 border-dashed p-6 text-center sm:p-8 ${dragging ? "border-blue-500 bg-blue-50" : "border-slate-300 bg-white"}`}
                >
                  <p className="mb-4 text-slate-600">Drag and drop a JPEG or PNG photo of an X-ray film here (max 10 MB)</p>
                  <div className="flex flex-col justify-center gap-3 sm:flex-row">
                    <button onClick={() => inputRef.current?.click()} className="rounded bg-blue-600 px-4 py-3 font-medium text-white">
                      Choose file
                    </button>
                    <button onClick={() => setMode("camera")} className="rounded border border-slate-400 bg-white px-4 py-3 font-medium">
                      Take photo
                    </button>
                  </div>
                  <input
                    ref={inputRef}
                    type="file"
                    accept="image/jpeg,image/png"
                    className="hidden"
                    onChange={(e) => { choose(e.target.files?.[0]); e.target.value = ""; }}
                  />
                </div>
                {inputError && <p className="rounded border border-red-200 bg-red-50 p-3 text-sm text-red-800" role="alert">{inputError}</p>}
              </>
            )}

            {mode === "camera" && <CameraCapture onCapture={choose} onCancel={() => setMode("idle")} />}

            {mode === "edit" && file && (
              <PrivacyEditor
                file={file}
                onConfirm={upload}
                onCancel={reset}
                cancelLabel={file.name === "capture.jpg" ? "Retake" : "Choose a different image"}
              />
            )}

            {mode === "loading" && sentUrl && (
              <div className="space-y-3">
                <img src={sentUrl} alt="Image being analysed" className="mx-auto max-h-[60vh] w-full rounded-lg bg-black object-contain" />
                <div className="flex items-center gap-3 rounded-lg bg-white p-4" role="status">
                  <span className="h-5 w-5 animate-spin rounded-full border-2 border-blue-600 border-t-transparent" />
                  Analysing image… this can take a few seconds.
                </div>
              </div>
            )}
          </>
        )}
        <Limitations />
      </div>
    </div>
  );
}
