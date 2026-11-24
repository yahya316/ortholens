import { useEffect, useRef, useState } from "react";
import { analyze, validateFile } from "./api.js";
import CameraCapture from "./components/CameraCapture.jsx";
import DemoGallery from "./components/DemoGallery.jsx";
import Limitations from "./components/Limitations.jsx";
import PrivacyEditor from "./components/PrivacyEditor.jsx";
import Results from "./components/Results.jsx";

const STEPS = ["Choose photo", "Privacy", "Analyse", "Result"];
const STEP_OF = { idle: 0, camera: 0, edit: 1, loading: 2, result: 3 };

const MODES = [
  { id: "screening", label: "Screening", hint: "Catch more", desc: "Lower cut-off: flags more photos, including more false alarms." },
  { id: "balanced", label: "Balanced", hint: "Fewer false alarms", desc: "Higher cut-off: fewer false alarms, but may miss more." },
];

function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-600 text-white shadow-sm" aria-hidden="true">
        <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="11" cy="11" r="6" />
          <path d="m20 20-4-4" />
          <path d="M9 11h4M11 9v4" />
        </svg>
      </div>
      <div>
        <h1 className="text-lg font-bold leading-tight text-slate-900">OrthoLens</h1>
        <p className="text-xs text-slate-500">Wrist &amp; forearm X-ray screening aid</p>
      </div>
    </div>
  );
}

function Stepper({ current }) {
  return (
    <ol className="flex items-center gap-1 overflow-x-auto text-xs sm:text-sm" aria-label="Progress">
      {STEPS.map((s, i) => {
        const done = i < current;
        const active = i === current;
        return (
          <li key={s} className="flex items-center gap-1" aria-current={active ? "step" : undefined}>
            <span
              className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-semibold ${
                done ? "bg-emerald-600 text-white" : active ? "bg-blue-600 text-white ring-2 ring-blue-100" : "bg-slate-100 text-slate-500 border border-slate-200"
              }`}
            >
              {done ? "✓" : i + 1}
            </span>
            <span className={`whitespace-nowrap ${active ? "font-semibold text-slate-900" : "text-slate-500"}`}>{s}</span>
            {i < STEPS.length - 1 && <span className="mx-1 h-px w-4 bg-slate-200 sm:w-8" aria-hidden="true" />}
          </li>
        );
      })}
    </ol>
  );
}

function SensitivityCard({ sens, setSens }) {
  const current = MODES.find((m) => m.id === sens);
  return (
    <div className="card">
      <h2 className="mb-2 text-xs font-bold uppercase tracking-wider text-slate-500">Sensitivity Mode</h2>
      <div className="grid grid-cols-2 gap-1 rounded-lg bg-slate-100 p-1" role="radiogroup" aria-label="Sensitivity">
        {MODES.map((m) => (
          <button
            key={m.id}
            role="radio"
            aria-checked={sens === m.id}
            onClick={() => setSens(m.id)}
            className={`rounded-md px-2 py-2 text-left transition ${sens === m.id ? "bg-white shadow-sm ring-1 ring-slate-200 text-slate-900" : "text-slate-600 hover:text-slate-900"}`}
          >
            <span className="block text-sm font-semibold">{m.label}</span>
            <span className="block text-[11px] text-slate-500">{m.hint}</span>
          </button>
        ))}
      </div>
      <p className="mt-2 text-xs text-slate-600 leading-relaxed">{current.desc}</p>
    </div>
  );
}

function HowItWorks() {
  const items = [
    ["1", "Clean", "The film is found, straightened and its lighting evened out."],
    ["2", "Two AI checks", "A detector looks for fracture regions; a classifier scores the cleaned image."],
    ["3", "Suspicion score", "Both are combined into a relative 0–1 score and compared with a cut-off."],
  ];
  return (
    <div className="card">
      <h2 className="mb-3 text-xs font-bold uppercase tracking-wider text-slate-500">How it works</h2>
      <ul className="space-y-3">
        {items.map(([n, t, d]) => (
          <li key={n} className="flex gap-2.5">
            <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-blue-50 text-xs font-bold text-blue-700 border border-blue-200">{n}</span>
            <div>
              <p className="text-xs font-semibold text-slate-800">{t}</p>
              <p className="text-xs text-slate-500 leading-relaxed">{d}</p>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function App() {
  const [mode, setMode] = useState("idle"); // idle | camera | edit | loading | result
  const [file, setFile] = useState(null); // chosen file (before redaction)
  const [sentUrl, setSentUrl] = useState(null); // preview of what is being uploaded
  const [result, setResult] = useState(null);
  const [inputError, setInputError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [demoMode, setDemoMode] = useState(false);
  const [sens, setSens] = useState("screening"); // decision mode sent as form field "mode"
  const [elapsed, setElapsed] = useState(0);
  const inputRef = useRef(null);
  const abortRef = useRef(null);
  const lastFileRef = useRef(null); // last uploaded (already redacted) file, for re-analysis

  useEffect(() => () => sentUrl && URL.revokeObjectURL(sentUrl), [sentUrl]);

  useEffect(() => {
    if (mode !== "loading") return undefined;
    setElapsed(0);
    const id = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, [mode]);

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

  async function upload(f, useMode = sens) {
    lastFileRef.current = f;
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setSentUrl(URL.createObjectURL(f));
    setMode("loading");
    try {
      setResult(await analyze(f, ctrl.signal, useMode));
      setMode("result");
    } catch (e) {
      if (e.name !== "AbortError") {
        setResult({ status: "error", message: "The request failed. Please try again.", findings: [], warnings: [] });
        setMode("result");
      }
    }
  }

  function rerun() {
    const next = sens === "screening" ? "balanced" : "screening";
    setSens(next);
    if (lastFileRef.current) upload(lastFileRef.current, next);
  }

  function pickDemo(f) {
    const err = validateFile(f);
    if (err) setInputError(err);
    else upload(f); // demo images are pre-vetted: one click straight to analysis
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <nav className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-2 px-4 py-3">
          <Logo />
          <div className="flex items-center gap-3 text-sm">
            <span className="hidden rounded bg-amber-50 border border-amber-200 px-2.5 py-1 text-xs font-semibold text-amber-800 sm:inline">
              Not a diagnostic tool
            </span>
            <label className="flex cursor-pointer items-center gap-2 text-slate-700">
              <input type="checkbox" checked={demoMode} onChange={(e) => setDemoMode(e.target.checked)} className="h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500" />
              Demo mode
            </label>
          </div>
        </div>
      </nav>

      <main className="mx-auto max-w-6xl px-4 py-6">
        {mode === "idle" && (
          <section className="mb-6 rounded-xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
            <div className="mb-2.5 flex flex-wrap items-center gap-2">
              <span className="inline-block rounded border border-blue-200 bg-blue-50 px-2.5 py-0.5 text-xs font-semibold text-blue-700">
                Screening &amp; teaching aid
              </span>
            </div>
            <h2 className="text-xl font-bold tracking-tight text-slate-900 sm:text-2xl">
              Look for suspected wrist and forearm fractures in a photo of an X-ray film.
            </h2>
            <p className="mt-2 text-sm text-slate-600 sm:text-base leading-relaxed">
              Upload or capture a photo. OrthoLens cleans it, runs two AI checks and highlights regions of interest. A qualified doctor must confirm every finding.
            </p>
            <div className="mt-4 flex flex-wrap gap-2 text-xs text-slate-600">
              <span className="rounded border border-slate-200 bg-slate-50 px-2.5 py-1">
                Processed in memory, nothing stored
              </span>
              <span className="rounded border border-slate-200 bg-slate-50 px-2.5 py-1">
                Blur names before upload
              </span>
              <span className="rounded border border-slate-200 bg-slate-50 px-2.5 py-1">
                JPEG or PNG, up to 10 MB
              </span>
            </div>
          </section>
        )}

        <div className="mb-5 card !py-3">
          <Stepper current={STEP_OF[mode]} />
        </div>

        <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_320px]">
          <div className="min-w-0 space-y-4">
            {mode === "result" && result ? (
              <Results result={result} onReset={reset} onRerun={lastFileRef.current ? rerun : null} />
            ) : (
              <>
                <div className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm font-medium text-amber-900" role="alert">
                  Blur or crop names and patient IDs before uploading.
                </div>

                {mode === "idle" && (
                  <div className="space-y-4">
                    {demoMode && <DemoGallery onPick={pickDemo} disabled={false} />}
                    <div
                      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                      onDragLeave={() => setDragging(false)}
                      onDrop={(e) => { e.preventDefault(); setDragging(false); choose(e.dataTransfer.files?.[0]); }}
                      className={`rounded-xl border-2 border-dashed p-8 text-center transition sm:p-12 ${
                        dragging ? "border-blue-500 bg-blue-50" : "border-slate-300 bg-white hover:border-slate-400"
                      }`}
                    >
                      <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-slate-100 text-slate-600" aria-hidden="true">
                        <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M12 16V4m0 0L7 9m5-5 5 5" />
                          <path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />
                        </svg>
                      </div>
                      <p className="text-base font-semibold text-slate-800">Drag and drop an X-ray photo here</p>
                      <p className="mt-1 text-xs text-slate-500">JPEG or PNG, max 10 MB</p>
                      <div className="mt-5 flex flex-col justify-center gap-3 sm:flex-row">
                        <button onClick={() => inputRef.current?.click()} className="btn-primary">Choose file</button>
                        <button onClick={() => setMode("camera")} className="btn-secondary">Take photo</button>
                      </div>
                      <input
                        ref={inputRef}
                        type="file"
                        accept="image/jpeg,image/png"
                        className="hidden"
                        onChange={(e) => { choose(e.target.files?.[0]); e.target.value = ""; }}
                      />
                    </div>
                    {inputError && <p className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800" role="alert">{inputError}</p>}
                  </div>
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
                  <div className="card space-y-4">
                    <div className="overflow-hidden rounded-lg border border-slate-200 bg-slate-900">
                      <img src={sentUrl} alt="Image being analysed" className="mx-auto max-h-[60vh] w-full object-contain" />
                    </div>
                    <div className="flex items-center gap-3 p-2" role="status">
                      <span className="h-5 w-5 animate-spin rounded-full border-2 border-blue-600 border-t-transparent" />
                      <div>
                        <p className="text-sm font-semibold text-slate-800">Analysing image… {elapsed}s</p>
                        <p className="text-xs text-slate-500">Cleaning photo, running detector and classifier. Usually takes a few seconds.</p>
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>

          <aside className="space-y-4">
            <SensitivityCard sens={sens} setSens={setSens} />
            <HowItWorks />
          </aside>
        </div>

        <div className="mt-6">
          <Limitations />
        </div>
      </main>

      <footer className="mx-auto max-w-6xl px-4 pb-8 text-center text-xs text-slate-500">
        OrthoLens is a screening and teaching aid, not a diagnosis. It gives no treatment, first-aid or referral advice.
      </footer>
    </div>
  );
}
