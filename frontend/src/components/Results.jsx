const STATUS_STYLE = {
  ok: ["border-blue-200 bg-blue-50/70 text-blue-950"],
  no_fracture_detected: ["border-emerald-200 bg-emerald-50/70 text-emerald-950"],
  retake: ["border-amber-200 bg-amber-50/70 text-amber-950"],
  out_of_scope: ["border-amber-200 bg-amber-50/70 text-amber-950"],
  error: ["border-red-200 bg-red-50/70 text-red-950"],
};

const TITLES = {
  retake: "Please retake the photo",
  out_of_scope: "Unsupported body region",
  error: "Something went wrong",
};

const DEFAULT_NOTICE = "OrthoLens is a screening and teaching aid, not a diagnosis. A qualified doctor must confirm every finding.";
const BODY_PART_PREFIX = "Body-part check not installed";

export const SCORE_TOOLTIP =
  "A rank-based score from 0 to 1: how this photo's detector and classifier outputs rank against a reference set of photos, " +
  "combined with fixed weights. It is NOT a probability and not a percentage; 0.60 does not mean 60% chance of a fracture. " +
  "The Sensitivity setting only changes the cut-off applied to this score.";

function ImagePanel({ title, src }) {
  return (
    <figure className="min-w-0 rounded-xl border border-slate-200 bg-white p-3 shadow-sm">
      <figcaption className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-500">{title}</figcaption>
      <div className="overflow-hidden rounded-lg bg-slate-950">
        <img src={src} alt={title} className="mx-auto max-h-[60vh] w-full object-contain" />
      </div>
    </figure>
  );
}

function ScoreBar({ score, threshold }) {
  const pos = (v) => `${Math.min(Math.max(v, 0), 1) * 100}%`;
  return (
    <div className="mt-3.5" role="img" aria-label={`Suspicion score ${score.toFixed(2)} on a scale from 0 to 1, cut-off ${threshold.toFixed(2)}`}>
      <div className="relative h-2.5 rounded-full bg-slate-200">
        <div
          className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-emerald-400 via-amber-400 to-rose-500"
          style={{ width: `${Math.min(Math.max(score, 0), 1) * 100}%` }}
        />
        <div
          className="absolute -top-1 bottom-[-4px] w-0.5 bg-slate-800 z-10"
          style={{ left: pos(threshold) }}
          title={`Cut-off ${threshold.toFixed(2)}`}
        />
      </div>
      <div className="mt-1.5 flex justify-between text-[11px] font-medium text-slate-500">
        <span>0 (low)</span>
        <span className="text-slate-700">cut-off: {threshold.toFixed(2)}</span>
        <span>1 (high)</span>
      </div>
    </div>
  );
}

export default function Results({ result, onReset, onRerun }) {
  const decision = result.decision; // "fracture" | "no_fracture" | null
  const style = (STATUS_STYLE[result.status] || STATUS_STYLE.error)[0];
  const hasImages = Boolean(result.original_image);
  const findings = result.findings || [];
  const warnings = result.warnings || [];
  const bodyPart = warnings.filter((w) => w.startsWith(BODY_PART_PREFIX));
  const otherWarnings = warnings.filter((w) => !w.startsWith(BODY_PART_PREFIX));
  const scores = result.scores;
  const total = result.timings_ms?.total ?? result.elapsed_ms;
  const decisionTitle =
    decision === "fracture" ? "Fracture suspected" : decision === "no_fracture" ? "No fracture detected" : TITLES[result.status] || "Result";

  return (
    <section className="space-y-4" aria-live="polite">
      {/* Notice */}
      <div className="rounded-lg border border-red-300 bg-red-50 p-3 text-xs font-semibold text-red-900" role="note">
        {result.notice || DEFAULT_NOTICE}
      </div>

      {/* Scope Warning */}
      {bodyPart.length > 0 && (
        <div className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-xs font-medium text-amber-900" role="alert">
          {bodyPart.map((w, i) => <p key={i}>{w}</p>)}
        </div>
      )}

      {/* Quality/Other Warnings */}
      {otherWarnings.length > 0 && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-900" role="alert">
          <p className="font-semibold">Quality notes:</p>
          <ul className="mt-1 ml-4 list-disc space-y-0.5">
            {otherWarnings.map((w, i) => <li key={i}>{w}</li>)}
          </ul>
        </div>
      )}

      {/* Primary Decision Banner */}
      <div className={`rounded-xl border p-5 shadow-sm ${style}`}>
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-xl font-bold tracking-tight">{decisionTitle}</h2>
          {scores && (
            <span className="rounded bg-white/80 border border-current px-2 py-0.5 text-xs font-semibold">
              Mode: {result.decision_mode === "balanced" ? "Balanced" : "Screening"}
            </span>
          )}
        </div>

        {scores && (
          <div className="mt-2 text-sm">
            <span className="font-semibold">Suspicion score {scores.combined.toFixed(2)}</span>{" "}
            <span className="text-xs opacity-75">(relative scale, not a probability)</span>{" "}
            <span
              className="inline-flex h-4 w-4 cursor-help items-center justify-center rounded-full border border-current text-[10px] font-bold"
              title={SCORE_TOOLTIP}
              aria-label={SCORE_TOOLTIP}
              role="img"
              tabIndex={0}
            >
              ?
            </span>
          </div>
        )}

        {scores && <ScoreBar score={scores.combined} threshold={scores.threshold} />}

        <p className="mt-3 text-sm leading-relaxed">{result.message}</p>

        {result.status !== "error" && total != null && (
          <p className="mt-3 text-xs opacity-70">
            Analysis time: {(total / 1000).toFixed(2)}s{result.model_version && result.model_version !== "none" ? ` · Model: ${result.model_version}` : ""}
          </p>
        )}
      </div>

      {/* Visual results */}
      {hasImages && result.status === "retake" && <ImagePanel title="Uploaded photo" src={result.original_image} />}
      {hasImages && result.status !== "retake" && result.status !== "error" && (
        <ImagePanel title={findings.length ? "Analyzed photo with suspected regions" : "Analyzed photo (de-identified)"} src={result.annotated_image || result.original_image} />
      )}

      {/* Cleaned image toggle */}
      {result.cleaned_image && result.status !== "retake" && result.status !== "error" && (
        <details className="card text-sm">
          <summary className="cursor-pointer font-medium text-slate-800 hover:text-blue-600">
            What the AI saw (cleaned image)
          </summary>
          <p className="mt-2 text-xs text-slate-500">
            The classifier processes this cleaned, contrast-flattened version to reduce glare/lighting bias. The detector identifies boxes on the original photo.
          </p>
          <div className="mt-2.5 overflow-hidden rounded-lg bg-slate-950">
            <img src={result.cleaned_image} alt="Cleaned image used by the classifier" className="mx-auto max-h-[50vh] w-full object-contain" />
          </div>
        </details>
      )}

      {/* Detected regions list */}
      {findings.length > 0 && (
        <div className="card">
          <h3 className="mb-2.5 text-xs font-bold uppercase tracking-wider text-slate-500">Detected Regions ({findings.length})</h3>
          <ul className="divide-y divide-slate-100">
            {findings.map((f, i) => (
              <li key={i} className="flex flex-wrap items-center justify-between gap-2 py-2 text-sm">
                <div className="flex items-center gap-2">
                  <span
                    className={`inline-block h-2.5 w-2.5 rounded-full ${f.low_confidence ? "bg-orange-500" : "bg-blue-600"}`}
                    aria-hidden="true"
                  />
                  <span className="font-medium text-slate-800">Suspected fracture {i + 1}</span>
                  {f.view > 0 && <span className="text-xs text-slate-500">(view {f.view + 1})</span>}
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-slate-700">confidence: {f.confidence.toFixed(2)}</span>
                  {f.low_confidence && (
                    <span className="rounded bg-orange-50 border border-orange-200 px-2 py-0.5 text-[11px] font-medium text-orange-800">
                      low confidence
                    </span>
                  )}
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Threshold and diagnostics detail */}
      {result.thresholds && result.status !== "error" && (
        <div className="rounded-lg bg-slate-100/70 p-3 text-[11px] text-slate-500 space-y-1">
          <p>
            Box display confidence ≥ {result.thresholds.conf} · Low confidence below {result.thresholds.low_conf} · Image size {result.thresholds.imgsz}px
            {result.quality?.sharpness != null && ` · Sharpness ${result.quality.sharpness} · Contrast ${result.quality.contrast}`}
          </p>
          {result.timings_ms && (
            <p className="text-slate-400">
              Pipeline breakdown: clean {result.timings_ms.clean}ms, detector {result.timings_ms.yolo}ms, classifier {result.timings_ms.resnet}ms
            </p>
          )}
        </div>
      )}

      {/* Action buttons */}
      <div className="flex flex-col gap-3 pt-2 sm:flex-row">
        <button onClick={onReset} className="btn-primary">
          Try another image
        </button>
        {onRerun && scores && (
          <button onClick={onRerun} className="btn-secondary">
            Re-analyse with {result.decision_mode === "balanced" ? "Screening" : "Balanced"} sensitivity
          </button>
        )}
        {result.annotated_image && result.status !== "error" && result.status !== "retake" && (
          <a href={result.annotated_image} download="ortholens-result.jpg" className="btn-secondary text-center">
            Save annotated image
          </a>
        )}
      </div>
    </section>
  );
}
