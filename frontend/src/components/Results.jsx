const STATUS_STYLE = {
  ok: ["border-blue-300 bg-blue-50 text-blue-900"],
  no_fracture_detected: ["border-emerald-300 bg-emerald-50 text-emerald-900"],
  retake: ["border-amber-300 bg-amber-50 text-amber-900"],
  out_of_scope: ["border-amber-300 bg-amber-50 text-amber-900"],
  error: ["border-red-300 bg-red-50 text-red-900"],
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
    <figure className="min-w-0 rounded-lg border border-slate-200 bg-white p-2">
      <figcaption className="mb-2 text-sm font-medium text-slate-600">{title}</figcaption>
      <img src={src} alt={title} className="w-full rounded bg-black object-contain" />
    </figure>
  );
}

export default function Results({ result, onReset }) {
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
      <div className="rounded-lg border-2 border-red-600 bg-red-50 p-3 text-sm font-semibold text-red-900" role="note">
        {result.notice || DEFAULT_NOTICE}
      </div>

      {bodyPart.length > 0 && (
        <div className="rounded-lg border border-yellow-400 bg-yellow-100 p-3 text-sm font-medium text-yellow-900" role="alert">
          {bodyPart.map((w, i) => <p key={i}>{w}</p>)}
        </div>
      )}

      {otherWarnings.length > 0 && (
        <div className="rounded-lg border border-yellow-400 bg-yellow-100 p-3 text-sm text-yellow-900" role="alert">
          <p className="font-semibold">Warnings</p>
          <ul className="ml-5 list-disc">
            {otherWarnings.map((w, i) => <li key={i}>{w}</li>)}
          </ul>
        </div>
      )}

      <div className={`rounded-lg border p-4 ${style}`}>
        <h2 className="text-xl font-bold">{decisionTitle}</h2>
        {scores && (
          <p className="mt-1 text-sm">
            Suspicion score {scores.combined.toFixed(2)}{" "}
            <span className="opacity-80">(relative scale, not a probability)</span>{" "}
            <span
              className="ml-1 inline-flex h-5 w-5 cursor-help items-center justify-center rounded-full border border-current text-xs font-bold"
              title={SCORE_TOOLTIP}
              aria-label={SCORE_TOOLTIP}
              role="img"
              tabIndex={0}
            >
              ?
            </span>
            <span className="ml-2 opacity-80">
              Cut-off {scores.threshold.toFixed(2)} ({result.decision_mode === "balanced" ? "Balanced" : "Screening"})
            </span>
          </p>
        )}
        <p className="mt-2">{result.message}</p>
        {result.status !== "error" && total != null && (
          <p className="mt-2 text-xs opacity-80">
            Processed in {(total / 1000).toFixed(2)} s{result.model_version && result.model_version !== "none" ? ` · Model: ${result.model_version}` : ""}
          </p>
        )}
      </div>

      {hasImages && result.status === "retake" && <ImagePanel title="Your photo" src={result.original_image} />}
      {hasImages && result.status !== "retake" && result.status !== "error" && (
        <ImagePanel title={findings.length ? "Your photo with suspected regions" : "Your photo (de-identified)"} src={result.annotated_image || result.original_image} />
      )}

      {result.cleaned_image && result.status !== "retake" && result.status !== "error" && (
        <details className="rounded-lg border border-slate-200 bg-white p-3 text-sm">
          <summary className="cursor-pointer font-medium">What the AI saw (cleaned image)</summary>
          <p className="mt-2 text-xs text-slate-600">The classifier looks at this cleaned, contrast-normalised version. The detector uses your original photo.</p>
          <img src={result.cleaned_image} alt="Cleaned image used by the classifier" className="mt-2 w-full rounded bg-black object-contain" />
        </details>
      )}

      {findings.length > 0 && (
        <div className="rounded-lg border border-slate-200 bg-white p-4">
          <h3 className="mb-2 font-semibold">Findings</h3>
          <ul className="divide-y divide-slate-100">
            {findings.map((f, i) => (
              <li key={i} className="flex flex-wrap items-center gap-x-3 gap-y-1 py-2 text-sm">
                <span
                  className={`inline-block h-3 w-3 shrink-0 rounded-sm ${f.low_confidence ? "border-2 border-dashed border-orange-500" : "bg-blue-600"}`}
                  aria-hidden="true"
                />
                <span className="font-medium">Suspected fracture {i + 1}</span>
                {f.view > 0 && <span>view {f.view + 1}</span>}
                <span className="font-semibold">box confidence {f.confidence.toFixed(2)}</span>
                {f.low_confidence && (
                  <span className="rounded bg-orange-100 px-2 py-0.5 text-xs font-semibold text-orange-800">low confidence</span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {result.thresholds && result.status !== "error" && (
        <p className="text-xs text-slate-500">
          Boxes shown at detector confidence ≥ {result.thresholds.conf}, low confidence below {result.thresholds.low_conf}, image size {result.thresholds.imgsz}, quality mode {result.thresholds.quality_mode}.
          {result.quality?.sharpness != null && ` Sharpness ${result.quality.sharpness}, contrast ${result.quality.contrast}.`}
          {result.timings_ms && ` Steps: clean ${result.timings_ms.clean} ms, detector ${result.timings_ms.yolo} ms, classifier ${result.timings_ms.resnet} ms.`}
        </p>
      )}

      <button onClick={onReset} className="w-full rounded bg-blue-600 px-4 py-3 font-medium text-white sm:w-auto">
        Try another image
      </button>
    </section>
  );
}
