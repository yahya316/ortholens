export const LIMITATIONS_TEXT =
  "Trained mainly on public X-ray datasets. Accuracy on phone photographs of hospital films is still being measured. Double-view photos, glare and non-wrist body parts can give wrong results. Always confirm with a doctor.";

export default function Limitations() {
  return (
    <details className="card group text-sm">
      <summary className="flex cursor-pointer list-none items-center justify-between font-semibold">
        <span>Limitations</span>
        <span className="text-slate-400 transition group-open:rotate-180" aria-hidden="true">▾</span>
      </summary>
      <p className="mt-3 text-slate-700">{LIMITATIONS_TEXT}</p>
    </details>
  );
}
