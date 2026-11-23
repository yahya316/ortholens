export const LIMITATIONS_TEXT =
  "Trained mainly on public X-ray datasets. Accuracy on phone photographs of hospital films is still being measured. Double-view photos, glare and non-wrist body parts can give wrong results. Always confirm with a doctor.";

export default function Limitations() {
  return (
    <details className="rounded-lg border border-slate-300 bg-white p-3 text-sm">
      <summary className="cursor-pointer font-medium">Limitations</summary>
      <p className="mt-2 text-slate-700">{LIMITATIONS_TEXT}</p>
    </details>
  );
}
