import { useEffect, useState } from "react";

/** Demo gallery: lists images from /demo/index.json. Hidden when there are none. */
export default function DemoGallery({ onPick, disabled }) {
  const [files, setFiles] = useState([]);

  useEffect(() => {
    let alive = true;
    fetch("/demo/index.json", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : []))
      .then((list) => alive && Array.isArray(list) && setFiles(list))
      .catch(() => {});
    return () => { alive = false; };
  }, []);

  if (files.length === 0) return null;

  async function pick(name) {
    const res = await fetch(`/demo/${encodeURIComponent(name)}`);
    const blob = await res.blob();
    onPick(new File([blob], name, { type: blob.type || "image/jpeg" }));
  }

  return (
    <div className="card border-slate-200 bg-slate-50">
      <p className="mb-3 text-xs font-bold uppercase tracking-wider text-slate-600">Sample Demo Images (click to analyse)</p>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4">
        {files.map((name) => (
          <button
            key={name}
            disabled={disabled}
            onClick={() => pick(name)}
            className="group overflow-hidden rounded-lg border border-slate-200 bg-white text-left shadow-sm transition hover:border-blue-400 disabled:opacity-50"
          >
            <img src={`/demo/${encodeURIComponent(name)}`} alt={name} loading="lazy" className="aspect-square w-full object-cover" />
            <span className="block truncate px-2 py-1.5 text-xs text-slate-700 font-medium">{name}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
