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
    <div className="rounded-lg border border-violet-300 bg-violet-50 p-3">
      <p className="mb-2 text-sm font-medium text-violet-900">Demo images (click to analyse)</p>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 md:grid-cols-4">
        {files.map((name) => (
          <button
            key={name}
            disabled={disabled}
            onClick={() => pick(name)}
            className="overflow-hidden rounded border border-violet-200 bg-white text-left disabled:opacity-50"
          >
            <img src={`/demo/${encodeURIComponent(name)}`} alt={name} loading="lazy" className="aspect-square w-full object-cover" />
            <span className="block truncate px-1 py-0.5 text-xs">{name}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
