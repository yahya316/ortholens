const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

export const MAX_BYTES = 10 * 1024 * 1024;
export const ALLOWED_TYPES = ["image/jpeg", "image/png"];

/** Client-side pre-check. Returns an error string or null. */
export function validateFile(file) {
  if (!ALLOWED_TYPES.includes(file.type)) return "Unsupported file type. Please use a JPEG or PNG image.";
  if (file.size > MAX_BYTES) return "File is too large. Maximum size is 10 MB.";
  return null;
}

/** Upload an image for analysis. Always resolves to an analysis-shaped object. */
export async function analyze(file, signal, mode = "screening") {
  const form = new FormData();
  form.append("file", file, "upload");
  form.append("mode", mode);
  let res;
  try {
    res = await fetch(`${API_URL}/api/analyses`, { method: "POST", body: form, signal });
  } catch (e) {
    if (e.name === "AbortError") throw e;
    return { status: "error", message: `Cannot reach the OrthoLens server at ${API_URL}. Is the backend running?`, findings: [] };
  }
  let data = null;
  try {
    data = await res.json();
  } catch {
    /* non-JSON body */
  }
  if (!data || !data.status) {
    return { status: "error", message: `Unexpected server response (HTTP ${res.status}).`, findings: [] };
  }
  return data;
}
