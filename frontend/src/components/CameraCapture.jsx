import { useEffect, useRef, useState } from "react";

/** Live camera (rear preferred) with capture. Calls onCapture(File) or onCancel(). */
export default function CameraCapture({ onCapture, onCancel }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [error, setError] = useState(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    async function start() {
      if (!navigator.mediaDevices?.getUserMedia) {
        setError("Camera access is not available in this browser or on this connection (HTTPS is required except on localhost). Use 'Choose file' instead.");
        return;
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        videoRef.current.srcObject = stream;
      } catch {
        setError("Could not open the camera. Check browser permissions, or use 'Choose file' instead.");
      }
    }
    start();
    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  function capture() {
    const v = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = v.videoWidth;
    canvas.height = v.videoHeight;
    canvas.getContext("2d").drawImage(v, 0, 0);
    canvas.toBlob(
      (blob) => blob && onCapture(new File([blob], "capture.jpg", { type: "image/jpeg" })),
      "image/jpeg",
      0.92
    );
  }

  if (error) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">
        <p>{error}</p>
        <button onClick={onCancel} className="mt-3 rounded bg-slate-800 px-4 py-2 text-white">Back</button>
      </div>
    );
  }
  return (
    <div className="space-y-3">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        onLoadedMetadata={() => setReady(true)}
        className="w-full rounded-lg bg-black"
      />
      <div className="flex gap-3">
        <button onClick={capture} disabled={!ready} className="flex-1 rounded bg-blue-600 px-4 py-3 font-medium text-white disabled:opacity-50">
          Capture
        </button>
        <button onClick={onCancel} className="rounded border border-slate-300 px-4 py-3">Cancel</button>
      </div>
    </div>
  );
}
