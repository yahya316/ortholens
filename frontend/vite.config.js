import fs from "node:fs";
import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const IMG = /\.(jpe?g|png)$/i;

/** Serves /demo/index.json listing images in public/demo (live in dev, emitted at build). */
function demoManifest() {
  const dir = path.resolve(process.cwd(), "public/demo");
  const list = () => (fs.existsSync(dir) ? fs.readdirSync(dir).filter((f) => IMG.test(f)).sort() : []);
  return {
    name: "demo-manifest",
    configureServer(server) {
      server.middlewares.use("/demo/index.json", (_req, res) => {
        res.setHeader("Content-Type", "application/json");
        res.setHeader("Cache-Control", "no-store");
        res.end(JSON.stringify(list()));
      });
    },
    generateBundle() {
      this.emitFile({ type: "asset", fileName: "demo/index.json", source: JSON.stringify(list()) });
    },
  };
}

export default defineConfig({
  plugins: [react(), tailwindcss(), demoManifest()],
  server: { port: 5173 },
});
