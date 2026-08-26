import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import basicSsl from '@vitejs/plugin-basic-ssl'
import { defineConfig } from 'vite'

// basicSsl + host:true + the /api proxy below are for testing the live-camera
// workout screen from a phone on the same Wi-Fi — getUserMedia requires a
// secure context, and a plain http://<lan-ip> origin doesn't qualify.
export default defineConfig({
  plugins: [react(), tailwindcss(), basicSsl()],
  server: {
    host: true,
    proxy: {
      '/api': { target: 'http://localhost:5000', changeOrigin: true },
    },
  },
  // onnxruntime-web's internal `new URL(..., import.meta.url)` references to
  // its own wasm variants otherwise get picked up by Vite's asset scanner and
  // copied into dist/ wholesale (27MB+ for a variant we don't even use) — we
  // ship our own trimmed copy from node_modules ourselves, see public/ort/
  // and phaseClassifier.ts's `ort.env.wasm.wasmPaths`.
  //
  // NOTE: the phase-classifier ONNX model only loads correctly under
  // `vite build && vite preview` (or the real deployment) — Vite's *dev*
  // server middleware intercepts the dynamic `import()` of onnxruntime-web's
  // own wasm-loader .mjs (even from public/) and fails to serve it
  // ("Failed to fetch dynamically imported module"). Confirmed working
  // fine on a plain static server; this is dev-server-only. If you need to
  // test the ONNX path locally, build + preview instead of `npm run dev`.
  optimizeDeps: { exclude: ['onnxruntime-web'] },
})
