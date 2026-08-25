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
})
