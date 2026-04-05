import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    // Raise warning threshold (leaflet is large by design)
    chunkSizeWarningLimit: 800,
    rollupOptions: {
      output: {
        // Manual chunks: each group gets its own cacheable file
        // Browser only re-downloads the chunk that changed
        manualChunks: {
          // Leaflet + react-leaflet — large, changes rarely
          'vendor-leaflet': ['leaflet', 'react-leaflet'],
          // Recharts — large, changes rarely
          'vendor-recharts': ['recharts'],
          // i18n stack
          'vendor-i18n': ['i18next', 'react-i18next', 'i18next-browser-languagedetector'],
          // Icons
          'vendor-icons': ['lucide-react'],
          // HTTP client
          'vendor-axios': ['axios'],
        },
      },
    },
  },
})

