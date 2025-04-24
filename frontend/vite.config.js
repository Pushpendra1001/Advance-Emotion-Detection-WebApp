import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
        }
      }
    }
  },
  server: {
    port: 5173,
    host: true,
    cors: {
      origin: [
        'https://emotiondetector-wlit.onrender.com',
        'https://emotiondetector-1.onrender.com'
      ],
      credentials: true
    },
    proxy: {
      '/api': {
        target: 'https://emotiondetector-1.onrender.com',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  },
  publicDir: 'public',
})
