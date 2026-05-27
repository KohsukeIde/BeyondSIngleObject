import path from "path"
import { reactRouter } from "@react-router/dev/vite";
import tailwindcss from "@tailwindcss/vite"
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig(({ mode }) => ({
  // Use root path for development, GitHub Pages path for production
  base: mode === 'production' ? '/BeyondSingleObject/' : '/',
  plugins: [reactRouter(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    // Allow serving large files
    hmr: {
      overlay: true
    },
    // Increase file size limits
    fs: {
      strict: false
    }
  },
  build: {
    // Don't inline large assets
    assetsInlineLimit: 0,
    // Chunk size warning limit
    chunkSizeWarningLimit: 1000
  },
  // Explicitly set public directory
  publicDir: 'public'
}))
