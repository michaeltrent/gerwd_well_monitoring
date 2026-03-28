import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: './',          // relative paths → works on any domain/subdirectory
  server: { port: 3000 },
})
