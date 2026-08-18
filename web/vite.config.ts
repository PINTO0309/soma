import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Dev port: 5273 (5173 is commonly taken by other vite projects / containers).
export const DEV_PORT = 5273;

export default defineConfig({
  base: './',
  plugins: [react()],
  server: {
    port: DEV_PORT,
    strictPort: true,
  },
});
