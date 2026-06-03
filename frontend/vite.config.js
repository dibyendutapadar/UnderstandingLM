import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base: "./" keeps asset paths relative so the static build works on
// GitHub Pages / Netlify / Vercel without extra config.
export default defineConfig({
  plugins: [react()],
  base: "./",
});
