import { defineConfig } from "vite";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";
import basicSsl from "@vitejs/plugin-basic-ssl";

const __dirname = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  // Demo 配置
  root: "demo",
  publicDir: "../public",
  build: {
    outDir: "../dist-demo",
    target: "esnext",
  },
  resolve: {
    alias: {
      "@lib": resolve(__dirname, "src"),
    },
  },
  plugins: [basicSsl()],
  server: {
    port: 3000,
    host: true, // 允许内网设备访问
  },
});
