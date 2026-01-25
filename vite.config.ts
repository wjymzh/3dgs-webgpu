import { defineConfig } from "vite";
import { resolve } from "path";
import basicSsl from "@vitejs/plugin-basic-ssl";

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
    https: true, // WebGPU 需要安全上下文
  },
});
