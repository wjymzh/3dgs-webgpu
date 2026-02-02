import { defineConfig } from "vite";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  build: {
    lib: {
      entry: resolve(__dirname, "src/index.ts"),
      name: "D5Techs3DGSLib",
      fileName: "3dgs-lib",
      formats: ["es", "cjs"],
    },
    outDir: "dist",
    sourcemap: true,
    minify: false,
    rollupOptions: {
      // 不打包 wgsl 文件作为外部依赖，而是内联
      output: {
        // 保持模块结构
        preserveModules: false,
      },
    },
  },
  resolve: {
    alias: {
      "@lib": resolve(__dirname, "src"),
    },
  },
  // 处理 .wgsl 文件作为原始字符串导入
  assetsInclude: ["**/*.wgsl"],
});
