import { defineConfig } from 'vitest/config';
import { resolve } from 'path';

export default defineConfig({
  test: {
    include: ['src/**/*.{test,spec}.{js,ts}'],
    exclude: ['node_modules', 'dist', 'demo'],
  },
  resolve: {
    alias: {
      '@lib': resolve(__dirname, 'src'),
    },
  },
});
