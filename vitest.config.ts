import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright'

export default defineConfig({
  test: {
    browser: {
      enabled: true,
      headless: false,
      provider: playwright(),
      instances: [{ browser: 'chromium' }]
    },
    testTimeout: 120_000,
    hookTimeout: 60_000,
    watch: true
  },
  publicDir: 'scripts',
  optimizeDeps: {
    exclude: ['onnxruntime-web']
  }
});