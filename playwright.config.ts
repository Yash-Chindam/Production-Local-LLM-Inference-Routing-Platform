import { defineConfig } from "@playwright/test";

const port = process.env.TEST_PORT ?? "8000";
const baseURL = `http://127.0.0.1:${port}`;

export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: process.env.CI ? [["github"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL,
    extraHTTPHeaders: { Authorization: "Bearer e2e-key" },
    trace: "retain-on-failure",
  },
  webServer: {
    command: `python -m uvicorn llm_router.app:app --app-dir src --host 127.0.0.1 --port ${port}`,
    url: `${baseURL}/healthz`,
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
    env: {
      ...process.env,
      ROUTER_API_KEYS: "e2e-key",
      ROUTER_ENVIRONMENT: "test",
    },
  },
});
