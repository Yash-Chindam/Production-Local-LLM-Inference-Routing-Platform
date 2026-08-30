import { expect, test } from "@playwright/test";

test("serves health and authenticated model catalog", async ({ request }) => {
  const health = await request.get("/healthz");
  expect(health.ok()).toBeTruthy();
  await expect(health.json()).resolves.toEqual({ status: "healthy" });

  const models = await request.get("/v1/models");
  expect(models.ok()).toBeTruthy();
  expect((await models.json()).data).toEqual(
    expect.arrayContaining([expect.objectContaining({ id: "general-local" })]),
  );
});

test("routes an OpenAI-compatible extraction request end to end", async ({ request }) => {
  const response = await request.post("/v1/chat/completions", {
    data: {
      model: "auto",
      messages: [{ role: "user", content: "Extract the account fields as JSON" }],
      routing: { privacy: "restricted", latency_tier: "interactive" },
    },
  });
  expect(response.status()).toBe(200);
  expect(response.headers()["x-route-model"]).toBe("small-specialist");
  const body = await response.json();
  expect(body).toMatchObject({
    object: "chat.completion",
    model: "small-specialist",
    routing: { task: "extraction" },
  });
  expect(body.routing.model_revision).toBeTruthy();
  expect(body.choices[0].message.role).toBe("assistant");
});

test("rejects invalid credentials", async ({ playwright }, testInfo) => {
  const anonymous = await playwright.request.newContext({
    baseURL: testInfo.project.use.baseURL,
    extraHTTPHeaders: { Authorization: "Bearer invalid-key" },
  });
  const response = await anonymous.get("/v1/models");
  expect(response.status()).toBe(401);
  await anonymous.dispose();
});

test("publishes scrape-ready metrics for a completed request", async ({ request }) => {
  const completion = await request.post("/v1/chat/completions", {
    data: {
      model: "auto",
      messages: [{ role: "user", content: "Summarize the quarterly report" }],
    },
  });
  expect(completion.status()).toBe(200);
  const routedModel = completion.headers()["x-route-model"];

  const metrics = await request.get("/metrics");
  expect(metrics.status()).toBe(200);
  expect(metrics.headers()["content-type"]).toContain("text/plain");

  const body = await metrics.text();
  expect(body).toContain(`router_requests_total{model="${routedModel}"`);
  expect(body).toContain("router_request_latency_seconds_bucket");
  expect(body).toContain("router_tokens_total");
  expect(body).toContain("router_predicted_quality_sum");
});
