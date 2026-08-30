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
