// Sustained load definition (section 16). Reproducible from this committed file:
//   k6 run -e BASE_URL=http://127.0.0.1:8000 -e API_KEY=dev-key benchmarks/workloads/steady.js
import http from "k6/http";
import { check } from "k6";

export const options = {
  scenarios: {
    steady: {
      executor: "constant-arrival-rate",
      rate: 20,
      timeUnit: "1s",
      duration: "5m",
      preAllocatedVUs: 40,
      maxVUs: 120,
    },
  },
  thresholds: {
    // Report quality and latency together; a passing run is not a quality claim.
    "http_req_duration{expected_response:true}": ["p(95)<2000", "p(99)<5000"],
    "http_req_failed": ["rate<0.01"],
  },
};

const BASE_URL = __ENV.BASE_URL || "http://127.0.0.1:8000";
const API_KEY = __ENV.API_KEY || "dev-key";

// Prompt-length distribution: short classification, medium extraction, long RAG.
const PROMPTS = [
  { task: "classification", text: "Classify this ticket: my card was charged twice." },
  { task: "extraction", text: "Extract the invoice number from: Invoice INV-4417, total 182.50 USD." },
  {
    task: "rag",
    text: "According to the documents provided, summarize the retention policy. ".repeat(24),
  },
];

export default function () {
  const prompt = PROMPTS[Math.floor(Math.random() * PROMPTS.length)];
  const response = http.post(
    `${BASE_URL}/v1/chat/completions`,
    JSON.stringify({
      model: "auto",
      messages: [{ role: "user", content: prompt.text }],
      max_tokens: 128,
      routing: { task: prompt.task, privacy: "private" },
    }),
    { headers: { "Content-Type": "application/json", Authorization: `Bearer ${API_KEY}` } },
  );

  check(response, {
    "served or rejected predictably": (r) => r.status === 200 || r.status === 503 || r.status === 429,
    "overload carries retry guidance": (r) => r.status !== 503 || r.headers["Retry-After"] !== undefined,
    "route is attributed": (r) => r.status !== 200 || r.headers["X-Route-Model"] !== undefined,
  });
}
