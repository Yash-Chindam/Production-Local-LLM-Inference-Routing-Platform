// Burst load definition (section 16): verifies bounded queue behaviour and
// predictable rejection rather than unbounded tail latency.
//   k6 run -e BASE_URL=http://127.0.0.1:8000 -e API_KEY=dev-key benchmarks/workloads/burst.js
import http from "k6/http";
import { check } from "k6";

export const options = {
  scenarios: {
    burst: {
      executor: "ramping-arrival-rate",
      startRate: 10,
      timeUnit: "1s",
      preAllocatedVUs: 100,
      maxVUs: 400,
      stages: [
        { target: 10, duration: "1m" },
        { target: 200, duration: "30s" },
        { target: 200, duration: "1m" },
        { target: 10, duration: "1m" },
      ],
    },
  },
  thresholds: {
    // Under saturation the platform must reject predictably, not queue without bound.
    "http_req_duration": ["p(99)<10000"],
    "checks": ["rate>0.99"],
  },
};

const BASE_URL = __ENV.BASE_URL || "http://127.0.0.1:8000";
const API_KEY = __ENV.API_KEY || "dev-key";

export default function () {
  const response = http.post(
    `${BASE_URL}/v1/chat/completions`,
    JSON.stringify({
      model: "auto",
      messages: [{ role: "user", content: "Classify this burst probe ticket." }],
      max_tokens: 64,
      routing: { task: "classification", privacy: "private" },
    }),
    { headers: { "Content-Type": "application/json", Authorization: `Bearer ${API_KEY}` } },
  );

  check(response, {
    "no unhandled failure": (r) => [200, 429, 503].includes(r.status),
    "rejection is explicit": (r) => r.status === 200 || r.json("error.type") !== undefined,
  });
}
