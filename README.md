# Production Local-LLM Inference & Routing Platform

An OpenAI-compatible control plane for routing requests across local model tiers. The
current inference backend is deterministic for CI and is replaceable by Ray Serve and
vLLM deployments.

The complete architecture and design targets are documented in
[`02-production-local-llm-inference-routing-platform.md`](02-production-local-llm-inference-routing-platform.md).

## Development

Requires Python 3.11 or newer and Node.js 20 or newer.

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
npm ci
python -m uvicorn llm_router.app:app --app-dir src --reload
```

The development bearer token is `dev-key`. Override it in every shared environment.
Production startup rejects that development key.

```bash
ROUTER_API_KEYS="replace-me" python -m uvicorn llm_router.app:app --app-dir src
```

Example request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer dev-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Extract invoice fields"}],"routing":{"privacy":"restricted"}}'
```

Every response records the selected model, immutable revision, inferred task, candidate
count, policy score, and route reason.

## Verification

```bash
ruff format --check .
ruff check .
mypy
pytest tests/unit tests/integration --cov=llm_router --cov-report=term-missing
npx playwright install chromium
npm run test:e2e
docker build -t local-llm-router:dev .
```

CI reports unit/static analysis, integration, and Playwright end-to-end tests separately.
The release-image build starts only after all three test layers pass.

After a successful CI run on `main`, CD creates a versioned OCI image artifact. Registry
or cluster publication remains disabled until an explicit deployment destination is
configured. Dependabot maintains Python, npm, and GitHub Actions dependencies, while the
PR labeler classifies API, test, CI/CD, documentation, and dependency changes.

Successful PR CI runs are merged automatically only for trusted same-repository authors
and Dependabot. Forks, drafts, and untrusted author associations are deliberately skipped;
repository branch-protection and review requirements continue to apply.

### Releases

Every push to `main` that passes CI is scanned for
[Conventional Commits](https://www.conventionalcommits.org/) since the last tag. A `feat:`
commit earns a minor release, `fix:` a patch release, and a declared breaking change
(`type!:` or a `BREAKING CHANGE` footer) always wins with a major release. A range with none
of those — only `docs:`, `ci:`, `build:`, `test:`, or `chore:` commits, as with most
Dependabot bumps — is deliberately left unreleased. When a release is warranted, the
workflow tags `main` (`vMAJOR.MINOR.PATCH`) and publishes a GitHub Release with
auto-generated notes. Tags are never created by hand, and nothing is ever tagged off a
branch other than `main`.

## Routing

Privacy, tenant entitlement, context size, and capability are hard filters: deterministic, and
applied before any score is computed. Among the models that remain, the router scores measured
quality for the task against observed queue delay, cost, engine saturation, and how complex the
request is predicted to be.

**Task and complexity** come from a calibrated classifier — multinomial naive Bayes over word
unigrams and bigrams, trained at start-up from
[`config/routing/task-classifier-v1.jsonl`](config/routing/task-classifier-v1.jsonl). It needs no
accelerator and no extra dependency, so a routing decision never waits on the models it is
choosing between. Its posteriors are temperature-scaled against a held-out split, so a confidence
reads as a probability, and a prediction under 0.5 abstains to the `general` task rather than
being trusted. A task the caller declares in `routing.task` is never overridden. Without the
dataset the router falls back to keyword rules.

On the held-out prompts in
[`benchmarks/datasets/routing-tasks-v1.jsonl`](benchmarks/datasets/routing-tasks-v1.jsonl) the
classifier gets every task right, 88% of complexity labels, and an expected calibration error of
0.035. Both datasets are small and were written by hand in one voice, so treat that as a floor
check on the mechanism, not as evidence of accuracy on real traffic: replace them with labelled
production prompts before relying on the numbers.

| Routing feature | Source |
|---|---|
| Task and complexity | The classifier; `low` leaves work on the cheapest capable model, `high` outweighs the specialization and cost terms. |
| Structured-output requirement | `routing.structured` excludes any model whose card sets `supports_structured_output: false`. |
| Quality by task and model | Mean benchmarked quality for the task from the catalog; the card's headline `quality` only for a task never measured. |
| Current queue delay | An exponentially weighted average of what requests for that model actually waited, replacing the catalog estimate after the first observation. |
| GPU capacity | Engine saturation — the worse of KV-cache occupancy and the share of admitted work not yet started — from the last metrics scrape, discarded after 30 seconds. |

One gateway faces one engine, so saturation costs every local model equally: it can tip an
eligible request to the approved external model, and it never reorders local models or overrides
privacy. Required modality is not a routing feature yet; message content is text only.

Every response reports `task_source` (`declared`, `classifier`, `abstained`, `keyword`, or
`cached`), `task_confidence`, and `complexity` beside the route reason.

## Serving backends

`ROUTER_BACKEND=mock` (the default) keeps CI deterministic and GPU-free.
`ROUTER_BACKEND=vllm` dispatches to a vLLM OpenAI-compatible server; a selected LoRA
adapter is served by name over the shared base model. An unreachable or failing engine
returns `502` with retry guidance, and `/readyz` fails while the engine is unhealthy.

Set `"stream": true` to receive OpenAI-compatible `text/event-stream` chunks. Streamed
results are cached under the same eligibility rules and replayed as chunks on a hit.

### External providers

The gateway never talks to a provider. An approved external model is reached through a
[LiteLLM](https://docs.litellm.ai/) proxy, configured in
[`config/litellm.yaml`](config/litellm.yaml), so provider credentials stay out of the gateway and
adding a provider does not change it. Every alias in that file must match a non-local model card
in the catalog; a test fails if they drift. The provider model named there is a placeholder: pick
the one your policy approves.

A request reaches the proxy only when all of these hold:

1. The operator has set `ROUTER_EXTERNAL_FALLBACK_ENABLED=true`. With `ROUTER_BACKEND=vllm` the
   gateway refuses to start unless `ROUTER_EXTERNAL_BASE_URL` is also set.
2. The effective privacy class is `public`, after any tenant floor has been applied.
3. The request sets `routing.allow_external_fallback`, and the tenant does not forbid it.

The rule is enforced twice. Routing never selects an external model for private or restricted
data, and the dispatch boundary refuses to send it even if routing were wrong, answering `500`
`policy_violation` and counting `router_rejections_total{reason="external_dispatch_refused"}`.

An eligible request also falls back when the local engine fails with it: unreachable, out of
memory, or circuit open. The response names the model that answered, its route reason says
which model it fell back from, and `router_fallbacks_total` counts it by cause. A fallback
response is cached under the model that produced it. The local engine and the proxy have separate
circuits, so a failing engine does not close the path to the provider. Streamed requests do not
fall back; they fail as described under [Failure behaviour](#failure-behaviour).

In the cluster the proxy is the only workload allowed to reach the internet, and only the gateway
may call it. This path has been tested against a stand-in transport, not against a running
LiteLLM proxy or a real provider.

### Ray Serve deployment

[`config/ray-serve.yaml`](config/ray-serve.yaml) is generated from the catalog, never
hand-edited, and verified by a test:

```bash
python -m llm_router.serving > config/ray-serve.yaml
```

It carries per-tier autoscaling (latency-sensitive tiers keep a warm replica), GPU pool
placement, tensor parallelism, prefix caching, quantization, and Multi-LoRA settings.

### Optimization variants

Quantization formats and speculative decoding are declared in the catalog as variants of one
immutable base revision, and each is an experiment until it is measured:

- In `development` a variant is rendered into `config/ray-serve.yaml` under `experiments`, as its
  own application beside the base model with no warm replica, so it can be benchmarked on the
  same hardware without taking traffic.
- It cannot reach `staging` or `production` without a baseline and a variant benchmark, and the
  catalog refuses to load if the two runs differ in dataset, workload, hardware, or concurrency:
  a delta only means something when nothing else changed.
- Quality is never traded for speed. A loss beyond the variant's tolerance blocks promotion
  however large the gain. Speculative decoding must also actually be faster, since low
  draft-token acceptance adds overhead; a quantized variant must show the GPU memory it was
  meant to save.
- Once in `production` its settings are written into the base model's engine arguments, and the
  cache identity changes so no response from before the promotion is reused.

`GET /v1/registry/variants` returns every variant with its quality, latency, throughput, memory,
and GPU-second deltas and each regression, or `null` where nothing has been measured.

Produce the evidence by running a committed dataset against the base model and against the
experiment, then committing both records under `benchmarks:`:

```bash
python -m llm_router.evaluation --base-url http://127.0.0.1:8001   --model general-local--general-gptq --model-revision mock-general@sha256:dev   --dataset benchmarks/datasets/extraction-v1.jsonl --dataset-version extraction-v1   --benchmark-id general-gptq-extraction --hardware nvidia-a10g --task extraction
```

It prints a catalog benchmark record, including GPU memory and draft-token acceptance when the
engine publishes them. Requests run one at a time, so use the k6 workloads for behaviour under
load. **Neither committed variant has been measured:** both sit in `development` with no
evidence, and no claim is made about what either gains.

## Deployment topology

[`deploy/kubernetes`](deploy/kubernetes) holds the namespaced manifests: gateway
Deployment and Service, GPU serving pool, Redis, the LiteLLM proxy, the MLflow server, KEDA
autoscaling on queue depth and p95 latency, a Prometheus `ServiceMonitor`, network policy, and
credentials sourced from the cluster secret manager. No secret material is committed. Unit tests enforce the contract:
unprivileged workloads, digest-pinned images, bounded resources, real probes, GPU pool
pinning, and `/metrics` reachable only from monitoring.

```bash
kubectl apply -k deploy/kubernetes       # one vLLM engine
kubectl apply -k deploy/overlays/ray     # Ray Serve across GPU pools
```

The same manifests install as a Helm chart, [`deploy/helm/llm-routing`](deploy/helm/llm-routing):

```bash
helm upgrade --install llm-routing deploy/helm/llm-routing \
  --namespace llm-routing --create-namespace \
  --set serving.mode=ray
```

| Value | Default | Purpose |
|---|---|---|
| `serving.mode` | `vllm` | `vllm` for one engine, `ray` for the Ray Serve topology below. |
| `images.*` | digest placeholders | One digest-pinned reference per workload. |
| `gateway.replicas` | `2` | Starting gateway size. |
| `gateway.autoscaling.minReplicas` / `maxReplicas` | `2` / `20` | KEDA bounds. |

The chart is generated from the manifests and never edited by hand; tests check that it renders
exactly what kustomize renders, in both modes.

```bash
python -m llm_router.chart            # rebuild after changing a manifest
python -m llm_router.chart --check    # exit 1 if the committed chart is stale
```

The base runs a single vLLM engine, which serves one model. The
[`deploy/overlays/ray`](deploy/overlays/ray) overlay replaces it with a KubeRay `RayService`
that serves every local model in the catalog behind one OpenAI-compatible endpoint:

- A head that schedules and never runs a model.
- One worker group per accelerator type, pinned by `nvidia.com/gpu.product`, so GPU pools stay
  separate. Each group is sized from the autoscaling bounds of the models placed on it, and a
  worker holds as many GPUs as the largest replica on its pool needs.
- Model weights are loaded from the location [governance](#governance-in-mlflow) records for
  each revision.

The `RayService` is generated from the catalog, never hand-edited, and verified by a test and
by CD:

```bash
python -m llm_router.topology > deploy/overlays/ray/ray-service.yaml
```

Under the overlay, engine metrics are scraped from the Ray pods by Prometheus. The gateway
cannot read a whole cluster from one address, so its own `router_engine_*` and `router_gpu_*`
gauges stay empty there and live load does not influence routing.

GPU support comes from the NVIDIA GPU Operator, installed cluster-wide with
[`deploy/gpu-operator/values.yaml`](deploy/gpu-operator/values.yaml). It provides the
`nvidia.com/gpu` resource, the node label the pools select on, and the DCGM GPU exporter.

Two Grafana dashboards in [`deploy/kubernetes/dashboards`](deploy/kubernetes/dashboards) ship as
a labelled ConfigMap for Grafana's sidecar: one for the gateway and router, one for engines and
GPUs. A `PrometheusRule` alerts on latency, load shedding, a stuck queue, fallback rate, canary
rollback, an open engine circuit, KV-cache pressure, and GPU memory. A test fails if a dashboard
or alert queries a metric the gateway does not publish.

None of this has been applied to a cluster. The manifests are schema-validated and
contract-tested; the Ray image, GPU product labels, and node sizes are placeholders to set for
the hardware you have.

Stateless ingress scales separately from GPU replicas. Set `ROUTER_REDIS_URL` so cache and
quota state are shared once the gateway runs more than one replica; without it both are
in-process and correct for a single replica only. Install the client with the extra:

```bash
python -m pip install -e ".[redis]"
```

CD renders the canary plans (one per track, each with its rollback target) and the governance
plan, verifies
`config/ray-serve.yaml` and the Ray topology against the catalog, and validates both rendered
topologies with kubeconform. It also checks the Helm chart against the manifests, lints it, and
packages it.
Applying to a cluster stays disabled until a deployment destination is configured.

## Model registry

[`config/registry.yaml`](config/registry.yaml) is the governed source of truth for what may
be served. A request can never introduce a model path, revision, or adapter.

- Model cards record license, tokenizer, revision, context limit, quantization, hardware
  requirement, intended tasks, limitations, and evaluation evidence. Promotion to
  `production` is rejected without evaluation references.
- Adapters bind to one immutable base revision, declare their dataset version and measured
  quality delta, and cannot be promoted with unresolved regressions.
- Deployment revisions record container digest, model and adapter checksums, Ray and vLLM
  configuration, GPU pool, and the previous revision used for rollback.

| Endpoint | Purpose |
|---|---|
| `GET /v1/models` | OpenAI-compatible catalog enriched with tier, stage, license, and quantization. |
| `GET /v1/registry/models/{id}` | Full model card with its benchmark evidence. |
| `GET /v1/registry/adapters` | Promoted LoRA and QLoRA adapters. |
| `GET /v1/registry/deployments` | Deployment revisions and rollback targets. |
| `GET /v1/registry/variants` | Optimization variants with their measured deltas. |
| `GET /v1/registry/canaries` | Canary plans by track, with live adapter state. |

Send `routing.domain` to request a domain adapter; the router applies the promoted adapter
with the largest measured quality gain for that base revision and task, or none at all.

### Adapter recipes

Every adapter in the catalog has a recipe in [`config/adapters`](config/adapters): the base
model commit it was trained from, the dataset version, and the LoRA hyperparameters. A recipe
is what makes an adapter reproducible, so a catalog adapter without one fails the check.

```bash
python -m pip install -e ".[training]"     # PEFT; add ".[unsloth]" for Unsloth
python -m llm_router.adapters check                                  # recipes agree with the catalog
python -m llm_router.adapters plan claims-extraction-lora            # the arguments a run would get
python -m llm_router.adapters train claims-extraction-lora --output ./out/claims
python -m llm_router.adapters register claims-extraction-lora --output ./out/claims
```

- `method: lora` trains over the full-precision base; `method: qlora` loads the base in 4-bit.
  `framework` selects PEFT or Unsloth, and both are given the same rank, alpha, target modules,
  commit, and seed.
- The base is pinned to a 40-character commit. A branch or tag is rejected, because it can move.
- A rank above 32 is rejected: the serving configuration would not load it.
- `register` runs the [artifact scan](#artifact-scanning) on the finished adapter and prints its
  catalog entry. The entry starts in `development` with no measured gain, and its revision is the
  artifact's digest. Promotion is a separate, reviewed change once a benchmark exists.

Training has not been run: it needs a GPU. The recipes' `hf_repo` and `hf_revision` are
placeholders, as the catalog's base models are mocks.

### Governance in MLflow

The catalog decides what is served; [MLflow](https://mlflow.org/docs/latest/) keeps the record.
`llm_router.governance` syncs the catalog into an MLflow model registry and tracking store:

| Catalog record | In MLflow |
|---|---|
| Model or adapter revision | One model version, tagged with its card (license, tier, hardware, limitations, base revision, dataset version), artifact location, and checksum. |
| Lifecycle stage | A version tag, plus a `staging` or `production` alias, so `models:/general-local@production` resolves. |
| Benchmark run | One run with its measurements as metrics and its dataset, workload, hardware, driver, and engine revision as parameters. |
| Stage change | One appended run in the promotions experiment: from, to, commit, and policy version. |

```bash
python -m pip install -e ".[governance]"
python -m llm_router.governance plan                      # what a first sync would record
python -m llm_router.governance sync --tracking-uri "$MLFLOW_TRACKING_URI"
python -m llm_router.governance verify --tracking-uri "$MLFLOW_TRACKING_URI"   # exit 3 on drift
python -m llm_router.governance history --name general-local --tracking-uri "$MLFLOW_TRACKING_URI"
```

- Governance flows one way. The gateway never reads MLflow, so a stage changed by hand there
  changes nothing in production; `verify` reports it and the next `sync` puts it back.
- A revision is immutable. If a recorded revision turns up with a different checksum, the sync
  is refused: a changed artifact needs a new revision.
- A revision the catalog replaces or removes is retired to `deprecated`, never deleted, so the
  rollback target stays on record.
- Models registered in MLflow by anyone else are left alone.

The sync records where an artifact belongs under `--artifact-root`; it does not upload weights.
CD renders the plan as an artifact. Running `sync` against a live MLflow is part of the deploy
step, which stays disabled until a destination is configured. The tests run against a real MLflow
on SQLite; the server deployment in [`deploy/kubernetes/mlflow.yaml`](deploy/kubernetes/mlflow.yaml)
has not been run on a cluster.

## Artifact scanning

Nothing is released without being scanned.

**Container.** CI scans the release image with [Trivy](https://trivy.dev/) and fails on any
critical or high vulnerability that has a fix, then starts the image and waits for `/healthz`.
The image takes the distribution's security updates at build time and ships without `pip`, so
the installer and the libraries it vendors are not there to be vulnerable.

**Models and adapters.** Loading weights can run code: a pickle checkpoint executes what it
names, and a repository that ships Python asks the loader to import it. `llm_router.artifact_scan`
inspects an artifact directory without loading anything from it.

```bash
python -m llm_router.artifact_scan ./artifacts/general-local --subject general-local
python -m llm_router.artifact_scan ./artifacts/claims-lora --kind adapter --expect sha256:...
```

| Rule | Refused |
|---|---|
| `pickle-format` | Any pickle-capable file (`.bin`, `.pt`, `.ckpt`, `.pkl`, ...), or a pickle under any other name. The finding names what it would import. |
| `executable-code` | Python, shell, or native libraries inside the artifact. |
| `remote-code` | A configuration with `auto_map` or `trust_remote_code`, or one that cannot be read. |
| `invalid-safetensors` | A header that does not account for exactly the bytes in the file. |
| `no-weights` | An artifact with no safetensors weights. |
| `symlink` | A link pointing outside the artifact. |
| `adapter-config` | An adapter that is not LoRA, or whose rank the engine would not load. |
| `checksum-mismatch` | A digest other than the one the catalog records. |

The digest covers every file's path and content, so it is the value to record as a checksum in
a deployment revision. The command exits 1 on any finding. It checks formats and provenance; it
does not judge what a model has learned. The catalog's checksums are placeholders, so no real
artifact has been scanned against them yet.

## Failure behaviour

| Condition | What the gateway does |
|---|---|
| Capacity saturated | `503` `overloaded` after the bounded admission wait, with `Retry-After: 1`. A streamed response holds its slot until the stream ends, so streaming cannot escape the concurrency bound. |
| GPU out of memory | `503` `engine_out_of_memory`, `Retry-After: 10`. The same request at the same size will fail again, so the message says to shorten the prompt or lower `max_tokens`. Counts toward the circuit. |
| Node lost or engine unreachable | `502` `backend_unavailable` for each of the first `ROUTER_ENGINE_FAILURE_THRESHOLD` consecutive failures; then the circuit opens. |
| Circuit open | `503` `engine_unavailable` immediately, without contacting the engine, with `Retry-After` set to the remaining cooldown. `/readyz` fails, so the orchestrator stops routing here. |
| Engine recovers | A healthy probe ends the cooldown early and lets exactly one trial request through. Only that request succeeding closes the circuit; if it fails, the cooldown restarts. |
| Shutdown | `/readyz` fails first, then admitted requests get `ROUTER_SHUTDOWN_GRACE_SECONDS` to finish before the engine client closes. Keep it under the pod's `terminationGracePeriodSeconds` (60). |

An engine error body is inspected for an out-of-memory report and then discarded, never forwarded:
it can echo the prompt it rejected. `router_engine_circuit_open` reports 0 closed, 0.5 half-open,
1 open.

A request that fails is not retried on another local model. Every local model shares the one
engine a gateway faces, so a retry would meet the same failure; the caller is told when to come
back. The one exception is a request already entitled to the
[external provider](#external-providers), which falls back to it.

Cold start is measured, not assumed (see `router_model_load_seconds` below). Tiers that keep a
warm replica never pay it on the request path. The high-capability tier scales to zero, so its
first request after an idle period waits for a full model load: budget for it, or raise its
`min_replicas`.

## Canaries and rollback

Models, adapters, and router policies are canaried on separate tracks, each with its own plan
naming exactly what it rolls back to:

```bash
python -m llm_router.canary
```

| Track | Canary | Rolls back to |
|---|---|---|
| `model` | A deployment revision | The revision in `previous_revision_id`. |
| `adapter` | A `staging` adapter | The `production` adapter for the same base revision and domain, or the base model alone. |
| `policy` | The policy `version` | `previous_version`. |

One rule decides every track. Failed readiness rolls back at once. Once 50 requests have been
observed, the canary rolls back if its error rate exceeds 1%, its p95 latency exceeds the
strictest tier objective among the models it serves, or its observed quality falls more than
0.05 below their benchmarked quality. Error rate and latency only count against the canary
when the stable baseline does not share the problem, so an engine outage that degrades both is
not blamed on the change. A canary that stays clean for 500 requests is reported ready.

**The adapter track runs inside the gateway.** A staged adapter is offered
`canary_traffic_percent` of eligible requests, bucketed by tenant and prompt so a retry cannot
flip between adapters. It is suspended automatically the moment it fails, and the production
adapter serves everything again. A canary's responses are never cached, so nothing it produced
outlives a rollback. Before this, a staged adapter with the larger measured gain took all of the
traffic.

`GET /v1/registry/canaries` lists every plan with live state for adapters (`in-progress`,
`ready-to-promote`, or `rolled-back`, with the reasons). `router_canary_requests_total` and
`router_canary_rollbacks_total` report it to Prometheus. State is per gateway replica: each
reaches the same verdict from its own share of traffic.

**Model and policy canaries are rollouts** of the serving pool or the gateway, so whatever
controls that rollout evaluates the plan through the same rule:

```bash
python -m llm_router.canary --plan model:deploy-0002 --observation observed.json --baseline stable.json
```

It exits `0` to promote, `2` to hold, and `3` to roll back, and prints the rollback target. No
rollout controller is wired to it yet, so on those two tracks the decision is automatic and the
action is not.

Promotion is never automatic on any track. It is a catalog change and goes through review.

## Tenants

What a caller may use is governance and lives in the catalog; the credential that proves which
tenant is calling stays in the environment and is never committed.

```bash
ROUTER_TENANT_KEYS="support-tooling:<key>,clinical-research:<key>"
```

Keys listed bare in `ROUTER_API_KEYS` belong to the `default` tenant, so an existing deployment
keeps working unchanged. Each tenant in [`config/registry.yaml`](config/registry.yaml) may set:

| Field | Effect |
|---|---|
| `permitted_tiers` / `permitted_models` | Hard filter applied before scoring; it can never be outscored. Empty means unrestricted. |
| `quota_requests_per_minute` | The tenant's own limit; absent defers to the platform default. |
| `minimum_privacy` | A floor, never a ceiling. The request is raised to it before anything reads the class. |
| `allow_external_fallback` | `false` denies external routing outright; absent defers to platform policy. |
| `quality_floor` | Raises, and never lowers, the floor a request asked for. |

An absent field never tightens an existing deployment: it defers to platform policy rather than
implying a restriction.

Quota and cache are scoped to the tenant rather than the credential, so rotating a key neither
resets a quota nor orphans a cache, and two keys for one tenant draw on one quota.

The privacy floor is applied before the cache lookup, not at routing. A tenant handling regulated
data cannot declare its traffic `public` and so become eligible for external routing or semantic
reuse, and it can never read an entry another tenant stored under `public`. When the class is
raised the route reason says so — the change is attributed, not silent.

A request that no entitled model can serve is refused with `422` rather than downgraded to a
model never validated for the task.

## Observability

`GET /metrics` returns Prometheus exposition text and is intentionally unauthenticated so
in-cluster scrapers can read it; restrict it with network policy rather than a bearer token.

| Metric | Purpose |
|---|---|
| `router_request_latency_seconds` | End-to-end latency histogram per model. |
| `router_time_to_first_token_seconds` | Admission-to-first-token delay. |
| `router_time_per_output_token_seconds` | Mean generation time per output token. |
| `router_tokens_total` | Prompt and completion tokens per model. |
| `router_inflight_requests` / `router_queued_requests` | Live capacity and queue depth. |
| `router_routes_total` | Requests per route with task and privacy class. |
| `router_external_fallback_total` | Requests answered by an external model. |
| `router_fallbacks_total` | Fallbacks after a local engine failure, by cause and by the models fallen back from and to. |
| `router_queue_delay_prediction_error_ms` | Predicted versus observed queue delay. |
| `router_rejections_total` | Quota, overload, and policy rejections. |
| `router_cache_events_total` | Cache lookups by cache and result. |
| `router_model_load_seconds` | Measured engine load and cold-start duration. |
| `router_observed_quality` / `router_quality_prediction_error` | Observed quality where it is checkable, against what routing predicted. |
| `router_structured_output_total` | Structured-output responses by validity. |

Engine and accelerator state is pulled through from the serving path on each scrape, so the
gateway stays the single scrape target and no poller runs when nobody is collecting. The
gateway reads the engine's own `/metrics` (vLLM's `vllm:*` series, plus `DCGM_FI_DEV_*` from a
GPU exporter beside it) and republishes:

| Metric | Purpose |
|---|---|
| `router_engine_running_requests` | Requests the engine is decoding — its live batch size. |
| `router_engine_batch_size` | Batch size sampled per scrape; the average is `_sum / _count`. |
| `router_engine_waiting_requests` | Requests queued inside the engine, not yet batched. |
| `router_engine_kv_cache_occupancy_ratio` | Fraction of the KV cache allocated. |
| `router_engine_preemptions_total` | Requests preempted under KV-cache pressure. |
| `router_gpu_utilization_ratio` | Accelerator utilization. |
| `router_gpu_memory_used_bytes` / `router_gpu_memory_total_bytes` | Framebuffer memory in use and installed. |

An unreachable engine costs a scrape its engine series, never an error: a missing sample stays
missing rather than being reported as zero.

Live traffic is ungraded, so the only quality signal observable in production is whether a
request that declared `routing.structured` actually returned parseable JSON. That is recorded
as an observed quality of 1 or 0 and compared with the quality routing predicted for the model
it picked. Full task-level quality comes from the offline harness below, never from sampled
traffic.

Cold start is measured rather than configured: `/readyz` is the one place that sees the engine
go from loading to serving, so the duration of that window is observed there. The window
reopens on every later recovery, so a reload after an out-of-memory eviction or a lost node is
measured too — not only the first start.

### Tracing

Every chat completion is one OpenTelemetry span carrying what is needed to explain the route:
tenant, effective and declared privacy class, task, model and revision, adapter, cache result,
score, candidate count, the route reason, and token usage. The response quotes it as
`X-Trace-Id`. Only the OpenTelemetry API is a runtime dependency, so tracing is a no-op until a
collector is configured:

```bash
python -m pip install -e ".[tracing]"
ROUTER_OTLP_ENDPOINT=http://otel-collector:4318/v1/traces
```

Prompts are redacted by privacy class, evaluated after any tenant floor has been applied:

| Class | Recorded |
|---|---|
| `restricted` | Length only. No digest: a digest of a short or templated prompt can be reversed by guessing. |
| `private` | Length and a SHA-256 digest, so repeats can be correlated. |
| `public` | Length and digest; a bounded prefix of the content only with `ROUTER_TRACE_PROMPT_CONTENT=true`. |

Completions are never recorded. A failed request records its error type and not its message,
because an engine error can echo the request it rejected.

## Caching

Cache keys bind the tenant, the active model-catalog fingerprint, and the generation
parameters, so promoting any model revision or routing policy version invalidates every
dependent entry. Responses carry `X-Cache: miss | exact | semantic`.

| Cache | Eligibility |
|---|---|
| Exact response | Deterministic requests (`temperature = 0`) that are not `restricted`. |
| Prefix | Reported per model revision for repeated instruction prefixes. |
| Semantic | Disabled by default; requires `public` privacy, deterministic generation, and an extraction, classification, or summarization task. |
| Router decision | Reuses stable task classification; cleared when the policy version changes. |

## Evaluation

Every optimization is graded against a committed dataset in
[`benchmarks/datasets`](benchmarks/datasets), never against sampled production traffic.
`execute()` runs each case through a caller-supplied transport and times it; `build_report()`
turns the outcomes into one `EvaluationReport` that always pairs quality with latency and
GPU cost, never one alone.

```bash
python -c "
from llm_router.evaluation import build_report, execute, load_dataset
cases = load_dataset('benchmarks/datasets/extraction-v1.jsonl')
outcomes = execute(cases, lambda case: (case.expected, True, 0.05))
print(build_report(outcomes, model_id='small-specialist', model_revision='rev-1').summary())
"
```

- Constrained tasks (extraction, classification) are scored by exact match, generative tasks
  by token overlap, and structured tasks score zero when the output is not valid JSON.
- `compare()` rejects a variant that buys latency or throughput with a quality or
  structured-validity regression, however small; `render_comparison()` prints the verdict
  with every regression reason.
- [`benchmarks/workloads`](benchmarks/workloads) holds reproducible k6 steady and burst load
  definitions. The burst scenario asserts bounded queue behavior — an explicit `429`/`503`
  rejection — rather than unbounded tail latency.

## Runtime settings

All settings use the `ROUTER_` prefix.

| Variable | Default | Purpose |
|---|---:|---|
| `ROUTER_API_KEYS` | `dev-key` | Comma-separated bearer tokens. |
| `ROUTER_MAX_CONCURRENCY` | `32` | Maximum in-flight requests. |
| `ROUTER_ADMISSION_TIMEOUT_SECONDS` | `0.25` | Time allowed to wait for capacity. |
| `ROUTER_QUOTA_REQUESTS_PER_MINUTE` | `120` | Per-token sliding-window quota. |
| `ROUTER_EXTERNAL_FALLBACK_ENABLED` | `false` | Operator gate for external fallback. |
| `ROUTER_EXTERNAL_BASE_URL` | _(empty)_ | LiteLLM proxy address; required with `vllm` when external fallback is enabled. |
| `ROUTER_EXTERNAL_API_KEY` | _(empty)_ | Key the gateway presents to the proxy. |
| `ROUTER_REDIS_URL` | _(empty)_ | Shared cache and quota state; in-process when empty. |
| `ROUTER_TENANT_KEYS` | _(empty)_ | `tenant:key` bindings; bare `ROUTER_API_KEYS` keys use the default tenant. |
| `ROUTER_OTLP_ENDPOINT` | _(empty)_ | OTLP/HTTP trace collector; tracing is a no-op when empty. |
| `ROUTER_TRACE_PROMPT_CONTENT` | `false` | Records a bounded prefix of `public` prompts only. |
| `ROUTER_BACKEND` | `mock` | `mock` or `vllm`. |
| `ROUTER_VLLM_BASE_URL` | `http://127.0.0.1:8001` | vLLM OpenAI-compatible endpoint. |
| `ROUTER_BACKEND_TIMEOUT_SECONDS` | `60` | Per-request engine timeout. |
| `ROUTER_ENGINE_FAILURE_THRESHOLD` | `5` | Consecutive engine failures before the circuit opens. |
| `ROUTER_ENGINE_COOLDOWN_SECONDS` | `30` | How long the circuit stays open before a trial request. |
| `ROUTER_SHUTDOWN_GRACE_SECONDS` | `20` | How long shutdown waits for admitted requests. |
| `ROUTER_REGISTRY_PATH` | `config/registry.yaml` | Governed model catalog; built-in profiles are used if absent. |
| `ROUTER_ROUTING_POLICY_VERSION` | `v1` | Invalidates router and response caches when changed. |
| `ROUTER_CACHE_ENABLED` | `true` | Master switch for all cache tiers. |
| `ROUTER_CACHE_TTL_SECONDS` | `300` | Exact-response entry lifetime. |
| `ROUTER_CACHE_MAX_ENTRIES` | `1024` | Bound on cached responses. |
| `ROUTER_SEMANTIC_CACHE_ENABLED` | `false` | Enables similarity reuse for approved tasks. |
| `ROUTER_SEMANTIC_SIMILARITY_THRESHOLD` | `0.92` | Minimum similarity for a semantic hit. |

External routing also requires public data and request-level opt-in. Private and restricted
requests are never eligible for an external route.
