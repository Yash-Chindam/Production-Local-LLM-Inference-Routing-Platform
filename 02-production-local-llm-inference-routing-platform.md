# Production Local-LLM Inference and Intelligent Routing Platform

**Document type:** Technical design specification  
**Purpose:** Define production local-model serving, routing and optimization  
**Implementation plan:** Intentionally excluded

> Proposed quality and performance values are design targets. Use them on a resume only after measuring a working implementation.

## 1. Use case

Many AI applications route every request to one expensive model. Extraction, classification and constrained generation may be handled by smaller local models, while difficult reasoning requires a larger model. This platform exposes one stable OpenAI-compatible API, serves multiple local models and LoRA adapters, and routes each request based on capability, predicted quality, latency, capacity, privacy and resource cost.

The project is inference infrastructure, not an end-user chatbot.

### Representative scenarios

1. Route extraction and classification to a quantized small model.
2. Route standard RAG requests to a general local model.
3. Route difficult reasoning or critique to a high-capability model.
4. Serve department-specific LoRA adapters over a shared base model.
5. Keep regulated data inside a private inference environment.
6. Compare batching, caching, quantization and speculative decoding under reproducible load.

## 2. Portfolio value

The project demonstrates:

- vLLM production inference.
- Ray Serve distributed deployment.
- LoRA/QLoRA model specialization.
- Quantization and GPU-memory optimization.
- Continuous batching and caching.
- Quality-, latency- and capacity-aware routing.
- Load testing and inference benchmarking.
- Autoscaling, backpressure and canary releases.
- Operational LLM observability.

## 3. Users and actors

| Actor | Responsibility |
|---|---|
| Application developer | Uses one stable API while internal models evolve. |
| ML engineer | Registers models, quantized variants and adapters. |
| Platform operator | Configures GPU pools, replicas, quotas and SLOs. |
| Evaluator | Maintains benchmark sets and approves promotions. |
| Router | Selects the model, adapter, priority and fallback path. |

## 4. Scope

### In scope

- OpenAI-compatible inference API.
- Multiple local language models.
- LoRA and QLoRA adapters.
- vLLM inference engines.
- Ray Serve deployment and autoscaling.
- Quantization experiments.
- Continuous batching.
- Prefix and semantic caching.
- SLA- and quality-aware routing.
- Canary deployment and rollback.
- GPU and request-level observability.
- Reproducible quality and load benchmarks.

### Out of scope

- Training foundation models from scratch.
- Serving arbitrary user-uploaded weights.
- Assuming speculative decoding always helps.
- Hiding quality regressions behind latency improvements.
- Assuming local inference is cheaper without accounting for utilization.

## 5. Architecture

```text
Client applications
        |
        v
LiteLLM / FastAPI compatibility gateway
        |
        v
Identity, quotas and request normalization
        |
        v
Routing policy service
  |            |              |
  |            |              +---- approved external fallback
  |            +------------------- general local model
  +-------------------------------- small specialist model
        |
        v
Ray Serve control plane
        |
        v
vLLM model and LoRA deployments
        |
        +---- Redis cache and quota state
        +---- MLflow model registry
        +---- Object storage for artifacts
        +---- Prometheus / Grafana / OpenTelemetry
```

## 6. Technology selection

| Technology | Responsibility | Selection rationale |
|---|---|---|
| vLLM | Inference engine | High-throughput serving, continuous batching, KV-cache management and production metrics. |
| Ray Serve | Distributed serving | Adds replicas, placement, autoscaling, health and multi-node coordination. |
| LiteLLM | Compatibility gateway | Normalizes local and external endpoints and centralizes quotas and fallbacks. |
| PEFT | LoRA lifecycle | Standard adapter creation and loading interfaces. |
| Unsloth | Efficient fine-tuning | Supports resource-efficient LoRA/QLoRA experiments. |
| MLflow | Model governance | Tracks artifacts, benchmarks, stages and promotion history. |
| Redis | Cache and quota state | Supports cache, counters and selected coordination data. |
| Kubernetes | Workload scheduling | Provides GPU scheduling, isolation and deployment control. |
| KEDA/Ray autoscaling | Capacity control | Scales validated components from queue and workload metrics. |
| Locust or k6 | Load generation | Creates reproducible sustained and burst workloads. |

## 7. Component design

### 7.1 Compatibility gateway

Responsibilities:

- Authenticate callers.
- Resolve tenant quotas and permitted model classes.
- Normalize model requests.
- Enforce context and output limits.
- Attach trace and routing metadata.
- Stream compatible responses.
- Return explicit overload and retry information.

### 7.2 Router

Routing features may include:

- Task class.
- Prompt and output length.
- Required modality.
- Structured-output requirement.
- Context size.
- Requested latency tier.
- Privacy classification.
- Current queue delay.
- GPU capacity.
- Historical quality by task and model.

Use deterministic rules for privacy and hard capability restrictions. Use a lightweight classifier or calibrated model for task and complexity prediction. Record the reason for every route.

### 7.3 Ray Serve control plane

Responsibilities:

- Deploy ingress and model replicas.
- Place workloads on appropriate GPU resources.
- Apply horizontal autoscaling.
- Limit queued and ongoing requests.
- Expose health and deployment state.
- Coordinate multi-node models only when hardware and workload require it.

### 7.4 vLLM engines

Responsibilities:

- Execute generation.
- Provide continuous batching.
- Manage KV-cache memory.
- Export inference metrics.
- Support tensor parallelism when required.
- Serve approved LoRA adapters where compatible.

### 7.5 Model lifecycle

- Hugging Face references immutable base-model revisions.
- PEFT/Unsloth creates LoRA or QLoRA adapters.
- MLflow stores models, adapters, benchmark evidence and lifecycle state.
- Every model card records license, revision, intended tasks, limitations and hardware requirements.

## 8. Model tiers

| Tier | Workload | Serving form |
|---|---|---|
| Small specialist | Extraction, classification and constrained generation | Quantized 3B-8B model or domain LoRA |
| General local | Summarization, standard RAG and moderate reasoning | 8B-14B instruction model |
| High capability | Complex reasoning, critique and difficult fallback | Larger multi-GPU model where practical |
| Approved external fallback | Unsupported capability or temporary saturation | Provider API behind explicit policy |

Select actual models using license, hardware fit and measured benchmark performance.

## 9. Optimization techniques

### Continuous batching

Combine compatible requests in the serving engine without requiring callers to batch work.

### Quantization

Evaluate AWQ, GPTQ or another supported format as independent variants. Measure memory, throughput, latency and quality.

### LoRA and QLoRA

Use adapters for domain specialization without duplicating the complete base model. Prefer Multi-LoRA serving when the engine and benchmark support it.

### Prefix caching

Use for repeated instructions and shared context. Track cache-hit rate and actual latency improvement.

### Semantic caching

Use only for deterministic, non-sensitive task classes. Include tenant/privacy class, model revision and generation parameters in cache eligibility.

### Speculative decoding

Treat as an experiment. Compare total latency and throughput over realistic prompts because low draft-token acceptance can add overhead.

### Backpressure

- Bound queues.
- Reject excess work predictably.
- Return retry guidance.
- Protect control-plane and health traffic from saturation.
- Avoid unlimited buffering and unbounded tail latency.

## 10. Request flow

```text
Authenticate caller
    -> resolve quota, privacy and latency class
    -> normalize request
    -> check eligible caches
    -> derive routing features
    -> score candidate models and adapters
    -> dispatch through Ray Serve to vLLM
    -> stream response
    -> apply declared fallback if required
    -> record route, metrics and feedback
```

## 11. Information model

### ModelProfile

- Immutable model revision.
- License and tokenizer.
- Context limit.
- Hardware requirement.
- Quantization form.
- Supported task classes.
- Quality and safety evaluation references.

### AdapterProfile

- Adapter and base-model revisions.
- Domain and intended task.
- Training dataset version.
- Benchmark improvement and regression evidence.

### RoutePolicy

- Eligible models and adapters.
- Privacy restrictions.
- Task and complexity rules.
- Latency objective and quality floor.
- Resource ceiling and fallback order.

### BenchmarkRun

- Dataset and workload version.
- Hardware and driver information.
- Container, engine and model revisions.
- Concurrency and prompt-length distribution.
- Engine settings, quality and latency results.

### DeploymentRevision

- Container digest.
- Model and adapter checksums.
- Ray/vLLM configuration.
- Kubernetes and GPU-pool settings.
- Promotion and rollback state.

## 12. Cache design

| Cache | Use | Restrictions |
|---|---|---|
| Prefix cache | Repeated system and context prefixes | Must match model and tokenized prefix |
| Exact response cache | Deterministic identical requests | Include tenant and parameters |
| Semantic cache | Approved non-sensitive repeatable tasks | Requires similarity and privacy policy |
| Router cache | Stable classification decisions | Invalidate with policy/model changes |

## 13. Reliability design

- Scale stateless ingress separately from GPU replicas.
- Keep warm replicas for latency-sensitive routes.
- Document optional-model cold-start time.
- Use readiness checks with model-load and inference probes.
- Stop routing to unhealthy replicas.
- Define behavior for GPU out-of-memory and node loss.
- Configure queue limits and graceful shutdown.
- Canary models, adapters and router policies separately.
- Preserve the previous revision for rollback.

## 14. Security design

- Authenticate users and services with short-lived credentials.
- Restrict model classes and quotas per tenant.
- Validate generation parameters.
- Never accept arbitrary model paths from a request.
- Scan container and model artifacts before release.
- Keep download tokens and provider keys in a secret manager.
- Redact sensitive prompts from traces and benchmark data.
- Enforce network policy between gateway, serving, storage and observability.

## 15. Observability

### Inference metrics

- Time to first token.
- Time per output token.
- P50/P95/P99 end-to-end latency.
- Requests and tokens per second.
- Current and average batch size.
- Queued and ongoing requests.
- GPU utilization and memory.
- KV-cache occupancy.
- Model-load and cold-start duration.

### Routing metrics

- Requests per route.
- Fallback frequency.
- Queue-delay prediction error.
- Predicted versus observed quality.
- Cache-hit rate.
- Quota and overload rejection.

## 16. Evaluation

- Maintain task-specific quality benchmarks.
- Test structured-output validity.
- Compare quantized and full-precision variants.
- Compare adapters with the base model.
- Use representative concurrency and prompt-length distributions.
- Run sustained and burst loads with Locust or k6.
- Measure GPU-seconds and estimated cost per successful request.
- Report quality and latency together.

### Proposed design targets

- No silent routing or fallback changes.
- Complete model, adapter and configuration attribution per response.
- Bounded queue behavior during overload.
- Reproducible load tests from committed workload definitions.
- Documented quality delta for every optimization variant.
- Automatic rollback after failed readiness or canary criteria.

## 17. Deployment topology

- LiteLLM/FastAPI gateway.
- Router service.
- Ray head and worker topology.
- GPU pools separated by accelerator type.
- vLLM deployments.
- Redis.
- MLflow and artifact storage.
- NVIDIA GPU Operator or equivalent support.
- Prometheus, Grafana and GPU exporter.
- Docker, Kubernetes, Helm and GitHub Actions.

## 18. Official references

- [Ray Serve LLM configuration](https://docs.ray.io/en/latest/serve/llm/user-guides/configuration.html)
- [Ray Serve LLM architecture](https://docs.ray.io/en/latest/serve/llm/architecture/overview.html)
- [vLLM documentation](https://docs.vllm.ai/)
- [MLflow documentation](https://mlflow.org/docs/latest/)
- [PEFT documentation](https://huggingface.co/docs/peft/)

