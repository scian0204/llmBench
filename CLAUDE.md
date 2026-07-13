# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM Bench v2 is a vLLM throughput benchmarking tool (Node.js + Express + undici). It streams concurrent requests to an OpenAI-compatible LLM server and measures throughput with correct methodology: constant-concurrency worker pools, wall-clock system throughput, TTFT/decode separation, and percentiles.

## Commands

```bash
npm install    # Install dependencies (express, undici)
npm start      # Start server (port 3100, PORT env to override)
npm run dev    # Start with --watch
npm run mock   # Start mock vLLM target (port 8009) for testing
```

## Verification

Run `npm run mock` + `npm start`, then benchmark `http://localhost:8009/v1` (model name: anything). The mock saturates at concurrency 16 (40 tok/s per request below, sharp degradation above). Auto mode must find optimal concurrency = 16. Manual mode decode TPS must be ~38-40.

## Architecture

Three files: `server.js` (engine + API), `public/index.html` (frontend, single file), `mock-target.js` (test target).

### Measurement methodology (server.js)

- **Constant-concurrency worker pool** (`runLevel`): N workers each loop "finish request, immediately start next" until the request budget is spent. No round-robin batches, no convoy effect.
- **System throughput** = total output tokens in the level's wall-clock window / window seconds. Never the sum of per-request TPS.
- **Per-request metrics** (`runOne`): TTFT (first token time), latency, decode TPS = (tokens-1)/(last token - first token). Prefill and decode are never mixed.
- **Percentiles**: p50/p95/p99 for latency, TTFT, decode TPS (`distStats`).
- **Warmup requests** are flagged `warmup: true` and excluded from all statistics and SSE stream cards.
- **Auto mode** (`runAuto`): warmup, then geometric probe 1,2,4,...,maxConcurrency; stop when throughput < 90% of peak; one midpoint-refinement round around the peak.
- **Stop** aborts in-flight requests immediately via a per-run `AbortController`; aborted requests get status `aborted` and are excluded from stats.
- Requests are raw undici streams to `{baseUrl}/chat/completions` with `stream_options: {include_usage: true}`; token count falls back to chunk count when usage is absent.

### API

- `POST /api/bench/start` — body: `baseUrl, model, prompt, mode(manual|auto), apiKey?, maxTokens, temperature, varyPrompt, includeCodebase, warmupRequests` + manual: `concurrency, totalRequests` / auto: `maxConcurrency, requestsPerLevel`. 409 if running.
- `POST /api/bench/stop` — abort immediately
- `POST /api/bench/reset` — clear state
- `GET /api/bench/state` — full snapshot (same shape as SSE `state` event)
- `GET /api/bench/stream` — SSE: `state` (every 500ms), `request-start`, `tokens` (batched every 100ms per request), `request-end`, `request-error`, `done`

The frontend is SSE-driven only; there is no polling loop. `state` snapshot includes `buckets` (per-second output token counts) for the live throughput timeline.

### Key patterns

- All benchmark state lives in the module-level `run` object (process memory only). `snapshot()` strips `apiKey` before serializing.
- Prompt variation (`varyPrompt`) inserts random Korean prefix/suffix to bust vLLM prefix caching.
- Codebase payload (`includeCodebase`) is pre-generated once per run (4 seeded variants, ~80KB each), never per request, to keep the bench client off the CPU hot path.
- Token text SSE events are coalesced per request and flushed every 100ms so high concurrency doesn't bottleneck on SSE writes.

### Frontend (public/index.html)

Single file, vanilla JS, dark cockpit-style UI (hairline dividers, mono numerals, single emerald accent). EventSource drives everything; reconnects after 2s on error. Canvas charts: throughput timeline (buckets) and auto-mode concurrency profile (throughput bars + decode line, optimal highlighted).
