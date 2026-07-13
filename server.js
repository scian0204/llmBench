// LLM Bench v2 — vLLM 처리량 벤치마크
//
// 측정 방법론:
//  - 상수 동시성 worker pool: 요청이 끝나는 즉시 다음 요청 투입 (convoy effect 제거)
//  - 시스템 처리량 = 측정 구간 벽시계 시간 동안 생성된 총 output 토큰 ÷ 시간
//  - TTFT(첫 토큰까지 시간)와 decode TPS((토큰수-1) ÷ (마지막-첫 토큰 시간)) 분리
//  - latency / TTFT / decode TPS 의 p50 / p95 / p99 percentile
//  - 웜업 요청은 통계에서 제외
//  - 자동 모드: 동시성 1→2→4→…→max 기하 탐색, peak 대비 90% 미만이면 중단,
//    peak 이웃과의 midpoint 정제 라운드 1회
//
// API:
//  POST /api/bench/start   벤치 시작
//  POST /api/bench/stop    즉시 중지 (in-flight 요청 abort)
//  POST /api/bench/reset   상태 초기화
//  GET  /api/bench/state   현재 상태 스냅샷
//  GET  /api/bench/stream  SSE: state(500ms), request-start/end/error, tokens(100ms 배치), done

const express = require("express");
const { Agent, request } = require("undici");

const dispatcher = new Agent({
  connections: 512,
  keepAliveTimeout: 30_000,
});

const app = express();
app.use(express.json());
app.use(express.static("public"));

// ── 프롬프트 변주 (vLLM prefix 캐시 회피) ─────────────────────────────

const PREFIX_POOL = [
  "먼저 오늘의 날짜는 {date}입니다. ",
  "참고로 현재 사용자의 위치는 {city} 지역입니다. ",
  "이 요청의 고유 식별자는 #{id} 입니다. ",
  "다음 정보를 참고해주세요: 항목 수가 {n}개 있습니다. ",
  "현재 처리 중인 작업 순번은 {n} 번째입니다. ",
  "추가 맥락: 사용자 그룹 {g}에 속한 멤버의 요청입니다. ",
  "배경 정보: 지난 {n}일 동안 수집된 데이터를 바탕으로 합니다. ",
  "이 요청과 관련된 문서 ID는 DOC-{id} 입니다. ",
  "참고 데이터: 총 {n}명의 응답자가 참여한 설문 결과입니다. ",
  "현재 세션 정보: 세션 번호 SESS-{id}로 처리 중입니다. ",
];

const SUFFIX_POOL = [
  "결과를 설명할 때 반드시 #{id}라는 참조 번호를 포함해주세요.",
  "추가로 관련 분야 {topic}에 대한 짧은 의견도 함께 작성해주세요.",
  "마무리로 1부터 {n}까지의 숫자 총합도 함께 계산해서 알려주세요.",
  "가능하다면 {topic} 관점에서 어떻게 해석할지 짧게 덧붙여주세요.",
  "답변 마지막에 '참조: #{id}' 라는 문구를 꼭 포함시켜주세요.",
  "추가 요청: 위 내용을 {n}개의 키워드로 요약해서 마지막에 붙여주세요.",
  "마지막으로 {city} 지역에서 이 주제를 적용할 수 있는 방법을 한 줄 써주세요.",
  "답변에 {topic}과 이 주제의 연관성을 한 문장 정도로 설명해주세요.",
  "별도로 #{id}번 요청에 대한 답변임을 마지막에 명시해주세요.",
  "위 내용과 관련하여 {n}개의 참고 문헌을 가상으로 구성해서 알려주세요.",
];

const TOPICS = [
  "양자컴퓨팅", "블록체인", "로보틱스", "생체공학", "나노기술",
  "에너지 저장", "우주개발", "디지털 트윈", "엣지컴퓨팅", "메타버스",
  "기후변화", "지속가능성", "순환경제", "탄소중립", "신재생에너지",
  "데이터프라이버시", "사이버보안", "클라우드네이티브", "자동화", "스마트시티",
];

const CITIES = ["서울", "부산", "인천", "대구", "광주", "대전", "울산", "제주", "수원", "익산", "춘천", "강릉"];

function pickRandom(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function varyPrompt(basePrompt) {
  const id = Math.floor(Math.random() * 999999);
  const n = Math.floor(Math.random() * 900) + 100;
  const city = pickRandom(CITIES);
  const topic = pickRandom(TOPICS);
  const date = `202${Math.floor(Math.random() * 6)}-${String(Math.floor(Math.random() * 12) + 1).padStart(2, "0")}-${String(Math.floor(Math.random() * 28) + 1).padStart(2, "0")}`;
  const group = String.fromCharCode(65 + Math.floor(Math.random() * 26)) + Math.floor(Math.random() * 20);

  const prefix = pickRandom(PREFIX_POOL)
    .replace("{id}", id).replace("{n}", n).replace("{city}", city).replace("{g}", group).replace("{date}", date);
  const suffix = pickRandom(SUFFIX_POOL)
    .replace("{id}", id).replace("{n}", n).replace("{city}", city).replace("{topic}", topic);

  return `${prefix}본 질문: ${basePrompt} ${suffix}`;
}

// ── 대규모 코드베이스 컨텍스트 생성 (긴 입력 모의) ─────────────────────
// 요청마다 생성하면 벤치 클라이언트가 CPU 병목이 되므로,
// 벤치 시작 시 변형 몇 개를 미리 생성해 두고 요청마다 골라 씀.

const CODEBASE_TARGET_CHARS = 80_000; // 대략 2만 토큰 안팎

function generateCodebase(seed) {
  let s = (seed >>> 0) || 1;
  const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  const pick = (a) => a[Math.floor(rnd() * a.length)];

  const NOUNS = ["User", "Order", "Invoice", "Session", "Document", "Payment", "Ticket", "Report", "Channel", "Profile", "Account", "Webhook", "Audit", "Metric", "Cluster", "Tenant"];
  const FIELDS = ["name", "status", "ownerId", "tags", "priority", "score", "region", "currency", "amount", "email", "phase", "label", "quota", "source"];
  const TYPES = ["string", "number", "boolean", "Date", "string[]", "Record<string, unknown>", "number | null"];
  const STATUSES = ["'draft'", "'active'", "'pending'", "'archived'", "'failed'", "'closed'"];
  const VERBS = ["find", "create", "update", "archive", "validate", "sync", "publish", "merge", "export", "resolve"];

  const chunks = [
    "// ============================================",
    "// Reference codebase (benchmark payload, seed=" + seed + ")",
    "// ============================================",
    "",
  ];
  let length = chunks.join("\n").length;
  let moduleIdx = 0;

  while (length < CODEBASE_TARGET_CHARS) {
    moduleIdx++;
    const noun = pick(NOUNS);
    const lower = noun.toLowerCase();
    const lines = [];

    lines.push(`// modules/${lower}/${lower}Service_${moduleIdx}.ts`);
    lines.push(`export interface ${noun}Record${moduleIdx} {`);
    lines.push(`  id: string;`);
    lines.push(`  createdAt: Date;`);
    lines.push(`  updatedAt: Date;`);
    const nFields = 6 + Math.floor(rnd() * 6);
    for (let i = 0; i < nFields; i++) {
      lines.push(`  ${pick(FIELDS)}_${i}: ${pick(TYPES)};`);
    }
    lines.push(`}`);
    lines.push(``);
    lines.push(`export type ${noun}Status${moduleIdx} = ${STATUSES.slice(0, 3 + Math.floor(rnd() * 3)).join(" | ")};`);
    lines.push(``);
    lines.push(`export class ${noun}Service${moduleIdx} {`);
    lines.push(`  private cache = new Map<string, ${noun}Record${moduleIdx}>();`);
    lines.push(``);
    const nMethods = 3 + Math.floor(rnd() * 3);
    for (let i = 0; i < nMethods; i++) {
      const verb = pick(VERBS);
      lines.push(`  async ${verb}${noun}${i}(id: string, payload: Partial<${noun}Record${moduleIdx}>): Promise<${noun}Record${moduleIdx} | null> {`);
      lines.push(`    const existing = this.cache.get(id);`);
      lines.push(`    if (!existing) {`);
      lines.push(`      throw new Error('${noun} not found: ' + id);`);
      lines.push(`    }`);
      lines.push(`    const next = { ...existing, ...payload, updatedAt: new Date() };`);
      lines.push(`    if (Object.keys(payload).length === 0) {`);
      lines.push(`      return existing;`);
      lines.push(`    }`);
      lines.push(`    this.cache.set(id, next);`);
      lines.push(`    return next;`);
      lines.push(`  }`);
      lines.push(``);
    }
    lines.push(`}`);
    lines.push(``);
    lines.push(`export function validate${noun}${moduleIdx}(input: Partial<${noun}Record${moduleIdx}>): string[] {`);
    lines.push(`  const errors: string[] = [];`);
    lines.push(`  if (!input.id) errors.push('id is required');`);
    for (let i = 0; i < 4; i++) {
      lines.push(`  if (input.${pick(FIELDS)}_${i} == null) errors.push('${pick(FIELDS)}_${i} is missing');`);
    }
    lines.push(`  return errors;`);
    lines.push(`}`);
    lines.push(``);

    const block = lines.join("\n");
    chunks.push(block);
    length += block.length + 1;
  }

  return chunks.join("\n");
}

// ── 실행 상태 ──────────────────────────────────────────────────────

let run = null; // 현재(또는 마지막) 벤치 실행

function newRun(config) {
  return {
    phase: "running", // running | done | stopped | error
    mode: config.mode,
    config,
    startedAt: Date.now(),
    endedAt: null,
    nextId: 1,
    requests: [],     // { id, level, warmup, status, startedAt, endedAt, ttft, latency, inputTokens, outputTokens, decodeTPS, error }
    tokenBuckets: [], // 초 단위 output 토큰 수 (실시간 처리량 타임라인)
    planned: null,    // manual 모드 총 요청 수 (진행률)
    auto: config.mode === "auto"
      ? { status: "warmup", currentLevel: 0, profile: [], peak: null, optimal: null }
      : null,
    abort: new AbortController(),
    codebasePool: config.includeCodebase
      ? Array.from({ length: 4 }, (_, i) => generateCodebase(config.seedBase + i))
      : null,
    error: null,
  };
}

function composePrompt(basePrompt, vary, codebaseText) {
  let p = vary ? varyPrompt(basePrompt) : basePrompt;
  if (codebaseText) {
    // nonce가 코드베이스 앞에 와야 vLLM prefix 캐시가 20k 토큰 prefill을 스킵하지 못함
    const nonce = vary ? `요청 식별자: ${Math.random().toString(36).slice(2)}\n` : "";
    p = `${nonce}<참조 코드베이스>\n${codebaseText}\n</참조 코드베이스>\n\n${p}`;
  }
  return p;
}

function buildPrompt(r) {
  const c = r.config;
  return composePrompt(c.prompt, c.varyPrompt, r.codebasePool ? pickRandom(r.codebasePool) : null);
}

function bucketTokens(r, atMs, n) {
  const i = Math.floor((atMs - r.startedAt) / 1000);
  if (i < 0) return;
  while (r.tokenBuckets.length <= i) r.tokenBuckets.push(0);
  r.tokenBuckets[i] += n;
}

// ── SSE ───────────────────────────────────────────────────────────

let sseClients = [];

function sse(event, data) {
  if (sseClients.length === 0) return;
  const msg = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  for (const c of sseClients) c.write(msg);
}

// 토큰 텍스트는 요청별로 모았다가 100ms마다 한 번에 방송 (토큰마다 방송하면 벤치 클라이언트 병목)
const tokenBuffers = new Map(); // id -> text
let tokenFlushTimer = null;
let stateTimer = null;

function queueTokenText(id, text) {
  tokenBuffers.set(id, (tokenBuffers.get(id) || "") + text);
}

function flushTokens() {
  for (const [id, text] of tokenBuffers) {
    sse("tokens", { id, text });
  }
  tokenBuffers.clear();
}

function startTickers() {
  stopTickers();
  tokenFlushTimer = setInterval(flushTokens, 100);
  stateTimer = setInterval(() => sse("state", snapshot()), 500);
}

function stopTickers() {
  if (tokenFlushTimer) { clearInterval(tokenFlushTimer); tokenFlushTimer = null; }
  if (stateTimer) { clearInterval(stateTimer); stateTimer = null; }
}

// ── 단일 요청 실행 (undici 직접 스트리밍) ───────────────────────────

async function runOne(r, level, warmup) {
  const c = r.config;
  const rec = {
    id: r.nextId++,
    level,
    warmup,
    status: "running",
    startedAt: Date.now(),
    endedAt: null,
    ttft: null,
    latency: null,
    inputTokens: 0,
    outputTokens: 0,
    decodeTPS: null,
    error: null,
  };
  r.requests.push(rec);
  sse("request-start", { id: rec.id, level, warmup });

  const url = c.baseUrl.replace(/\/$/, "") + "/chat/completions";
  const headers = { "content-type": "application/json" };
  if (c.apiKey) headers.authorization = `Bearer ${c.apiKey}`;

  const payload = {
    model: c.model,
    messages: [{ role: "user", content: buildPrompt(r) }],
    stream: true,
    stream_options: { include_usage: true },
  };
  if (c.maxTokens !== null) payload.max_tokens = c.maxTokens;
  if (c.temperature !== null) payload.temperature = c.temperature;

  try {
    const res = await request(url, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
      dispatcher,
      signal: r.abort.signal,
      headersTimeout: 120_000,
      bodyTimeout: 600_000, // 청크 간 idle 타임아웃
    });

    if (res.statusCode !== 200) {
      const text = await res.body.text();
      throw new Error(`HTTP ${res.statusCode}: ${text.slice(0, 300)}`);
    }

    const decoder = new TextDecoder();
    let buf = "";
    let firstTokenAt = null;
    let lastTokenAt = null;
    let chunkTokens = 0; // usage 미제공 시 fallback (vLLM은 청크당 대략 1토큰)
    let usage = null;

    const parseFrame = (frame) => {
      for (const line of frame.split(/\r?\n/)) {
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") continue;
        let obj;
        try { obj = JSON.parse(data); } catch { continue; }
        const delta = obj.choices?.[0]?.delta?.content;
        if (delta) {
          const now = Date.now();
          if (firstTokenAt === null) firstTokenAt = now;
          lastTokenAt = now;
          chunkTokens++;
          if (!warmup) bucketTokens(r, now, 1);
          queueTokenText(rec.id, delta);
        }
        if (obj.usage) usage = obj.usage;
      }
    };

    for await (const chunk of res.body) {
      buf += decoder.decode(chunk, { stream: true });
      let m;
      // SSE는 CRLF 프레임 구분도 허용 (일부 게이트웨이/프록시)
      while ((m = buf.match(/\r?\n\r?\n/))) {
        const frame = buf.slice(0, m.index);
        buf = buf.slice(m.index + m[0].length);
        parseFrame(frame);
      }
    }
    buf += decoder.decode();
    if (buf.trim()) parseFrame(buf); // 미종결 마지막 프레임 (usage가 담길 수 있음)

    rec.endedAt = Date.now();
    rec.latency = (rec.endedAt - rec.startedAt) / 1000;
    rec.ttft = firstTokenAt !== null ? (firstTokenAt - rec.startedAt) / 1000 : null;
    // 대상 서버가 준 값이라 신뢰 불가 — 숫자 강제 (문자열이면 프론트 innerHTML까지 흘러감)
    rec.outputTokens = Number(usage?.completion_tokens) || chunkTokens;
    rec.inputTokens = Number(usage?.prompt_tokens) || 0;
    rec.decodeTPS =
      rec.outputTokens > 1 && lastTokenAt > firstTokenAt
        ? (rec.outputTokens - 1) / ((lastTokenAt - firstTokenAt) / 1000)
        : null;
    rec.status = "ok";
    sse("request-end", {
      id: rec.id,
      latency: round2(rec.latency),
      ttft: rec.ttft !== null ? round2(rec.ttft) : null,
      inputTokens: rec.inputTokens,
      outputTokens: rec.outputTokens,
      decodeTPS: rec.decodeTPS !== null ? round2(rec.decodeTPS) : null,
    });
  } catch (err) {
    rec.endedAt = Date.now();
    if (r.abort.signal.aborted) {
      rec.status = "aborted"; // 중지로 끊긴 요청 — 통계 제외
    } else {
      rec.status = "error";
      rec.error = String(err?.message || err);
      sse("request-error", { id: rec.id, error: rec.error });
    }
  }
}

// ── 상수 동시성 worker pool ─────────────────────────────────────────
// 요청이 끝나면 즉시 다음 요청 투입 — 측정 내내 동시성이 concurrency로 유지됨.

async function runLevel(r, concurrency, totalRequests, { warmup = false } = {}) {
  const t0 = Date.now();
  const before = r.requests.length;
  let dispatched = 0;

  const workers = [];
  for (let w = 0; w < concurrency; w++) {
    workers.push(
      (async () => {
        while (!r.abort.signal.aborted && dispatched < totalRequests) {
          dispatched++;
          await runOne(r, concurrency, warmup);
        }
      })()
    );
  }
  await Promise.all(workers);

  const recs = r.requests.slice(before);
  const ok = recs.filter((x) => x.status === "ok");
  const failed = recs.filter((x) => x.status === "error").length;
  const wallSec = (Date.now() - t0) / 1000;
  const tokens = ok.reduce((s, x) => s + x.outputTokens, 0);
  const decodeVals = ok.map((x) => x.decodeTPS).filter((v) => v !== null).sort((a, b) => a - b);
  const latVals = ok.map((x) => x.latency).sort((a, b) => a - b);
  const ttftVals = ok.map((x) => x.ttft).filter((v) => v !== null).sort((a, b) => a - b);

  return {
    concurrency,
    throughput: wallSec > 0 ? round2(tokens / wallSec) : 0, // 시스템 전체 tokens/sec (벽시계 기준)
    avgDecodeTPS: decodeVals.length ? round2(decodeVals.reduce((s, v) => s + v, 0) / decodeVals.length) : 0,
    p50Latency: round2(pct(latVals, 50)),
    p95Latency: round2(pct(latVals, 95)),
    p50TTFT: round2(pct(ttftVals, 50)),
    success: ok.length,
    failed,
    durationSec: round2(wallSec),
  };
}

// ── 벤치 실행 ───────────────────────────────────────────────────────

async function runManual(r) {
  const c = r.config;
  r.planned = c.totalRequests + c.warmupRequests;
  if (c.warmupRequests > 0 && !r.abort.signal.aborted) {
    await runLevel(r, 1, c.warmupRequests, { warmup: true });
  }
  await runLevel(r, c.concurrency, c.totalRequests);
}

async function runAuto(r) {
  const c = r.config;
  const auto = r.auto;

  if (c.warmupRequests > 0 && !r.abort.signal.aborted) {
    auto.status = "warmup";
    await runLevel(r, 1, c.warmupRequests, { warmup: true });
  }

  const tested = new Map(); // concurrency -> level result

  const probe = async (conc, status) => {
    auto.status = status;
    auto.currentLevel = conc;
    const lr = await runLevel(r, conc, conc * c.requestsPerLevel);
    if (r.abort.signal.aborted) return null; // 중지로 잘린 단계는 프로파일에 남기지 않음
    tested.set(conc, lr);
    auto.profile = [...tested.values()].sort((a, b) => a.concurrency - b.concurrency);
    return lr;
  };

  // 기하 탐색: 1, 2, 4, ..., maxConcurrency
  const levels = [];
  for (let cc = 1; cc <= c.maxConcurrency; cc *= 2) levels.push(cc);
  if (!levels.includes(c.maxConcurrency)) levels.push(c.maxConcurrency);

  let peak = null;
  for (const conc of levels) {
    if (r.abort.signal.aborted) break;
    const lr = await probe(conc, "probing");
    if (!lr) break; // 중지됨
    if (lr.success === 0) break; // 해당 단계 전멸 — 더 올려도 의미 없음
    if (!peak || lr.throughput > peak.throughput) {
      peak = lr;
      auto.peak = { concurrency: peak.concurrency, throughput: peak.throughput };
    } else if (lr.throughput < peak.throughput * 0.9) {
      break; // peak 대비 90% 미만 — 포화 지점 지남
    }
  }

  // ponytail: midpoint 정제 1라운드 — 더 정밀한 최적점이 필요하면 golden-section 탐색으로 확장
  if (peak && !r.abort.signal.aborted) {
    const sorted = [...tested.keys()].sort((a, b) => a - b);
    const i = sorted.indexOf(peak.concurrency);
    const neighbors = [sorted[i - 1], sorted[i + 1]].filter((v) => v !== undefined);
    for (const nb of neighbors) {
      if (r.abort.signal.aborted) break;
      const mid = Math.round((peak.concurrency + nb) / 2);
      if (tested.has(mid) || mid === peak.concurrency) continue;
      const lr = await probe(mid, "refining");
      if (lr && lr.success > 0 && lr.throughput > peak.throughput) {
        peak = lr;
        auto.peak = { concurrency: peak.concurrency, throughput: peak.throughput };
      }
    }
  }

  if (!r.abort.signal.aborted) {
    auto.status = "done";
    auto.optimal = peak ? { concurrency: peak.concurrency, throughput: peak.throughput } : null;
  }
}

async function execute(r) {
  try {
    if (r.mode === "auto") await runAuto(r);
    else await runManual(r);
    r.phase = r.abort.signal.aborted ? "stopped" : "done";
  } catch (err) {
    r.phase = "error";
    r.error = String(err?.message || err);
  } finally {
    r.endedAt = Date.now();
    if (run === r) {
      stopTickers();
      flushTokens();
      sse("done", { phase: r.phase, error: r.error });
      sse("state", snapshot());
    }
  }
}

// ── 통계 ───────────────────────────────────────────────────────────

function round2(v) {
  return Math.round(v * 100) / 100;
}

function pct(sortedAsc, p) {
  if (sortedAsc.length === 0) return 0;
  const idx = Math.min(sortedAsc.length - 1, Math.max(0, Math.ceil((p / 100) * sortedAsc.length) - 1));
  return sortedAsc[idx];
}

function distStats(values) {
  const sorted = [...values].sort((a, b) => a - b);
  if (sorted.length === 0) return { avg: 0, p50: 0, p95: 0, p99: 0 };
  return {
    avg: round2(sorted.reduce((s, v) => s + v, 0) / sorted.length),
    p50: round2(pct(sorted, 50)),
    p95: round2(pct(sorted, 95)),
    p99: round2(pct(sorted, 99)),
  };
}

function computeSummary(reqs) {
  const ok = reqs.filter((x) => x.status === "ok");
  if (ok.length === 0) return null;

  const windowStart = Math.min(...ok.map((x) => x.startedAt));
  const windowEnd = Math.max(...ok.map((x) => x.endedAt));
  const windowSec = (windowEnd - windowStart) / 1000;
  const totalOutputTokens = ok.reduce((s, x) => s + x.outputTokens, 0);

  return {
    totalRequests: ok.length,
    totalOutputTokens,
    totalInputTokens: ok.reduce((s, x) => s + x.inputTokens, 0),
    throughput: windowSec > 0 ? round2(totalOutputTokens / windowSec) : 0,
    requestRate: windowSec > 0 ? round2(ok.length / windowSec) : 0,
    latency: distStats(ok.map((x) => x.latency)),
    ttft: distStats(ok.map((x) => x.ttft).filter((v) => v !== null)),
    decodeTPS: distStats(ok.map((x) => x.decodeTPS).filter((v) => v !== null)),
  };
}

function liveTPS(r) {
  const b = r.tokenBuckets;
  if (b.length < 2) return 0;
  // 마지막(진행 중) 버킷 제외, 직전 5개 평균
  const done = b.slice(Math.max(0, b.length - 6), b.length - 1);
  if (done.length === 0) return 0;
  return round2(done.reduce((s, v) => s + v, 0) / done.length);
}

function snapshot() {
  if (!run) return { phase: "idle" };
  const r = run;
  const reqs = r.requests.filter((x) => !x.warmup);
  const ok = reqs.filter((x) => x.status === "ok");
  const now = r.endedAt ?? Date.now();
  const { apiKey, ...safeConfig } = r.config;

  return {
    phase: r.phase,
    mode: r.mode,
    error: r.error,
    config: safeConfig,
    elapsed: round2((now - r.startedAt) / 1000),
    active: r.requests.filter((x) => x.status === "running").length,
    completed: ok.length,
    failed: reqs.filter((x) => x.status === "error").length,
    planned: r.planned,
    liveTPS: liveTPS(r),
    totalOutputTokens: ok.reduce((s, x) => s + x.outputTokens, 0),
    summary: computeSummary(reqs),
    auto: r.auto
      ? {
          status: r.auto.status,
          currentLevel: r.auto.currentLevel,
          profile: r.auto.profile,
          peak: r.auto.peak,
          optimal: r.auto.optimal,
        }
      : null,
    buckets: r.tokenBuckets.slice(-600),
    requests: reqs.slice(-500).map((x) => ({
      id: x.id,
      level: x.level,
      status: x.status,
      latency: x.latency !== null ? round2(x.latency) : null,
      ttft: x.ttft !== null ? round2(x.ttft) : null,
      inputTokens: x.inputTokens,
      outputTokens: x.outputTokens,
      decodeTPS: x.decodeTPS !== null ? round2(x.decodeTPS) : null,
      error: x.error,
    })),
  };
}

// ── 설정 검증 ──────────────────────────────────────────────────────

function intIn(v, def, min, max) {
  const n = parseInt(v, 10);
  if (Number.isNaN(n)) return def;
  return Math.min(max, Math.max(min, n));
}

function floatIn(v, def, min, max) {
  const n = parseFloat(v);
  if (Number.isNaN(n)) return def;
  return Math.min(max, Math.max(min, n));
}

function validateConfig(body) {
  const { baseUrl, model, prompt } = body;
  if (!baseUrl || !model || !prompt) {
    throw new Error("baseUrl, model, prompt는 필수입니다");
  }
  try {
    new URL(baseUrl);
  } catch {
    throw new Error("baseUrl이 올바른 URL이 아닙니다");
  }

  const mode = body.mode === "auto" ? "auto" : "manual";
  const cfg = {
    mode,
    baseUrl: String(baseUrl),
    model: String(model),
    prompt: String(prompt),
    apiKey: body.apiKey ? String(body.apiKey) : null,
    // null = 요청 payload에서 생략 → vLLM 서버 기본값 적용
    maxTokens: body.maxTokens == null || body.maxTokens === "" ? null : intIn(body.maxTokens, 1024, 1, 128_000),
    temperature: body.temperature == null || body.temperature === "" ? null : floatIn(body.temperature, 0.7, 0, 2),
    varyPrompt: body.varyPrompt !== false,
    includeCodebase: body.includeCodebase === true,
    seedBase: Math.floor(Math.random() * 1_000_000),
  };

  if (mode === "manual") {
    cfg.concurrency = intIn(body.concurrency, 4, 1, 512);
    cfg.totalRequests = intIn(body.totalRequests, cfg.concurrency * 3, 1, 100_000);
    cfg.warmupRequests = intIn(body.warmupRequests, 0, 0, 20);
  } else {
    cfg.maxConcurrency = intIn(body.maxConcurrency, 64, 1, 512);
    cfg.requestsPerLevel = intIn(body.requestsPerLevel, 3, 1, 20); // 단계당 요청 수 = 동시성 × 이 값
    cfg.warmupRequests = intIn(body.warmupRequests, 2, 0, 20);
  }
  return cfg;
}

// ── HTTP 엔드포인트 ────────────────────────────────────────────────

app.post("/api/bench/start", (req, res) => {
  if (run && run.phase === "running") {
    return res.status(409).json({ error: "벤치마크가 이미 실행 중입니다" });
  }
  let cfg;
  try {
    cfg = validateConfig(req.body || {});
  } catch (err) {
    return res.status(400).json({ error: err.message });
  }
  run = newRun(cfg);
  startTickers();
  execute(run);
  res.json({ ok: true });
});

app.post("/api/bench/stop", (_req, res) => {
  if (run && run.phase === "running") {
    run.abort.abort();
  }
  res.json({ ok: true });
});

app.post("/api/bench/reset", (_req, res) => {
  if (run && run.phase === "running") {
    run.abort.abort();
  }
  stopTickers();
  tokenBuffers.clear();
  run = null;
  sse("state", { phase: "idle" });
  res.json({ ok: true });
});

app.get("/api/bench/state", (_req, res) => {
  res.json(snapshot());
});

// 대상 서버의 모델 목록 프록시 (브라우저 직접 호출은 CORS에 막힘)
app.post("/api/models", async (req, res) => {
  const { baseUrl, apiKey } = req.body || {};
  try {
    new URL(baseUrl);
  } catch {
    return res.status(400).json({ error: "baseUrl이 올바른 URL이 아닙니다" });
  }
  const headers = {};
  if (apiKey) headers.authorization = `Bearer ${apiKey}`;
  try {
    const r = await request(String(baseUrl).replace(/\/$/, "") + "/models", {
      headers,
      dispatcher,
      headersTimeout: 5_000,
      bodyTimeout: 5_000,
    });
    if (r.statusCode !== 200) {
      const text = await r.body.text();
      throw new Error(`HTTP ${r.statusCode}: ${text.slice(0, 200)}`);
    }
    const data = await r.body.json();
    const models = (data.data || []).map((m) => String(m.id)).filter(Boolean);
    res.json({ models });
  } catch (err) {
    // AggregateError(연결 거부 등)는 message가 비어 있어 code를 우선 노출
    const msg = err?.code || err?.errors?.[0]?.code || err?.message || String(err);
    res.status(502).json({ error: String(msg) });
  }
});

// 실제 전송될 프롬프트 표본 1개 생성 (변주는 요청마다 달라짐)
app.post("/api/bench/preview", (req, res) => {
  const { prompt, varyPrompt: vary, includeCodebase } = req.body || {};
  if (!prompt) {
    return res.status(400).json({ error: "prompt가 필요합니다" });
  }
  const codebase = includeCodebase === true ? generateCodebase(Math.floor(Math.random() * 1_000_000)) : null;
  const preview = composePrompt(String(prompt), vary !== false, codebase);
  res.json({ preview, chars: preview.length });
});

app.get("/api/bench/stream", (req, res) => {
  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.write(`event: state\ndata: ${JSON.stringify(snapshot())}\n\n`);
  sseClients.push(res);
  req.on("close", () => {
    sseClients = sseClients.filter((c) => c !== res);
  });
});

const PORT = process.env.PORT || 3100;
app.listen(PORT, () => {
  console.log(`LLM Bench v2: http://localhost:${PORT}`);
});
