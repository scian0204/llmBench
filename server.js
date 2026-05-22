const express = require("express");
const cors = require("cors");
const { EventEmitter } = require("events");
const { ChatOpenAI } = require("@langchain/openai");
const { HumanMessage } = require("@langchain/core/messages");

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static("public"));

const emitter = new EventEmitter();
emitter.setMaxListeners(100);

// ── Prompt variation (vLLM 캐시 회피) ────────────────────────────────
// 사용자의 기본 프롬프트에 무작위 컨텍스트를 삽입하여
// 각 요청마다 다른 프롬프트를 생성합니다.
// 구조/길이는 비슷하게 유지하면서 문자열만 다르게 만들어 캐시를 회피합니다.

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
  "추가 요청: 위 내용을 {n}개의 키워드로 요약해서到最后에 붙여주세요.",
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

function pickRandom(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function varyPrompt(basePrompt) {
  const id = Math.floor(Math.random() * 999999);
  const n = Math.floor(Math.random() * 900) + 100;
  const city = pickRandom(["서울", "부산", "인천", "대구", "광주", "대전", "울산", "제주", "수원", "익산", "춘천", "강릉"]);
  const topic = pickRandom(TOPICS);
  const date = `202${Math.floor(Math.random() * 6)}-${String(Math.floor(Math.random() * 12) + 1).padStart(2, "0")}-${String(Math.floor(Math.random() * 28) + 1).padStart(2, "0")}`;
  const group = String.fromCharCode(65 + Math.floor(Math.random() * 26)) + Math.floor(Math.random() * 20);

  const prefix = pickRandom(PREFIX_POOL)
    .replace("{id}", id).replace("{n}", n).replace("{city}", city).replace("{g}", group).replace("{date}", date);
  const suffix = pickRandom(SUFFIX_POOL)
    .replace("{id}", id).replace("{n}", n).replace("{city}", city).replace("{topic}", topic);

  return `${prefix}본 질문: ${basePrompt} ${suffix}`;
}

// 현재 측정 상태
let measuring = false;
let measuringConfig = null;
let results = [];
let autoState = null;
let startTime = 0;

// SSE 스트림 리스너 관리
let streamListeners = [];

function broadcastSSE(event, data) {
  const msg = `event: ${event}\ndata: ${JSON.stringify(data, null, 0)}\n\n`;
  for (const ws of streamListeners) {
    ws.write(msg, "utf8");
  }
}

app.get("/api/stream", (req, res) => {
  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.write(": connected\n\n", "utf8");
  streamListeners.push(res);

  req.on("close", () => {
    streamListeners = streamListeners.filter((s) => s !== res);
  });
});

app.post("/api/measure", async (req, res) => {
  if (measuring) {
    return res.json({ error: "측정이 이미 진행 중입니다" });
  }

  const {
    baseUrl,
    model,
    prompt,
    mode = "manual",
    concurrent = 1,
    maxTokens = 1024,
    temperature = 0.7,
    rounds = 1,
    apiKey,
    autoMaxConcurrent = 100,
    autoWarmup = 2,
    autoPerRound = 3,
    varyPrompt = true,
  } = req.body;

  if (!baseUrl || !model || !prompt) {
    return res.status(400).json({ error: "필수 입력값이 누락되었습니다" });
  }

  measuring = true;
  results = [];
  startTime = Date.now();
  autoState = null;

  if (mode === "auto") {
    measuringConfig = {
      mode: "auto",
      model,
      baseUrl,
      autoMaxConcurrent,
      autoPerRound,
    };
    runAutoBenchmark(
      baseUrl,
      model,
      prompt,
      maxTokens,
      temperature,
      apiKey,
      autoMaxConcurrent,
      autoWarmup,
      autoPerRound,
      varyPrompt
    ).finally(() => {
      measuring = false;
      broadcastSSE("bench-complete", { reason: "done" });
    });
  } else {
    measuringConfig = {
      mode: "manual",
      concurrent,
      rounds,
      model,
      baseUrl,
    };
    runManualBenchmark(
      baseUrl,
      model,
      prompt,
      concurrent,
      maxTokens,
      temperature,
      rounds,
      apiKey,
      varyPrompt
    ).finally(() => {
      measuring = false;
      broadcastSSE("bench-complete", { reason: "done" });
    });
  }

  res.json({ ok: true, message: "측정이 시작되었습니다" });
});

app.post("/api/stop", (_req, res) => {
  measuring = false;
  broadcastSSE("bench-complete", { reason: "stopped" });
  res.json({ ok: true, message: "측정 중지됨" });
});

app.get("/api/status", (_req, res) => {
  if (measuring || results.length > 0) {
    const elapsed = (Date.now() - startTime) / 1000;
    const totalTokens = results.reduce((s, r) => s + r.outputTokens, 0);
    const avgTPS = elapsed > 0 ? totalTokens / elapsed : 0;

    const resp = {
      measuring,
      completed: results.length,
      totalTokens,
      elapsed: Math.round(elapsed * 100) / 100,
      avgTPS: Math.round(avgTPS * 100) / 100,
      config: measuringConfig,
    };

    if (autoState) {
      resp.autoState = autoState;
    }

    return res.json(resp);
  }
  res.json({
    measuring: false,
    completed: 0,
    totalTokens: 0,
    elapsed: 0,
    avgTPS: 0,
    config: null,
  });
});

app.get("/api/results", (_req, res) => {
  const elapsed = (Date.now() - startTime) / 1000;
  const totalTokens = results.reduce((s, r) => s + r.outputTokens, 0);
  const avgTPS = elapsed > 0 ? totalTokens / elapsed : 0;

  const summary = {
    totalRequests: results.length,
    totalTokens,
    elapsed: Math.round(elapsed * 100) / 100,
    avgTPS: Math.round(avgTPS * 100) / 100,
    avgLatency:
      results.length > 0
        ? Math.round(
            results.reduce((s, r) => s + r.latency, 0) / results.length * 100
          ) / 100
        : 0,
    minTPS:
      results.length > 0
        ? Math.round(Math.min(...results.map((r) => r.tps)) * 100) / 100
        : 0,
    maxTPS:
      results.length > 0
        ? Math.round(Math.max(...results.map((r) => r.tps)) * 100) / 100
        : 0,
  };

  if (autoState) {
    summary.autoProfile = autoState.profile;
    summary.optimalConcurrent = autoState.optimalConcurrent;
    summary.optimalTPS = autoState.optimalTPS;
  }

  res.json({ results, summary });
});

app.get("/api/reset", (_req, res) => {
  measuring = false;
  results = [];
  autoState = null;
  measuringConfig = null;
  broadcastSSE("bench-complete", { reason: "reset" });
  for (const s of streamListeners) {
    s.end();
  }
  streamListeners = [];
  res.json({ ok: true });
});

// ── Manual benchmark ──────────────────────────────────────────

async function runManualBenchmark(
  baseUrl,
  model,
  prompt,
  concurrent,
  maxTokens,
  temperature,
  rounds,
  apiKey,
  varyPrompt
) {
  const totalRuns = concurrent * rounds;
  let completed = 0;

  for (let round = 0; round < rounds && measuring; round++) {
    const promises = [];
    for (let i = 0; i < concurrent; i++) {
      promises.push(
        runSingleRequest(
          { baseUrl, model, prompt, maxTokens, temperature, apiKey, varyPrompt },
          round * concurrent + i
        )
      );
    }

    const roundResults = await Promise.allSettled(promises);
    for (const r of roundResults) {
      if (r.status === "fulfilled") {
        results.push(r.value);
        completed++;
        console.log(
          `[${completed}/${totalRuns}] TPS: ${r.value.tps.toFixed(2)}, Latency: ${r.value.latency.toFixed(2)}s, Tokens: ${r.value.outputTokens}`
        );
      } else {
        console.error(`요청 실패:`, r.reason?.message || r.reason);
      }
    }
  }
}

// ── Auto benchmark ────────────────────────────────────────────
// Concurrency를 단계별로 늘려가며 TPS를 측정하고,
// 전체 TPS(동시접속 수 × 단일 TPS)가 가장 높은 지점을 찾습니다.
//
// 단계: 1 → 2 → 4 → 8 → 16 → 32 → 64 → (max 제한)
// 각 단계에서 여러 번 측정하여 평균을 내고,
// 전 단계 대비 전체 TPS가 90% 이하로 떨어지면 종료.

async function runAutoBenchmark(
  baseUrl,
  model,
  prompt,
  maxTokens,
  temperature,
  apiKey,
  maxConcurrent,
  warmupRounds,
  perLevelRounds,
  varyPrompt
) {
  const profile = []; // [{ concurrent, avgTPS, totalTPS, avgLatency, success, failed }]
  let peakTotalTPS = 0;
  let peakConcurrent = 1;

  autoState = {
    phase: "warmup",
    currentConcurrent: 1,
    peakTotalTPS: 0,
    peakConcurrent: 1,
    optimalConcurrent: null,
    optimalTPS: null,
    profile,
    done: false,
  };

  // Step 1: warmup — concurrency=1 로 warmup 횟수 측정 (모델 로딩 등 제외)
  console.log("[Auto] Warmup 시작");
  const warmupResults = await runConcurrencyLevel(
    { baseUrl, model, prompt, maxTokens, temperature, apiKey },
    1,
    warmupRounds,
    varyPrompt
  );
  if (warmupResults.failed > 0 && warmupResults.success === 0) {
    console.error("[Auto] Warmup 실패 — 모델 연결을 확인하세요");
    autoState.done = true;
    results.push(...warmupResults.results);
    return;
  }
  results.push(...warmupResults.results);

  // Concurrency 단계: 1, 2, 4, 8, 16, 32, 64, ...
  const steps = [1];
  for (let c = 2; c <= maxConcurrent; c *= 2) steps.push(c);
  if (!steps.includes(maxConcurrent)) steps.push(maxConcurrent);
  steps.sort((a, b) => a - b);

  for (const conc of steps) {
    if (!measuring) break;

    autoState.phase = "testing";
    autoState.currentConcurrent = conc;
    console.log(`[Auto] 동시접속=${conc} 측정 중...`);

    const levelResults = await runConcurrencyLevel(
      { baseUrl, model, prompt, maxTokens, temperature, apiKey },
      conc,
      perLevelRounds,
      varyPrompt
    );
    results.push(...levelResults.results);

    if (levelResults.success === 0) {
      console.error(`[Auto] 동시접속=${conc} 전체 실패`);
      profile.push({
        concurrent: conc,
        avgTPS: 0,
        totalTPS: 0,
        avgLatency: 0,
        success: 0,
        failed: levelResults.failed,
      });
      autoState.profile = profile;
      break;
    }

    // totalTPS = 동시접속 시 모든 요청의 TPS 합계
    const totalTPS =
      levelResults.results.reduce((s, r) => s + r.tps, 0);
    const avgTPS = totalTPS / levelResults.success;
    const avgLatency =
      levelResults.results.reduce((s, r) => s + r.latency, 0) /
      levelResults.success;

    profile.push({
      concurrent: conc,
      avgTPS: Math.round(avgTPS * 100) / 100,
      totalTPS: Math.round(totalTPS * 100) / 100,
      avgLatency: Math.round(avgLatency * 100) / 100,
      success: levelResults.success,
      failed: levelResults.failed,
    });

    autoState.profile = profile;

    if (totalTPS > peakTotalTPS) {
      peakTotalTPS = totalTPS;
      peakConcurrent = conc;
      autoState.peakTotalTPS = Math.round(peakTotalTPS * 100) / 100;
      autoState.peakConcurrent = peakConcurrent;
    }

    console.log(
      `[Auto] 동시접속=${conc} totalTPS=${totalTPS.toFixed(
        1
      )}, peakTotalTPS=${peakTotalTPS.toFixed(1)}`
    );

    // Degradation 감지: peak 대비 90% 미만이면 종료
    if (
      conc > peakConcurrent &&
      totalTPS < peakTotalTPS * 0.9
    ) {
      console.log(
        `[Auto] TPS 저하 감지 — 최적 동시접속=${peakConcurrent} (peak TPS=${peakTotalTPS.toFixed(1)})`
      );
      break;
    }
  }

  autoState.phase = "complete";
  autoState.done = true;
  autoState.optimalConcurrent = peakConcurrent;
  autoState.optimalTPS = Math.round(peakTotalTPS * 100) / 100;
  console.log(
    `[Auto] 완료 — 최적 동시접속=${peakConcurrent}, TPS=${peakTotalTPS.toFixed(1)}`
  );
}

async function runConcurrencyLevel(options, concurrent, rounds, varyPrompt) {
  let success = 0;
  let failed = 0;
  const levelResults = [];

  for (let round = 0; round < rounds && measuring; round++) {
    const promises = [];
    for (let i = 0; i < concurrent; i++) {
      promises.push(
        runSingleRequest(
          { ...options, varyPrompt },
          levelResults.length + failed + i
        )
      );
    }

    const settled = await Promise.allSettled(promises);
    for (const r of settled) {
      if (r.status === "fulfilled") {
        levelResults.push(r.value);
        success++;
      } else {
        failed++;
        console.error(`  요청 실패:`, r.reason?.message || r.reason);
      }
    }
  }

  return { results: levelResults, success, failed };
}

// ── Shared helpers ────────────────────────────────────────────

function createChatModel(options) {
  const config = {
    model: options.model,
    temperature: options.temperature,
    maxTokens: options.maxTokens,
    configuration: {
      baseURL: options.baseUrl,
    },
  };
  if (options.apiKey) {
    config.apiKey = options.apiKey;
  }
  return new ChatOpenAI(config);
}

async function runSingleRequest(options, index) {
  const t0 = Date.now();
  const chat = createChatModel(options);

  broadcastSSE("request-start", { id: index });

  try {
    const actualPrompt = options.varyPrompt
      ? varyPrompt(options.prompt)
      : options.prompt;
    const stream = await chat.stream([new HumanMessage(actualPrompt)]);
    let fullText = "";
    let usageMetadata = null;

    for await (const chunk of stream) {
      // stream() yields AIMessageChunk objects
      // content can be string or array of content parts
      let text = "";
      const raw = chunk.content || chunk.message?.content || "";
      if (typeof raw === "string") text = raw;
      else if (Array.isArray(raw)) text = raw.map((p) => p.text || p.value || "").join("");

      if (text) {
        fullText += text;
        broadcastSSE("token", { id: index, text });
      }
      // usage_metadata uses snake_case keys (output_tokens, input_tokens, total_tokens)
      if (chunk.usage_metadata) {
        usageMetadata = chunk.usage_metadata;
      }
    }

    const t1 = Date.now();
    const latency = (t1 - t0) / 1000;
    const outputTokens = usageMetadata?.output_tokens || 0;
    const totalTokens = usageMetadata?.total_tokens || 0;
    const inputTokens = usageMetadata?.input_tokens || 0;
    const tps = latency > 0 ? outputTokens / latency : 0;

    broadcastSSE("request-end", {
      id: index,
      latency: Math.round(latency * 100) / 100,
      outputTokens,
      inputTokens,
      totalTokens,
      tps: Math.round(tps * 100) / 100,
    });

    return {
      index,
      latency: Math.round(latency * 100) / 100,
      outputTokens,
      inputTokens,
      totalTokens,
      tps: Math.round(tps * 100) / 100,
      timestamp: new Date().toISOString(),
    };
  } catch (err) {
    broadcastSSE("request-error", { id: index, error: err.message || String(err) });
    throw err;
  }
}

const PORT = process.env.PORT || 3100;
app.listen(PORT, () => {
  console.log(`LLM Bench 서버 시작: http://localhost:${PORT}`);
});
