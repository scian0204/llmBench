// 모의 vLLM 서버 — OpenAI 호환 /v1/chat/completions 스트리밍
// 검증용: 동시 16까지 요청당 ~40 tok/s, 초과하면 전체 처리량이 급감하도록 모델링.
// 자동 모드가 최적 동시성 ~16을 찾아내면 벤치 로직이 맞다는 뜻.
//
// 실행: npm run mock  (포트 8009)

const http = require("http");

const PORT = process.env.MOCK_PORT || 8009;
const PER_REQ_TPS = 40; // 포화 전 요청당 토큰 속도
const SATURATION = 16;  // 이 동시성까지 선형 확장

let active = 0;

// 전체 용량: active<=16 → 40*active (선형), 초과 → 640*16/active (감쇠)
function perRequestRate() {
  if (active <= SATURATION) return PER_REQ_TPS;
  return (PER_REQ_TPS * SATURATION * SATURATION) / (active * active);
}

const WORDS = ["하늘", "바다", "구름", "바람", "나무", "강물", "별빛", "노을", "숲길", "파도"];

const server = http.createServer((req, res) => {
  if (req.method !== "POST" || !req.url.includes("/chat/completions")) {
    res.writeHead(404).end();
    return;
  }

  let body = "";
  req.on("data", (c) => (body += c));
  req.on("end", () => {
    let payload;
    try {
      payload = JSON.parse(body);
    } catch {
      res.writeHead(400, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: "bad json" }));
      return;
    }

    const totalTokens = Math.min(payload.max_tokens || 128, 120);
    const promptChars = JSON.stringify(payload.messages || "").length;
    const id = "mock-" + Math.random().toString(36).slice(2, 10);

    res.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      connection: "keep-alive",
    });

    active++;
    let sent = 0;
    let closed = false;
    res.on("close", () => {
      if (!closed) { closed = true; active--; }
    });

    const chunk = (delta, usage) =>
      "data: " +
      JSON.stringify({
        id,
        object: "chat.completion.chunk",
        choices: delta !== null ? [{ index: 0, delta: { content: delta }, finish_reason: null }] : [],
        ...(usage ? { usage } : {}),
      }) +
      "\n\n";

    const ttft = 100 + active * 5; // 동시성 오를수록 첫 토큰 지연 증가

    const sendNext = () => {
      if (closed) return;
      if (sent >= totalTokens) {
        res.write(
          chunk(null, {
            prompt_tokens: Math.ceil(promptChars / 4),
            completion_tokens: totalTokens,
            total_tokens: Math.ceil(promptChars / 4) + totalTokens,
          })
        );
        res.write("data: [DONE]\n\n");
        closed = true;
        active--;
        res.end();
        return;
      }
      res.write(chunk(WORDS[sent % WORDS.length] + " "));
      sent++;
      setTimeout(sendNext, 1000 / perRequestRate()); // 동시성에 따라 토큰 간격 동적 변화
    };

    setTimeout(sendNext, ttft);
  });
});

server.listen(PORT, () => {
  console.log(`Mock vLLM: http://localhost:${PORT}/v1  (포화점=${SATURATION})`);
});
