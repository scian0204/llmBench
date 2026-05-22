# LLM Bench

vLLM TPS 벤치마크 도구. LangChain JS를 기반으로 LLM 서버의 성능을 측정합니다.

## 기능

- **수동 모드** - 설정한 동시접속 수와 라운드로 TPS 측정
- **자동 모드** - 동시접속 수를 단계별(1→2→4→8→...)로 증가시키며 최적 TPS 탐색
- **실시간 SSE 스트리밍** - 측정 진행 상황 실시간 조회
- **Prompt Variation** - 무작위 컨텍스트 삽입으로 vLLM 캐시 회피

## 사용법

```bash
npm install
npm start
```

서버는 기본 `http://localhost:3100`에서 실행됩니다.

## API

- `GET /api/stream` - SSE 실시간 스트림
- `POST /api/measure` - 벤치마크 시작
- `POST /api/stop` - 벤치마크 중지
- `GET /api/status` - 진행 상태 조회
- `GET /api/results` - 결과 조회
- `GET /api/reset` - 상태 초기화

## 의존성

- Express, LangChain JS, CORS, dotenv
