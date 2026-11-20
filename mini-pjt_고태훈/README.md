# 미니 PJT: 가족 1주 식단 Agent

## 무엇을 푸나
구성원마다 기피·선호·하루 칼로리·먹는 끼니가 다른 4인 가족의 1주(21끼) 식단을, 예산과 지난 식단 중복까지 지키며 짜 주는 문제.

## 활용한 패턴 (Day 1~7)
- Day 4: Tool Calling — `@tool` + pydantic 입력 스키마로 도메인 도구 7개를 정의하고, LLM 은 요청을 인자로 바꾸기만 한다 (숫자 계산·식단 생성은 도구가 함)
- Day 3: Agent (ReAct 루프) — `langchain.agents.create_agent` 로 모델 ↔ 도구 반복 그래프 구성, Bedrock `ChatBedrockConverse`
- Day 2: RAG — 메뉴 197개·구성원 4명을 문서로 만들어 Chroma + Titan Embeddings v2 로 검색 (`search_docs`), 데이터 해시가 바뀌면 인덱스 자동 재생성
- Day 7: 구조화 출력 — 도구 결과를 `content_and_artifact` 로 나눠 LLM 에는 짧은 요약, trace 에는 전체 Plan JSON
- Day 7: 메모리 — `InMemorySaver` 체크포인터와 `thread_id` 로 같은 대화에서 조건(예산·출석 변경)을 이어받아 다시 짜기
- Day 5: Human-in-the-loop — `HumanInTheLoopMiddleware` 로 `save_plan` 실행 직전에 멈추고 "예" 승인 시에만 저장 (`Command(resume=...)`)
- Day 5: 가드레일 — 기피 항목(건강 정보) 마스킹, 가족 기본 정보 변경 거절, 프롬프트 인젝션 거절, 서버 쪽 저장 안전장치(`SAVE_ALLOWED`)
- Day 7: 평가 — 규칙 기반 자동 검사(`run_eval.py`) + RAGAS 지표(`ragas_eval.py`)

## 아키텍처

```
POST /query {question, thread_id?}
        │
        ▼
  src/agent.py  run_query ── 저장 확인 대기 중이면 "예/아니오" 판정 → 승인 재개 / 취소
        │
        ▼
  create_agent 그래프 (Bedrock Claude, InMemorySaver)
     model ──► HumanInTheLoopMiddleware.after_model ──► tools ──► model ...
                         │ save_plan 이면 멈춤(interrupt)
        tools
        ├─ plan_week      정수 계획 (PuLP 모델 + HiGHS 솔버): 기피·예산·칼로리·3주 중복·반복 제약
        ├─ check_plan     식단 조건 독립 재검사
        ├─ weekly_summary 주간 이유 요약 / 끼니별 이유 (data/reasons 파일)
        ├─ get_family_info, get_past_plan   가족 정보(기피는 비공개) / 지난 식단 기록
        ├─ save_plan      data/history 저장 (승인 후에만)
        └─ search_docs    src/retriever.py  Chroma + Titan 임베딩
        │
        ▼
  {answer, contexts: [{doc_id, text}], trace: [{step, input, output}]}
```

- LLM 은 질문 해석과 설명만 한다. 식단·금액·칼로리 합계·조건 검사는 모두 `src/tools.py` 의 코드가 한다
- contexts 는 LLM 과 무관하게 결정론적으로 모은다 (검색 결과, 조회한 구성원, 식단에 쓰인 메뉴 문서)

## 실행 방법

1. 설치 (Python 3.14 에서 확인)
   ```bash
   python -m venv .venv
   .venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt   # 또는 ./run.sh install
   ```
2. `.env.example` 을 `.env` 로 복사하고 값을 채운다 (`.env` 는 제출·커밋하지 않는다)

   | 키 | 설명 |
   |---|---|
   | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` | Bedrock 접속 계정 |
   | `MODEL_ID` | 대화 모델 (기본 `us.anthropic.claude-sonnet-4-6`) |
   | `EMBED_MODEL_ID` | 임베딩 모델 (기본 `amazon.titan-embed-text-v2:0`) |
   | `CHROMA_DIR` | 벡터 DB 폴더. 없으면 첫 실행 때 만든다 (약 1분) |
   | `HISTORY_DIR`, `REASONS_DIR` | 확정 식단 저장 폴더, 끼니별 이유 파일 폴더 |
   | `TODAY` | 선택. 기준 날짜(YYYY-MM-DD), 지난 주차 판단에 쓴다 |
3. 서버 실행
   ```bash
   uvicorn src.agent:app --port 8000   # 또는 ./run.sh serve
   ```
4. 요청 (`question` 필수, 대화를 이어 갈 때만 `thread_id`)
   ```bash
   curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
     -d '{"question": "이번 주 식단 35만 원으로 짜줘", "thread_id": "t1"}'
   ```
   같은 `thread_id` 로 `"토요일 저녁은 첫째가 없어. 다시 짜줘"` → `"확정해줘"`(저장 확인 질문) → `"예"`(저장)
5. 평가
   ```bash
   python evaluation/run_eval.py --out evaluation/round5_report.md   # 규칙 검사 (./run.sh eval round5)
   python evaluation/run_eval.py --only 1,12                          # 일부만 (같은 thread 문항은 함께)
   python evaluation/ragas_eval.py --rule-out evaluation/round5_report.md   # RAGAS + 규칙 검사를 한 번에
   ```
   평가는 thread 마다 시드 기록을 임시 폴더로 복사해 돌리므로 `data/history` 가 바뀌지 않고, `TODAY=2026-11-19` 로 고정된다.

## RAGAS 평가 결과
`evaluation/ragas_report.md` (30문항, 채점 모델 Bedrock Claude Sonnet 4.6, 임베딩 Titan v2, 채점 프롬프트 한국어 변환)
- context_recall: 0.81
- context_precision: 0.92
- faithfulness: 0.87
- answer_relevancy: 0.29
- (참고) factual_correctness(recall): 0.84

- 채점용 contexts 에는 제출 contexts 앞에 도구 결과 요약(식단 배치·총액 등)을 붙였다. 식단 답변의 근거가 메뉴 문서가 아니라 도구 결과에 있기 때문이다 (붙이기 전 식단 문항 faithfulness 0.03 → 붙인 뒤 0.98)
- answer_relevancy 가 낮은 것은 Titan 임베딩의 한국어 문장 유사도 한계(같은 뜻의 바꿔 말하기도 0.44~0.79)와, "저장할까요?" 같은 짧은 확인·거절 답변 때문이다

## 인-아웃 세트 통과율 (자체 평가)
`evaluation/test_queries.csv` 30건 (positive 12 · negative 6 · edge 7 · guardrail 5)
- 1차 (`round1_report.md`): 23 / 26 통과
- 2차 (`round2_report.md`, 개선 후): 26 / 28 통과
- 최종 (`round5_report.md`): 30 / 30 통과 (식단 12/12 조건 만족, 근거 없는 숫자 0, 기피 누출 0, 평균 8.8초)
- 개선폭: +7 건 (23 → 30. 주요 개선 사항: 메뉴 이름 속 글자 오탐 제거, 부반찬 기피 노출 차단, 주차가 있는 요청은 되묻지 않고 바로 식단 생성, 거절 시 예시 숫자·없는 기능 지어내기 금지, 지난 식단 겹침 최소화 옵션, 프롬프트 인젝션·기피 조회 가드레일)
- 중간 회차: `round3_report.md` 28/28, `round4_report.md` 28/28 (데이터 보강 후)

## 트라이앤에러 회고
- 시도했지만 실패한 접근
  - PuLP 내장 CBC 솔버: 빠듯한 예산·칼로리 조합에서 멈추고 시간 제한도 무시했다 → HiGHS(highspy)로 교체, 같은 모델이 5초 안팎에 풀림
  - 기피 항목 암호화: KMS 를 쓸 수 없어 포기 → 저장은 원본 그대로 두고 출력(answer·contexts·trace·벡터 DB) 전체에서 "누가 무엇을 기피하는지"를 마스킹
  - 데이터 보강만으로 대체 메뉴 줄이기: 공용 메뉴 12개를 추가해도 대체 횟수가 42 → 41회로 거의 그대로 (대체가 선호 점수를 더 받기 때문)
  - 프롬프트만으로 저장 통제: "확정할 때만 저장"을 프롬프트에만 맡기면 흔들림 → HumanInTheLoop + 서버 쪽 `SAVE_ALLOWED` 로 이중화
- 최종 채택한 접근
  - "LLM 은 해석·설명, 코드는 계산·검증": 식단은 정수 계획으로 짜고, 응답 숫자는 모두 도구 결과에서 인용하며, 평가가 출처를 자동 검사한다
  - 해가 없으면 벌점 여유 변수로 다시 풀어 "예산을 N원 늘리면 된다"처럼 무엇을 얼마나 풀면 되는지 안내
  - 답변은 끼니별 메뉴 + 주간 이유 요약, 끼니별 상세 이유는 파일로 분리해 답변 길이와 설명 품질을 함께 잡음
- 남은 한계 · 향후 개선 방향
  - 기피 대체 메뉴가 주 41회로 많다 → 목적 함수에 대체 감점 추가
  - "최대한 싸게"가 예산이 빠듯할 때 효과가 거의 없다 → 비용 가중치 재설계
  - 대화 메모리가 서버 메모리(InMemorySaver)라 재시작하면 사라진다 → SQLite/Postgres 체크포인터
  - answer_relevancy 측정 한계 → 다국어 임베딩(Cohere 등) 권한이 생기면 교체

## 핵심 코드 위치
- `src/agent.py:113` — 메인 에이전트 그래프 (`create_agent` + `HumanInTheLoopMiddleware`, 시스템 프롬프트는 `:29`)
- `src/agent.py:223` — `run_query` (저장 승인 재개, 응답 조립), `:286` — `POST /query`
- `src/tools.py` — 도메인 도구: 최적화 모델 `build_model`(`:352`), `make_plan`(`:806`), `verify_plan`(`:656`), `plan_week` 도구(`:968`), `save_plan`(`:1049`)
- `src/retriever.py` — RAG 파이프라인: `get_vectorstore`(`:100`), `search_docs`(`:139`), `contexts_for_plan`(`:159`)
- `evaluation/run_eval.py`, `evaluation/ragas_eval.py` — 자체 평가 스크립트
