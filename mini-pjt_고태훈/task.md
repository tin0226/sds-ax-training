# TASK · 가족 1주 식단 Agent 구현 계획

근거 문서: `SERVICE.md`, `CLAUDE.md`, 설계 문서 `~/.gstack/projects/mini-pjt/SDS-master-design-20261118-161221.md`
구조 참고: `reference/final_scenario.py` (Bedrock LLM 생성, `@tool` 도구, RAG 검색 도구, 에이전트 조립 흐름)

- 단위: task 1개 = 약 1시간
- 전체: **16개 task, 약 16시간**. RAG 포함, 선택 task까지 모두 진행한다
- 진행 순서는 필수 10개(T1~T10) → 나머지 6개(T11~T16)
- 각 단계에서 새 파일을 만들기 전에 사용자에게 확인받는다

---

## 0. 구현 전에 정해 둘 기본값 (설계 문서 Open Questions)

정하지 않으면 T1·T4에서 막힌다. 아래 기본값으로 진행하고, 바꾸고 싶으면 T1 전에 고친다.

| # | 질문 | 기본값 |
|---|---|---|
| OQ1 | 아침 메뉴 적합성, 빵+국 조합 | 제한 없음 (규칙 추가 안 함) |
| OQ2 | 집 밖 끼니가 있는 날 칼로리 | 먹는 끼니 수 k에 비례: [min×k/3, max×k/3] |
| OQ3 | 주 안 반복 상한 | 모든 구성 2회 (밥은 3회) |
| OQ4 | 기피 정도 | 모두 금지 (감점 없음) |
| OQ5 | 같은 구성 기피자가 여럿 | 각자 1팩 |
| OQ6 | 서버 위치 | `src/agent.py` 안에 FastAPI `app` |
| -   | `extra_avoid` 형식 | `{"구성원 또는 '전체'": [menu_id 또는 tag]}`, plan_week·check_plan이 같은 함수로 태그를 펼친다 |
| -   | `cost_weight` | 1 또는 5만 허용 |
| -   | 가격 | 모두 100원 단위 |
| -   | 기본 주간 예산 | 400,000원 (주 72인분 기준, 평균 메뉴 선택 시 약 39만원) |
| -   | 칼로리 범위 | 모두 1인분이라 같은 상을 먹는 가족의 칼로리가 비슷하다. 그래서 범위가 겹치게 잡는다: 아빠 1,900~2,600 / 엄마 1,600~2,200 / 첫째 1,800~2,500 / 둘째 1,500~2,200 |
| -   | 주식 종류(밥·떡·면·빵) | 태그처럼 취급한다 (기피·선호에 "면" 등을 쓸 수 있음) |
| -   | 메뉴 칼로리 | T5에서 10% 낮춤 (한 끼 평균 745 → 약 670kcal). 평균이 칼로리 범위의 가장자리에 걸려 해 찾기가 매우 어려웠음 |
| -   | 기피 정보 보호 | 암호화 없이 출력 마스킹 (KMS 권한 없음 → 암호화는 하지 않기로 결정). 이유 문장 "둘째는 기피로 대체", 벡터 DB·contexts 구성원 문서 "기피: 비공개", check_plan 위반 메시지에 메뉴 이름 생략, 둘째 note 의 기피 문장 삭제. T8 `get_family_info` 는 기피를 "비공개 N개"로만. 남는 한계: eaters·대체 메뉴로 기피를 짐작하는 것은 막지 못함. 질문에서 사용자가 직접 말한 extra_avoid 는 trace 에 남김(check_plan 검증용) |
| -   | 솔버 | HiGHS(highspy, PuLP 인터페이스), threads=1, gap 0.2%, 시간 제한 60초. PuLP 내장 CBC 2.10.3 은 빠듯한 예산에서 멈추고 시간 제한을 무시해 교체 (CLAUDE.md 반영) |
| -   | 출석표 | 아빠 평일 점심 X / 첫째 주말 아침·일요일 저녁(학원) X / 둘째 월~목 점심(급식) X / 엄마 전부 O |

---

## 1. 필수 task (MVP, 10시간)

### T1. 프로젝트 뼈대와 환경 (1h)
- 폴더 `src/`, `data/history/`, `evaluation/` 생성
- 패키지 설치: langchain, langgraph, langchain-aws, langchain-chroma, pulp, fastapi, uvicorn, python-dotenv
- `.env`: AWS 자격 증명, `MODEL_ID`, `EMBED_MODEL_ID`, `HISTORY_DIR=data/history`
- Bedrock 호출 1회, `pulp.PULP_CBC_CMD(threads=1)` 로 작은 문제 1개 풀어 보기
- 완료 기준: LLM 응답과 CBC 풀이가 둘 다 로컬에서 성공
- 참고: `requirements.txt`, `.env.example` 은 새 파일이라 만들기 전에 확인받는다 (CLAUDE.md "요청하지 않은 파일 금지")

### T2. 더미 데이터 작성 (1h)
- `data/menus.csv`: 주식 16(밥 3·떡 5·면 3·빵 5), 국 30, 메인반찬 30, 부반찬 100. 열: `menu_id, category, staple_type, name, price_per_pack, kcal_per_serving, servings_per_pack, tags, description`
- `data/family.json`: 기본 예산 300,000원, 4명(기피·선호·하루 칼로리 범위·7×3 출석표). 아빠 평일 점심 없음, 첫째 주말 아침 없음
- `data/history/2026-W45.json` ~ `2026-W47.json`: 메뉴 id가 `menus.csv`에 실제로 있는 3주치
- 완료 기준: 메뉴 id 참조 오류 0건, 기본 예산으로 사람이 봐도 해가 있을 만한 가격대

### T3. 도메인 공통 로직 (`src/tools.py` 상단) (1h)
- 데이터 로딩, 태그→메뉴 id 펼치기(`Avoid(p)`), 선호 점수 `pref(p,m)` (menu_id가 tag보다 우선, 기피가 선호보다 우선)
- 출석표 + `attendance_changes` 반영 → `A[s]`
- 주차 계산: target week = history 최신 주 + 1, 최근 3주 (국, 메인반찬) 조합 집합 `H`
- `eaters(s,m)`, `packs(s,m)` 계산
- 완료 기준: Python에서 직접 불러 값이 손 계산과 일치

### T4. `plan_week` 최적화 모델 1: 상차림·기피·예산 (1h)
- 변수 `x[s,m]`, `z[s,p,m]`
- 제약 1~5: 구성 개수, 아무도 못 먹는 메뉴 금지, 단일 구성 대체, 부반찬 최소 1개, 예산
- 목적 함수: 선호 − cost_weight×(비용/10,000) − ε·rank
- 완료 기준: 기본 데이터로 해가 나오고 기피 위반 0건

### T5. `plan_week` 최적화 모델 2: 칼로리·중복·반복 + Plan JSON (1h)
- 제약 6 칼로리(사람·요일), 7 최근 3주 조합 중복 (`w[s,h]`), 8 주 안 반복 상한.
- 결과를 설계 문서의 **Plan JSON** 형식으로 변환 (끼니별 eaters, substitutes, reason, daily_kcal, repeat_from_history)
- `plan_id`(입력+데이터 해시)로 메모리 보관
- 풀이 시간 측정. 30초 넘으면 대체 변수를 선호 상위 10개로 축소
- 완료 기준: 같은 인자로 두 번 돌려 같은 결과, 풀이 30초 이하

### T6. `check_plan` 독립 검증기 (1h)
- Plan JSON을 받아 `menus.csv`·`family.json`·history로 다시 계산: 기피, 예산, 칼로리, 중복, 반복, 상차림 구성, 합계 금액
- 최적화 코드와 함수를 공유하지 않고 따로 계산 (태그 펼치기만 T3 공용 함수 사용)
- 완료 기준: T5 결과에 위반 0건, 일부러 기피 메뉴를 넣은 Plan JSON에서 위반을 잡아냄

### T7. 해 없음 처리: 사전 검사 + 완화 재풀이 (1h)
- 사전 검사: 출석자별·끼니별·구성별 먹을 수 있는 메뉴가 없으면 `no_allowed_menu`
- 완화 재풀이: 예산·칼로리(양쪽)·중복·반복(메뉴별)에 여유 변수 + 벌점(우선순위: 중복 < 반복 < 예산 < 칼로리), 결과를 "예산 +18,000원 필요" 같은 문장 목록으로 `relaxations` 반환
- 완료 기준: 예산 10만원으로 돌리면 `infeasible` + 예산 완화안이 나옴

### T8. 나머지 도구와 `@tool` 래핑 (1h)
- `get_family_info(member?)` (기피는 "비공개 N개"로만 돌려줌), `save_plan(plan_id)` (같은 주 파일 있으면 덮어쓰지 않음, 모르는 plan_id면 안내 문구)
- `plan_week`, `check_plan` 포함 5개 도구에 `@tool` + 한국어 docstring (LLM이 인자를 채울 수 있게 인자 설명을 구체적으로)
- 완료 기준: 각 도구를 `.invoke({...})` 로 직접 호출해 정상 응답

### T9. 에이전트 + `POST /query` (`src/agent.py`) (1h)
- `create_agent(ChatBedrockConverse(temperature=0), tools, system_prompt, checkpointer=InMemorySaver())`
- 시스템 프롬프트: 숫자는 도구 결과만 인용, 식단 요청은 반드시 `plan_week`, "확정"일 때만 `save_plan`, 같은 대화에서는 직전 `week_id`·예산·변경을 이어받기
- FastAPI `POST /query {question, thread_id?}` → `{answer, contexts, trace}`
  - `trace`: 이번 요청에서 새로 생긴 메시지만 잘라 도구 호출 순서·인자·결과 기록
  - `contexts`: 마지막 Plan JSON에 나온 메뉴·구성원 문서 (서버가 결정론적으로 조립)
- 완료 기준: `uvicorn src.agent:app` 후 "이번 주 식단 짜줘 예산 30만원" 이 세 키로 응답

### T10. 평가 질문 + 평가 스크립트 + 리포트 (1h)
- `evaluation/test_queries.csv` 10개 이상: 기본 식단, 예산 지정, 출석 변경, 추가 기피, 같은 thread 다시 짜기, 확정, 해 없음, 가족 정보 질문
- `evaluation/run_eval.py`: thread마다 시드 history를 임시 폴더로 복사 → `/query` 호출 → 마지막 `plan_week` 결과를 `check_plan`으로 검사 → `answer` 숫자가 trace·contexts에 있는지 검사
- `evaluation/report.md` 생성: 통과율, 숫자 불일치 수, 응답 시간
- 완료 기준: 리포트가 생성되고 실패 항목이 무엇인지 보임

---

## 2. 이어서 할 task (T11~T16)

### T11. RAG: `src/retriever.py` + `search_docs` 도구 (1h)
- 메뉴 행·구성원 요약을 문서로 만들어 Chroma + Titan Embeddings 인덱싱
- `search_docs` 도구를 에이전트에 연결하고 검색 결과를 `contexts`에 합침
- 결정: RAG는 넣는다 (폴더 규약의 `src/retriever.py`)

### T12. 평가 실행 후 버그 수정 1차 (1h)
- 리포트 실패 항목 수정, 시스템 프롬프트 다듬기 (인자 추출 오류, 숫자 지어내기)

### T13. 평가 실행 후 버그 수정 2차 (1h)
- 같은 thread 이어 가기, 확정 후 다시 짜기, 해 없음 답변 문구 점검

### T14. 끼니별 reason 품질 (1h)
- "아빠 없는 평일 점심 → 아이들 선호", "둘째 매운맛 기피로 돈가스 대체" 수준의 이유 문장을 도구가 생성

### T15. 데이터 보강 (1h)
- 칼로리·가격 분포 조정, 선호·기피를 다양화해 평가 질문이 의미 있게 갈리도록

### T16. 실행 안내와 제출 정리 (1h)
- 실행 방법(.env, uvicorn, 평가 명령) 정리, 불필요 파일 정리

---

## 3. 공수

| 구분 | task | 시간 |
|---|---|---|
| 핵심 | T1 ~ T10 | 10h |
| 이어서 | T11 ~ T16 | 6h |
| 합계 | 16개 | 16h |

결정: 16개 모두 진행한다 (가용 10시간을 넘는 6시간은 추가로 확보). 위험 요소는 T4·T5 최적화 모델(디버깅이 길어질 수 있음)이므로 이 둘은 앞쪽 시간에 몰아서 한다.

## 진행 현황

| task | 상태 |
|---|---|
| T1 | 완료 · CBC 풀이, Bedrock LLM·임베딩(1024차원), create_agent·fastapi import 확인. PuLP 4.x는 CBC가 없어 2.9로 고정 |
| T2 | 완료 · 메뉴 185개(주식 16, 국 32, 메인 32, 부반찬 105), 가족 4명, 기록 W45~W47(새 출석표 반영). 참조 오류 0건, 한 끼 평균 745kcal |
| T3 | 완료 · `src/tools.py` 공통 로직(데이터 로딩, 기피·선호, 출석 변경, 칼로리 범위, 주차·최근 3주 조합 H, eaters·packs). 손 계산과 일치 |
| T4 | 완료 · `build_context` / `build_model`(제약 1~5, 목적 함수) / `solve_model` / `read_solution`. 기피 위반 0건, 같은 입력 같은 해. gap 0.2% 적용 |
| T5 | 완료 · 제약 6~8(칼로리·최근 3주 조합 중복·주 안 반복), Plan JSON(`format_plan`), plan_id·보관소, `make_plan`. 4개 경우 위반 0건, 풀이 1~9초, 같은 입력 같은 plan_id·식단. 예산 28만원 이하는 해 없음 → T7. 솔버를 HiGHS로 교체 |
| T6 | 완료 · `verify_plan`(check_plan 본체): 구성·카탈로그·기피·팩/금액/총액·예산·칼로리·3주 중복·반복을 독립 재계산. 정상 식단 3개 위반 0건, 변조 10종 모두 검출 |
| T11 | 완료 (T7보다 먼저 진행) · `src/retriever.py`: 메뉴 185 + 구성원 4 문서, Chroma + Titan v2(첫 인덱싱 58초, 이후 0.2초, 데이터 해시가 바뀌면 재생성), `search_docs`(kind 필터, exclude 태그 제외), `contexts_for_plan`(결정론적). 한계: 벡터 검색은 기피/선호를 구분 못 함 → 정확한 판단은 get_family_info |
| T7 | 완료 · `precheck`(먹을 메뉴 없음 → `no_allowed_menu`), `build_model(relax=True)`(예산·칼로리 양쪽·3주 중복·메뉴별 반복에 벌점 여유 변수, 기피에는 없음), `describe_relaxations`, `make_plan` 상태 ok / no_allowed_menu / infeasible / timeout. 예산 10만원 → "186,000원 늘려야 함", 제안 예산 286,000원으로 재시도하면 ok·검사 통과. 메인 5개만 허용 → 반복 상한 완화안. 완화 재풀이 포함 약 12초 |
| T8 | 완료 · `@tool` plan_week(pydantic 스키마, content_and_artifact: LLM 요약 3.4천 자 / Plan JSON 1.9만 자는 artifact), check_plan(plan_id), get_family_info(기피 "비공개 N개"), save_plan(덮어쓰기 금지, 모르는 id 안내), `DOMAIN_TOOLS`. 임시 기록 폴더로 저장 → 다음 계획 주 W49로 넘어감 확인 |
| T9 | 완료 · `src/agent.py`(create_agent + InMemorySaver, 시스템 프롬프트, POST /query, trace 는 이번 요청 메시지만, contexts 결정론적 조립, 오류여도 세 키). save_plan 확정 안전장치(SAVE_ALLOWED contextvar). 시나리오 6개 확인: 35만원 → ok·검사 통과 / 같은 thread 다시 짜기 → 예산·week_id 이어받음 / "확정" → W48 저장(임시 폴더) / thread 없이 확정 → thread_id 안내 / 둘째 기피 → get_family_info·비공개 / 10만원 → 예산 완화안만 안내. sonnet-4-5 일일 토큰 한도 초과로 **MODEL_ID 를 us.anthropic.claude-sonnet-4-6 으로 변경**(.env, .env.example) |
| T10 | 진행 중 · `evaluation/test_queries.csv`(16문항: 식단 6, 같은 thread 4(다시 짜기·확정·확정 후 다시 짜기), 해 없음 1, thread 없는 확정 1, 질의응답 4 — q15 "지난주 식단" 추가하며 `get_past_plan` 도구 신설(tools.py·agent.py), q15 단독 실행 통과. q16 "예산 관계없이 한 달 겹치지 않게"는 기능 추가 없이 문항만: 겹침 0끼 + 답변에 비교 기준 '3주' 명시 여부 확인), `evaluation/run_eval.py`(TestClient 로 POST /query, thread마다 시드 기록 복사, check_plan·expect·숫자 출처·기피 누출 검사, report.md 작성). 전체 실행 완료(sonnet-4-6): **16/16 통과**, 식단 9/9 check_plan 만족, 근거 없는 숫자 0, 기피 누출 0, 평균 13.9초·최대 28.4초 → `evaluation/report.md`. 관찰: q05 답변 "엄마 가지튀김 제외"처럼 eaters 기반 표현이 누출 검사(기피·싫어·안 먹 키워드)에 안 걸림 / q16 get_past_plan 6회 호출, "예산 관계없이"는 기본 예산 40만원 사용 |
| 추가 | 완료 · 저장 전 사용자 확인: `HumanInTheLoopMiddleware(interrupt_on={"save_plan"})`. "확정" → 서버가 "W48 식단(총액 …원)을 저장할까요? 예/아니오" → 같은 thread 에서 "예" 승인 시에만 저장, "아니오"면 취소, 다른 질문이면 취소 후 처리. SAVE_ALLOWED 는 승인 재개 요청에서만 True. trace 는 재개 시 앞 요청의 도구 호출도 포함, 대기 중이면 `pending_confirmation`. CSV 19문항(q08 확인 대기, q08b 승인 저장, q17~q19 거절 흐름) — t1·t2 8문항 통과 |
| 추가 | 진행 중 · 지난 주차는 다시 짜기만 가능, 확정 불가: `today()`(.env TODAY, 없으면 실제 날짜)·`current_week()`·`is_past_week()`, Plan JSON `is_past_week`, 요약에 "확정 불가" 주의, `save_block_reason`(지난 주차·이미 확정·없는 plan_id) → HITL `when` 조건으로 저장 확인을 묻지 않고 도구가 사유 반환(도구 안에서도 재검사). run_eval 은 TODAY=2026-11-19 고정, 서버 오류 응답은 실패 처리. 도구 단계 검증 완료(W44 짜기 ok·저장 거부, W48 저장 가능). 계정 변경 후 에이전트 경유 확인: t1(q06~q09) 통과, q20 은 처음에 "다시 짜줘"를 이전 대화 요청으로 읽어 thread_id 안내 → 프롬프트 수정("주차·조건이 있으면 '다시'여도 새 요청") 후 q20·q21·q11 통과. q11(식단 없이 "확정해줘") 답변을 "확정할 식단이 없습니다."로 시작하도록 프롬프트·기대값 변경, 통과 |
| 추가 | 완료 · 지난 식단 시드 W41~W44(4주, 각 21끼) 추가 → 기록 W41~W47 7주. W45~W47과 같은 생성 규칙(출석표·기피 대체·부반찬 최소 1개), 규칙 위반 0, 기존 파일 해시 동일. W48 계획은 그대로(W45~47 비교, 같은 plan_id), W44 다시 짜기는 W41~43과 비교해 ok·검사 통과·저장 거부 |
| 추가 | 완료 · 가족 기본 정보 변경 불가: family.json 을 쓰는 도구 없음(코드상 이미 불가) + 시스템 프롬프트 규칙(첫 문장 고정 거절, 이번 주 한정 출석 변경·추가 기피만 허용, 기피 해제는 이번 주도 불가, 없는 기능·다른 창구 지어내기 금지, 거절 시 기피 항목 되풀이 금지, 예시에도 숫자 지어내기 금지). 테스트 q22~q25(기피·칼로리·기본 일정·선호) + `family_unchanged`·`answer_not` 검사 추가 → 4/4 통과. CSV 25문항 |
| 평가 1회차 | `evaluation/round1_report.md`: 23/26 통과, 식단 11/11 check_plan 만족, 근거 없는 숫자 0, 평균 11.0초. 실패 3건 분석 → 모두 평가 쪽 문제: q07·q09 누출은 메뉴 이름 "생선까스"의 '생선' 오탐(누출 검사에서 메뉴 이름 제거 후 검사, 근거 문장 표시하도록 수정), q21 은 W44 가 시드에 추가돼 `no_file` 기대값이 틀림(`file_unchanged` 로 변경). t1·t3 재실행 7/7 통과. round1 리포트는 기준 기록으로 보존 |
| T12 | 완료 · 1회차 리포트·답변 검토로 찾은 에이전트 문제 수정: (1) 부반찬 기피 노출("엄마 가지튀김 제외") → 이유 문장에서 "누가 어떤 부반찬 안 먹음" 삭제(eaters 에만 남김) + 프롬프트 금지 + 누출 검사에 "구성원+기피 메뉴 이름+제외/안 먹음" 패턴 추가, (2) q16 get_past_plan 7회 호출 → 중복 비교는 plan_week 가 자동(최근 3주)이므로 plan_week 만 부르고 기준을 밝히도록 프롬프트 수정. q01·q05·q16 재실행 3/3 통과(q16 도구 호출 8 → 1회) |
| 추가 | 완료 · 주간 이유 요약: `weekly_summary(plan_id, slot?)` 도구 신설 — 주간 이유를 코드가 Plan JSON 에서 계산(상차림·총액, 안 먹는 끼니, 선호 반영 끼니 수, 기피 대체 횟수, 다양성, 칼로리·조건 검사), slot 을 주면 그 끼니 상세 이유. 끼니별 이유는 `REASONS_DIR/<plan_id>.json`(기본 data/reasons, .gitignore)에 저장, 서버 재시작 후에도 파일에서 조회. plan_week LLM 요약에서 끼니별 이유·금액·칼로리 제거(3.4천 → 1.8천 자), 답변은 "주간 이유 요약 + 끼니별 메뉴 한 줄"(표·이모지 없음). 평가: REASONS_DIR 임시 폴더, `reasons_file`·`tool_arg` 조건, q26·q27(t4) 추가 → q01·q26·q27 3/3 통과. SERVICE.md 가치 문장 수정 |
| T13 | 완료 · 전체 재평가 `evaluation/round2_report.md`: 26/28 통과, 평균 9.2초(1회차 11.0초). 점검 대상(같은 thread 누적 조건, 확정→예/아니오, 확정 후 다시 짜기, 해 없음 안내) 모두 정상. 실패 2건은 LLM 이 규칙을 들쭉날쭉 따른 것: q20(주차가 있는데 되물음) → 식단 요청 판단 순서 명시("조건이 하나라도 있으면 무조건 plan_week, 되묻지 않음"), q25(예시에 "30만 원" 지어냄) → 안내 시 예시 대신 고정 문장 사용. q20·q21·q23·q25 2회 반복 모두 통과 |
| T14 | 완료 · 사용자 요청 반영: (1) 식단 없이 "확정"/조건 없는 "다시 짜줘"는 thread_id 표현 없이 "확정할(다시 짤) 식단이 없습니다. 먼저 식단을 짜 달라고 요청해 주세요."만, (2) 식단 답변 순서를 "주차·총액 → 21끼 끼니별 메뉴 한 줄(생략 금지) → 마지막에 주간 이유 요약 → plan_id·이유 파일 경로·확정 안내"로 변경, (3) 끼니별 이유를 사람이 읽는 `data/reasons/<plan_id>.md`(주간 요약 + 끼니별 메뉴·대체·금액·이유)로도 정리(json 은 도구용), (4) 끼니별 이유에 "선호 반영: 구성원-메뉴" 추가. 평가: 식단 답변에 21끼 메인반찬이 모두 있는지 자동 검사, q11 `answer_not=thread_id`, 리포트 미리보기 600자. q01·q11·q26·q27 4/4 통과 |
| 평가 3회차 | `evaluation/round3_report.md`: **28/28 통과**, 식단 12/12 check_plan 만족(+ 21끼 메뉴가 답변에 모두 있음), 근거 없는 숫자 0, 기피 누출 0, 평균 9.4초·최대 23.5초 (1회차 23/26·11.0초 → 2회차 26/28·9.2초 → 3회차 28/28·9.4초) |
| T15 | 완료 · 사용자 선택: 데이터 보강만(가중치 변경 X). menus.csv 끝에 가족 누구의 기피에도 안 걸리는 국 6(SP33~38)·메인 6(MS33~38) 추가 → 197개, family.json 주식 선호 추가(아빠 밥 2, 엄마 현미밥 2). 둘째 먹을 수 있는 국 23→29, 메인 21→27. 벡터 DB 201 문서로 자동 재생성, 최저 비용 286,000→281,700원. 기본·30만·W44 식단 check_plan 통과. **효과 한계**: 대체 메뉴 42→41회(새 공용 메뉴 1~4회만 사용) — 대체가 선호 점수를 더 주기 때문이라 목적 함수 가중치(대체 감점)로만 개선 가능, "최대한 싸게"도 여전히 효과 없음(30만 보통=싸게=300,000원). 데이터 생성 스크립트에는 이번 추가분이 없으므로 menus.csv 는 직접 관리 |
| T16 | 완료 · `README.md` 신설(폴더 구성, 설치, .env 키 설명, `uvicorn src.agent:app`, /query 요청·thread 이어가기·확정 예/아니오 흐름, 평가 명령과 회차 결과). 정리: 초기 16문항 리포트 `evaluation/report.md` 삭제(round1~3 이 대체), 구조 참고 자료 `reference/` 폴더 삭제(위 근거 문서 줄의 `reference/final_scenario.py` 는 기록으로만 남음). `run_eval.py` 기본 출력은 여전히 report.md 이므로 회차 리포트는 `--out` 으로 지정 |
| ragas 평가 | `evaluation/ragas_eval.py` 신설(run_eval.run() 재사용, Bedrock 채점, 지표 faithfulness·answer_relevancy·context_precision@10·context_recall·factual_correctness), test_queries.csv 에 `reference` 열 추가(T15 데이터 기준 도구로 계산한 총액·완화 금액 등), 결과 `evaluation/ragas_report.md`·`.csv`. 1회차: 규칙 검사 27/28(q16 실패: 3주 중복 1 > 0 — plan_week 에 '중복 최소화' 옵션이 없어 기본 식단과 같은 결과, T15 전에는 우연히 0), ragas 평균 faithfulness 0.38(qa 0.84, plan 0.03), answer_relevancy 0.30, context_precision 0.52, context_recall 0.13(plan 0.00), factual_correctness 0.57(plan 0.11). 한계: 식단의 끼니 배치·총액은 contexts 가 아니라 trace 에만 있어 plan 문항 faithfulness·recall 이 구조적으로 0 근처, 긴 식단 답변은 채점 출력 4096 토큰을 넘어 faithfulness 파싱 실패(18건, '-'), answer_relevancy 는 영어 채점 프롬프트로 한국어 질문을 비교해 전반적으로 낮음 |
| ragas 2회차 | 개선 4건: (1) `plan_week` 에 `min_history_overlap`(최근 3주와 겹친 끼니 1개당 100점 감점, 켰을 때만 plan_id 인자에 포함해 기존 id 유지) + 프롬프트 규칙 → q16 겹침 1→0, 규칙 검사 **28/28**, (2) 채점용 contexts 에 도구 결과 요약(식단 배치·총액·완화안·지난 식단·주간 요약·저장 결과)을 붙임(`--contexts agent+tools`, 제출 contexts 는 그대로), (3) 채점 출력 한도 4096→16000 → 파싱 실패 18→0건, (4) 채점 프롬프트 한국어 변환(`--lang korean`). factual_correctness 는 recall 모드로 변경(정답은 요약이라 F1 은 자세한 답을 깎음). 결과 `evaluation/ragas_report.md`: faithfulness 0.38→**0.84**(plan 0.03→0.98), context_precision 0.52→0.81, context_recall 0.13→0.73, factual_correctness 0.57→0.82, answer_relevancy 0.30→0.28(변화 없음: Titan 임베딩의 한국어 바꿔 말하기 유사도가 0.44~0.79에 그침, Cohere 다국어 임베딩은 계정 권한 없음). 남은 낮은 점수: 확정·거절 짧은 답의 answer_relevancy, q03·q04 recall 0(도구 요약에 '최대한 싸게'·'매운맛 제외' 조건이 안 들어감), q14·q15 factual 0(정답 예시와 답변 예시가 다름) |
| ragas 3회차 | 캐시(2회차 에이전트 응답)로 재채점. 채점용 contexts 에 요청 조건 줄(`[plan_conditions]`: 최대한 싸게·이번 요청 추가 기피·출석 변경·겹침 최소화) 추가 → q03·q04 context_recall 0→1.00, q14·q15 정답을 조건 위주로 수정, 캐시 재사용 시 CSV 최신 정답·`--only` 반영. 결과: 규칙 28/28, faithfulness 0.84, answer_relevancy 0.28, context_precision 0.81→**0.95**, context_recall 0.73→**0.81**(plan 0.98), factual_correctness 0.81. q14·q15 factual 0 은 ragas recall 정의(tp=답변 주장 중 정답에 있는 것) 때문 — 직접 검증 시 q14 정답 주장 5/5 를 답변이 담음. 리포트에 한계 문구 추가. 중간에 Bedrock 일일 토큰 한도 → 사용자가 .env region 변경 후 재개 |
| 평가 4회차 | `evaluation/round4_report.md`: **28/28 통과**, 식단 12/12 check_plan 만족, 근거 없는 숫자 0, 기피 누출 0, 평균 9.3초·최대 20.7초. T15 데이터·`min_history_overlap` 반영 후 첫 전체 평가. q16 3주 중복 0/10, 식단 총액이 모두 정답(reference) 값과 일치 |
| 제출 규약 정리 | `reference/산출물규약.txt` 기준: (1) API 응답 contexts=[{doc_id,text}], trace=[{step,input,output}] (`agent.py` to_api_contexts/to_api_trace, 내부 trace 는 그대로), 평가 스크립트 형식 검사 추가, (2) test_queries.csv 규약 7개 컬럼 + 자동 검사 열, id 1~30 숫자(note 에 이전 qNN), category positive 12·negative 6·edge 7·guardrail 5, 가드레일 2건 추가(29 프롬프트 인젝션, 30 관리자 사칭 기피 조회) + 프롬프트 [보안]·기피 조회 규칙, (3) SERVICE.md 템플릿 5개 절로 재배치(상차림 규칙 → 3절, 정책 5건, 성공 기준 90%), (4) README.md 템플릿 구조로 재작성(Day 번호는 비워 둠), (5) run.sh 신설, ragas_eval `--rule-out`. 최종 평가 `round5_report.md` **30/30**(category 전부 통과, 평균 8.8초), `ragas_report.md` faithfulness 0.87·answer_relevancy 0.29·context_precision 0.92·context_recall 0.81 |

## 4. 진행 순서와 의존 관계

```
T1 ─ T2 ─ T3 ─ T4 ─ T5 ─┬─ T6 ─┐
                        ├─ T7 ─┼─ T8 ─ T9 ─ T10
                        │      │         │
                        └──────┘   T11 ──┘ (T9 이전 아무 때나 가능)
```
