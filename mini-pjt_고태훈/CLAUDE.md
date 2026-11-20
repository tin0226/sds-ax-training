# 프로젝트 규칙

## 기술 스택
- Python, LangChain, LangGraph
- 모델은 Amazon Bedrock (ChatBedrockConverse)
- Agent 생성은 langchain.agents 의 create_agent 를 쓴다
- 식단 최적화는 PuLP 로 모델을 만들고 HiGHS 솔버(highspy)로 푼다. PuLP 내장 CBC 2.10.3 은 이 모델에서 멈추고 시간 제한을 무시해 쓰지 않는다

## 폴더 구조 (제출 규약. 바꾸지 않는다)
src/agent.py       메인 에이전트 그래프 (POST /query FastAPI app 도 여기에 둔다)
src/tools.py       도메인 도구
src/retriever.py   RAG 파이프라인
data/              사용한 문서와 데이터
evaluation/        test_queries.csv 와 평가 리포트

## 주고받는 형식 (제출 규약)
- POST /query 로 받고 question 필드를 읽는다 (대화를 이어 갈 때만 선택 필드 thread_id)
- 답은 answer, contexts, trace 세 키로 돌려준다. contexts 는 [{doc_id, text}], trace 는 [{step(도구 이름), input, output}]
- evaluation/test_queries.csv 는 규약 7개 컬럼(id, category, input, expected_traits, forbidden, expected_tools, note)을 앞에 두고, 자동 검사용 열(thread_id, expect_type, checks, expect, reference)을 뒤에 둔다

## 도메인 규칙
- 설계 문서: ~/.gstack/projects/mini-pjt/SDS-master-design-20261118-161221.md (구현 전에 읽는다)
- 한 끼 = 주식 1 + 국 1 + 메인반찬 1 + 서로 다른 부반찬 3. 모든 메뉴는 완제품이다
- 가격·칼로리 합계, 기피 검사, 식단 생성은 tools.py 의 도구가 한다. LLM 이 숫자를 계산하지 않는다
- 기피는 절대 조건이다. 주식·국·메인반찬은 기피자에게만 대체 메뉴, 부반찬은 대체 없이 최소 1개는 먹을 수 있게 한다
- 최근 3주와의 중복은 (국, 메인반찬) 조합 기준으로 센다
- 식단 확정(data/history 저장)은 사용자가 확정을 요청할 때만 한다. save_plan 은 HumanInTheLoopMiddleware 로 실행 전에 멈추고, 같은 thread 에서 사용자가 "예"로 승인해야 저장된다 (SAVE_ALLOWED 는 승인 재개 요청에서만 True)
- data/family.json 은 읽기 전용이다. 가족 기본 정보를 바꾸는 도구를 만들지 않고, 에이전트는 변경 요청을 "가족 기본 정보(일정·선호·기피·칼로리)는 변경할 수 없습니다."로 거절한다
- 구성원별 기피 항목은 건강 정보일 수 있다. 계산에만 쓰고, 출력(도구 결과·answer·contexts·trace·벡터 DB)에는 "누가 무엇을 기피하는지"를 쓰지 않는다 ("둘째는 기피로 대체", "기피: 비공개"). 사용자가 이번 질문에서 직접 말한 추가 기피는 예외

## 코드 규칙
- 파일 하나에 한 가지 역할만 둔다
- 함수와 도구에는 한국어 docstring 을 쓴다
- 비밀 값은 .env 에서 읽고 코드에 적지 않는다

## 하지 말 것
- 요청하지 않은 파일을 새로 만들지 않는다
- 기존 파일을 통째로 다시 쓰지 않는다. 바뀐 부분만 고친다
