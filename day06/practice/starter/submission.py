"""Day 6 — Supervisor 인시던트 대응팀.

전문 Agent 셋을 Supervisor가 지휘합니다.

[중요]
  1. 함수·상수 이름을 바꾸지 마세요.
  2. `route_question`은 LLM을 호출하지 않습니다. 결정적 규칙으로 배분하세요.
     실제 운영에서는 모델이 배분하지만, 규칙으로 먼저 나눠 보면
     '무엇을 근거로 나누는가'가 분명해집니다.

[설계 요약]
  - 워커 Agent 3종(log / metric / runbook)은 실습 코드(specialist_agents.py)처럼
    `create_agent(llm, tools, system_prompt=..., name=...)`로 만듭니다.
  - Supervisor는 `langgraph_supervisor.create_supervisor` 대신 `StateGraph`로 직접 조립합니다.
    배분을 LLM이 아니라 `route_question`(결정적 규칙)이 맡아야
    "어느 Agent로 갔는가"가 언제 물어도 같은 답이 되기 때문입니다.
  - 워커 산출물은 Supervisor 한 곳에서 `judge_output` 게이트를 거친 뒤에만 메시지로 남습니다.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
REGION = "us-east-1"


def _default_llm():
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(model=MODEL_ID, region_name=REGION, temperature=0)


# ══════════════════════════════════════════════════════════════════
# 기초 파트 (7점)
# ══════════════════════════════════════════════════════════════════

# 전문 Agent 이름 3개. 이름이 곧 담당 범위의 선언이며, 그래프 노드 이름으로도 그대로 쓰입니다.
AGENT_NAMES: list[str] = ["log_agent", "metric_agent", "runbook_agent"]

# Agent별 도구. 도메인 기준으로 나누고, 같은 도구가 두 Agent에 들어가지 않게 합니다.
#   - log_agent     : 애플리케이션 로그와 과거 장애 이력(로그성 기록)
#   - metric_agent  : 수치 지표와 배포 이력(지표 변화의 원인 후보)
#   - runbook_agent : 대응 절차 문서와 서비스 담당자
AGENT_TOOLS: dict[str, list[str]] = {
    "log_agent": ["get_logs", "query_recent_incidents"],
    "metric_agent": ["get_metrics", "get_deploy_history"],
    "runbook_agent": ["search_runbook", "get_service_owner"],
}


# ──────────────────────────────────────────────────────────────────
# 워커가 쓸 도구 (교육용 더미 데이터입니다)
#   도구 이름과 인자는 Day3·Day4에서 만든 도구와 똑같이 맞춰 두었습니다.
#   나중에 실제 도구(MCP 등)로 바꿔도 이름이 같으므로 AGENT_TOOLS는 그대로 유지됩니다.
# ──────────────────────────────────────────────────────────────────

@tool
def get_logs(service: str, since: str = "") -> str:
    """서비스의 최근 애플리케이션 로그(에러·예외·스택트레이스)를 조회한다."""
    # 교육용 더미 데이터입니다.
    data = {
        "payment-api": (
            "[payment-api 로그] 최근 15분: NullPointerException 1243건, "
            "HTTP 503 412건 (출처: payment-api/app.log)"
        ),
        "order-api": "[order-api 로그] 최근 15분: TimeoutException 37건 (출처: order-api/app.log)",
    }
    return data.get(service, f"'{service}' 서비스의 로그를 찾을 수 없습니다.")


@tool
def query_recent_incidents(service: str) -> str:
    """서비스의 최근 장애 이력(과거 인시던트 기록)을 조회한다."""
    # 교육용 더미 데이터입니다.
    data = {
        "payment-api": "payment-api 최근 장애 2건: INC-1021(DB 커넥션 고갈), INC-0987(인증서 만료)",
    }
    return data.get(service, f"'{service}' 서비스의 최근 장애 이력이 없습니다.")


@tool
def get_metrics(service: str) -> str:
    """서비스의 현재 지표(5분 에러율, p99 지연, CPU, 메모리)를 조회한다."""
    # 교육용 더미 데이터입니다.
    data = {
        "payment-api": "payment-api 지표: 5분 에러율 7.8%, p99 지연 2350ms, CPU 82%, 메모리 71%",
        "order-api": "order-api 지표: 5분 에러율 0.4%, p99 지연 310ms, CPU 35%, 메모리 48%",
    }
    return data.get(service, f"'{service}' 서비스의 지표를 찾을 수 없습니다.")


@tool
def get_deploy_history(service: str) -> str:
    """서비스의 최근 배포 이력(버전, 배포 시각)을 조회한다."""
    # 교육용 더미 데이터입니다.
    data = {
        "payment-api": "payment-api 배포: v2.14.0 (오늘 09:40), 직전 v2.13.2 (3일 전)",
    }
    return data.get(service, f"'{service}' 서비스의 배포 이력이 없습니다.")


@tool
def search_runbook(symptom: str) -> str:
    """증상 키워드로 장애 대응 런북(절차)을 검색한다."""
    # 교육용 더미 데이터입니다.
    data = {
        "인증서": "[런북 RB-07 인증서 만료] 1) 만료일 확인 2) 보안팀에 재발급 요청 3) 배포 후 헬스체크",
        "5xx": "[런북 RB-02 5xx 급증] 1) 최근 배포 확인 2) 에러율 5% 초과 시 롤백 3) 10분 내 상황 공유",
        "롤백": "[런북 RB-03 롤백] 1) 직전 안정 버전 확인 2) 배포 도구로 롤백 3) 지표 정상화 확인",
    }
    for key, value in data.items():
        if key in symptom:
            return value
    return f"'{symptom}'에 해당하는 런북이 없습니다. 등록 키워드: {', '.join(data)}"


@tool
def get_service_owner(service: str) -> str:
    """서비스 담당자와 온콜 연락처를 조회한다."""
    # 교육용 더미 데이터입니다.
    data = {
        "payment-api": "payment-api 담당: 결제플랫폼팀 / 1차 온콜 이서준 / 2차 온콜 김하늘",
    }
    return data.get(service, f"'{service}' 서비스의 담당자를 찾을 수 없습니다.")


# 이름 → 도구 객체. AGENT_TOOLS의 이름 목록으로 실제 도구를 찾아 Agent에 붙입니다.
_TOOL_REGISTRY = {
    t.name: t
    for t in (
        get_logs,
        query_recent_incidents,
        get_metrics,
        get_deploy_history,
        search_runbook,
        get_service_owner,
    )
}


# ──────────────────────────────────────────────────────────────────
# 라우팅 규칙
# ──────────────────────────────────────────────────────────────────
# 판정은 소문자로 바꾼 질문에 대해 합니다.
#   - 한국어 키워드는 부분 문자열로 찾습니다. (조사가 붙어도 걸리도록)
#   - 짧은 영어 단어는 단어 경계(\b)로 찾습니다. 예: `log`가 `login`, `blog`에 걸리지 않도록.

_LOG_KEYWORDS = [
    "로그", "에러", "예외", "스택", "트레이스", "익셉션", "오류 메시지", "오류 로그",
    "오류가 찍", "에러 메시지", "덤프", "크래시", "널포인터", "타임아웃",
]
_LOG_PATTERNS = [
    r"\blogs?\b", r"\blogging\b", r"\bexceptions?\b", r"\berrors?\b",
    r"\bstack\s*traces?\b", r"\bstacks?\b", r"\btraces?\b", r"\btraceback\b",
    r"\bstderr\b", r"\bcrash(es)?\b", r"exception", r"\bnpe\b",
    # HTTP 5xx: 5xx, 500, 502, 503, 504 ... (1500ms 같은 숫자에는 걸리지 않게 앞뒤 숫자 배제)
    r"\b5xx\b", r"(?<!\d)5\d\d(?!\d)",
]

_METRIC_KEYWORDS = [
    "지표", "수치", "지연", "레이턴시", "응답시간", "응답 시간", "응답속도", "응답 속도",
    "메모리", "오류율", "에러율", "실패율", "배포", "대시보드", "사용률", "사용량",
    "처리량", "트래픽", "부하", "모니터링", "그래프", "퍼센트", "디스크",
]
_METRIC_PATTERNS = [
    r"\bmetrics?\b", r"\blatency\b", r"\bcpu\b", r"\bmemory\b", r"\bram\b",
    r"\bp9[59]\b", r"\bp50\b", r"\bthroughput\b", r"\bqps\b", r"\btps\b", r"\brps\b",
    r"\bdeploy(s|ment|ments|ed)?\b", r"\brelease\b", r"\bdashboards?\b",
    r"\berror\s*rate\b", r"\butilization\b", r"\busage\b", r"\bslo\b", r"\bsli\b",
    r"\bresponse\s*time\b", r"\bgrafana\b", r"\bprometheus\b",
]

_RUNBOOK_KEYWORDS = [
    "절차", "대응", "방법", "런북", "매뉴얼", "메뉴얼", "담당", "누구", "어떻게",
    "가이드", "조치", "연락", "온콜", "플레이북", "지침", "순서", "해결책", "복구 방안",
    "책임자", "에스컬레이션",
]
_RUNBOOK_PATTERNS = [
    r"\brunbooks?\b", r"\bplaybooks?\b", r"\bprocedures?\b", r"\bmanuals?\b",
    r"\bhow\s+(to|do|can|should)\b", r"\bowners?\b", r"\bon-?call\b", r"\bwho\b",
    r"\bsop\b", r"\bguide(line)?s?\b", r"\bescalat(e|ion)\b", r"\bplaybook\b",
]

# "에러율·오류율·error rate"는 지표입니다. 안에 든 "에러"·"error"가 log 규칙에 걸리지 않도록
# log 판정 전에 이 표현들을 지웁니다. (공개 케이스 "오류율 지표" → metric_agent만)
#
# re.ASCII: 기본 설정에서는 한글도 단어 문자(\w)라서 "5xx가", "log를"처럼 조사가 붙으면
# \b 경계가 생기지 않습니다. ASCII 모드로 두면 한글이 경계로 인식되어 영문 키워드가 잡힙니다.
_RATE_TERMS = re.compile(r"에러\s*율|오류\s*율|실패\s*율|\berror\s*rates?\b", re.ASCII)

_RULES: dict[str, tuple[list[str], list[str]]] = {
    "log_agent": (_LOG_KEYWORDS, _LOG_PATTERNS),
    "metric_agent": (_METRIC_KEYWORDS, _METRIC_PATTERNS),
    "runbook_agent": (_RUNBOOK_KEYWORDS, _RUNBOOK_PATTERNS),
}
_COMPILED_RULES = {
    name: (kws, [re.compile(p, re.ASCII) for p in pats]) for name, (kws, pats) in _RULES.items()
}


def _matches(text: str, agent: str) -> bool:
    keywords, patterns = _COMPILED_RULES[agent]
    if any(k in text for k in keywords):
        return True
    return any(p.search(text) for p in patterns)


def route_question(question: str) -> list[str]:
    """질문을 처리할 Agent 목록을 반환합니다. LLM을 호출하지 않습니다.

    반환 규칙:
      - 로그·에러·예외·스택·트레이스·5xx, 영문 log/exception/trace -> log_agent
      - 지표·수치·지연·레이턴시·CPU·메모리·오류율·배포, 영문 metric/latency -> metric_agent
      - 절차·대응·방법·런북·매뉴얼·담당·누구·어떻게, 영문 runbook -> runbook_agent
      - 여러 주제가 섞이면 해당하는 Agent를 모두 반환
      - 어디에도 해당하지 않으면 빈 목록 (Supervisor가 직접 답한다)

    **필요한 것만 반환하세요.** 전부 부르면 정확도는 오르지만 호출 비용이 폭증합니다.
    """
    # 지저분한 입력(None, 숫자, 공백만 있는 문자열)에도 예외 없이 빈 목록을 돌려줍니다.
    if not isinstance(question, str):
        return []
    text = question.strip().lower()
    if not text:
        return []

    targets: list[str] = []
    for agent in AGENT_NAMES:
        # log 판정에서만 "에러율" 류 표현을 지운 텍스트를 씁니다.
        probe = _RATE_TERMS.sub(" ", text) if agent == "log_agent" else text
        if _matches(probe, agent):
            targets.append(agent)  # AGENT_NAMES 순서대로 한 번씩만 → 중복이 생기지 않음
    return targets


# ──────────────────────────────────────────────────────────────────
# Supervisor 그래프
# ──────────────────────────────────────────────────────────────────

class SupervisorState(TypedDict, total=False):
    """Supervisor 그래프의 상태.

    - messages : 대화 기록 (add_messages로 누적)
    - targets  : route_question이 고른 Agent 목록
    - done     : 실행을 마친 Agent 목록
    - routed   : 배분을 이미 했는지 여부 (targets가 []로 들어와도 첫 진입이면 배분)
    """

    messages: Annotated[list[BaseMessage], add_messages]
    targets: list[str]
    done: list[str]
    routed: bool


_WORKER_PROMPTS = {
    "log_agent": (
        "너는 인시던트 대응팀의 로그 분석 전문가다. 애플리케이션 로그, 에러, 예외, 스택트레이스,\n"
        "과거 장애 이력 확인만 담당한다. 반드시 도구로 조회한 결과에 근거해 답하고,\n"
        "건수 같은 수치와 출처(로그 파일)를 함께 적어 한 문단으로 보고하라.\n"
        "담당 범위 밖 질문에는 '담당 범위가 아닙니다'라고만 답하라."
    ),
    "metric_agent": (
        "너는 인시던트 대응팀의 지표 분석 전문가다. 에러율, 지연(p99), CPU, 메모리 같은 지표와\n"
        "배포 이력 확인만 담당한다. 반드시 도구로 조회한 수치에 근거해 답하고,\n"
        "추측으로 원인을 단정하지 말고 관측된 수치만 한 문단으로 보고하라.\n"
        "담당 범위 밖 질문에는 '담당 범위가 아닙니다'라고만 답하라."
    ),
    "runbook_agent": (
        "너는 인시던트 대응팀의 런북 전문가다. 장애 대응 절차와 서비스 담당자 확인만 담당한다.\n"
        "반드시 search_runbook, get_service_owner 결과에 근거해 답하고 런북 번호를 인용하라.\n"
        "담당 범위 밖 질문에는 '담당 범위가 아닙니다'라고만 답하라."
    ),
}

_SUPERVISOR_PROMPT = (
    "너는 인시던트 대응팀의 Supervisor다. 이 질문은 로그·지표·런북 어느 전문 Agent의\n"
    "담당에도 해당하지 않는다. 장애 대응 범위 밖이면 짧게 정중히 안내하고,\n"
    "로그·지표·대응 절차 조회를 도와줄 수 있다고 알려라."
)


def _get_text(message: Any) -> str:
    """ChatBedrockConverse는 content를 블록 리스트로 주기도 하므로 텍스트만 모아 반환합니다."""
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return str(content or "")


def _last_user_question(messages: list[BaseMessage]) -> str:
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage) or getattr(m, "type", "") == "human":
            return _get_text(m)
    return ""


def build_supervisor(llm=None):
    """Supervisor 그래프를 만들어 컴파일해 반환합니다.

    반드시 지킬 것:
      - AGENT_NAMES의 이름이 모두 그래프 노드로 존재해야 합니다.
      - supervisor 노드에서 각 Agent로 가는 경로가 있어야 합니다.
      - 종료 경로가 있어야 합니다.

    langgraph_supervisor를 써도 되고 StateGraph로 직접 만들어도 됩니다.

    그래프 모양:
        START → supervisor ─┬→ log_agent ─────┐
                            ├→ metric_agent ──┼→ supervisor (남은 대상이 있으면 다시 배분)
                            ├→ runbook_agent ─┘
                            └→ END
    """
    # 모델은 반드시 함수 안에서 만듭니다. 채점기는 llm 인자로 가짜 모델을 꽂습니다.
    llm = llm or _default_llm()

    from langchain.agents import create_agent

    # ── 워커 Agent 3종: 담당 도구만 붙여 좁게 만듭니다 ──
    workers = {
        name: create_agent(
            llm,
            [_TOOL_REGISTRY[t] for t in AGENT_TOOLS[name]],
            system_prompt=_WORKER_PROMPTS[name],
            name=name,
        )
        for name in AGENT_NAMES
    }

    # ── supervisor 노드: 결정적 규칙으로 배분, 해당 없으면 직접 답변 ──
    def supervisor(state: SupervisorState) -> dict:
        if state.get("routed"):
            return {}  # 이미 배분했으면 다음 대상 선택은 조건부 엣지가 맡습니다.

        question = _last_user_question(state.get("messages", []))
        targets = route_question(question)
        update: dict = {"targets": targets, "done": [], "routed": True}
        if not targets:
            # 어느 Agent에도 해당하지 않으면 Supervisor가 직접 답합니다. (워커 호출 0회)
            reply = llm.invoke(
                [SystemMessage(_SUPERVISOR_PROMPT), HumanMessage(question or "(빈 질문)")]
            )
            update["messages"] = [
                AIMessage(content=f"[supervisor] {_get_text(reply)}", name="supervisor")
            ]
        return update

    def next_step(state: SupervisorState) -> str:
        done = set(state.get("done") or [])
        for target in state.get("targets") or []:
            if target not in done:
                return target
        return END

    # ── 워커 노드: 서브 Agent 실행 → judge_output 게이트 → 메시지 기록 ──
    def make_worker_node(name: str):
        agent = workers[name]

        def worker_node(state: SupervisorState) -> dict:
            question = _last_user_question(state.get("messages", []))
            result = agent.invoke({"messages": [HumanMessage(question)]})
            output = _get_text(result["messages"][-1]).strip()

            verdict = judge_output(name, output)
            if verdict["keep"]:
                content = f"[{name}] {output}"
            else:
                # 억제해도 사유는 남깁니다. 게이트가 과하게 잠긴 건지 워커가 부실한 건지
                # 나중에 구분할 수 있어야 하기 때문입니다.
                content = f"[{name}] 산출물 억제됨 (사유: {verdict['reason']})"

            return {
                "messages": [AIMessage(content=content, name=name)],
                "done": list(state.get("done") or []) + [name],
            }

        return worker_node

    builder = StateGraph(SupervisorState)
    builder.add_node("supervisor", supervisor)
    for name in AGENT_NAMES:
        builder.add_node(name, make_worker_node(name))
        builder.add_edge(name, "supervisor")  # 워커가 끝나면 Supervisor로 복귀

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        next_step,
        {**{name: name for name in AGENT_NAMES}, END: END},
    )
    return builder.compile()


# ══════════════════════════════════════════════════════════════════
# 심화 파트 (3점) — 못 해도 과제는 통과합니다
# ══════════════════════════════════════════════════════════════════

# 근거 없이 단정하는 표현
_ASSERTION = re.compile(
    r"확실|틀림없|분명(히|합니다)|단언|의심의 여지|100\s*%|백퍼|무조건|"
    r"\bdefinitely\b|\bcertainly\b|\bguaranteed\b|\bundoubtedly\b|\bwithout (a )?doubt\b",
    re.IGNORECASE,
)
# 근거로 인정하는 출처 표시
_SOURCE = re.compile(
    r"출처|근거:|참고:|인용|\.log\b|https?://|\[source|source:|RB-\d+|INC-\d+|런북 ",
    re.IGNORECASE,
)


def judge_output(agent_name: str, output: str) -> dict:
    """워커 산출물을 내보내기 전에 걸러냅니다. LLM을 호출하지 않습니다.

    워커가 만든 것을 그대로 사용자에게 주면 근거 없는 단정과 빈 답변이 섞입니다.
    Supervisor는 게이트 역할을 해야 합니다.

    판정 규칙:
      - 내용이 비었거나 20자 미만          -> {"keep": False, "reason": "저가치"}
      - "확실합니다" "틀림없습니다" 같은
        근거 없는 단정이 있는데 수치·출처가 없음 -> {"keep": False, "reason": "근거 없는 단정"}
      - 그 외                              -> {"keep": True, "reason": ...}

    Returns:
        {"keep": bool, "reason": str} 형태의 dict
    """
    text = (output if isinstance(output, str) else str(output or "")).strip()

    # 1) 빈 값·짧은 값
    if len(text) < 20:
        return {"keep": False, "reason": "저가치"}

    # 2) 단정 표현이 있는데 근거(수치·출처)가 없음
    if _ASSERTION.search(text):
        # "100%" 같은 단정 표현 속 숫자는 근거로 치지 않도록 지운 뒤 숫자를 찾습니다.
        stripped = _ASSERTION.sub(" ", text)
        has_number = bool(re.search(r"\d", stripped))
        has_source = bool(_SOURCE.search(text))
        if not (has_number or has_source):
            return {"keep": False, "reason": "근거 없는 단정"}
        return {"keep": True, "reason": f"통과: {agent_name} 단정 표현에 수치·출처 근거 있음"}

    # 3) 그 외
    return {"keep": True, "reason": f"통과: {agent_name} 산출물 길이·근거 기준 충족"}
