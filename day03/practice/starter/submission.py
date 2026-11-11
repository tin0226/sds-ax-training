"""Day 3 — 장애 원인 조사 Agent.

도구 세 개로 장애 원인을 좁혀 갑니다.

[중요]
  1. 함수·상수 이름을 바꾸지 마세요.
  2. 모델은 함수 안에서 만듭니다. `llm` 인자가 오면 그것을 씁니다.
  3. 도구는 결정적이어야 합니다. 같은 인자에 항상 같은 결과를 돌려주세요.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
REGION = "us-east-1"

# 교육용 더미 데이터입니다. 실제 운영 데이터와 무관합니다.
DATA = json.loads(
    (Path(__file__).resolve().parent.parent / "data" / "ops_fixtures.json").read_text(
        encoding="utf-8"
    )
)


def _default_llm():
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(model=MODEL_ID, region_name=REGION, temperature=0)


# ══════════════════════════════════════════════════════════════════
# 기초 파트 (7점)
# ══════════════════════════════════════════════════════════════════


def _check_service(service: str, table: dict) -> Optional[str]:
    """서비스 이름이 비었거나 데이터에 없으면 안내 문자열을, 정상이면 None을 돌려줍니다.

    모델이 다음 판단을 할 수 있도록 '조회 가능한 서비스 목록'을 함께 알려 줍니다.
    """
    name = (service or "").strip()
    if not name:
        return (
            "서비스 이름이 비어 있습니다. 조회할 서비스 이름을 넣어 주세요. "
            f"조회 가능한 서비스: {sorted(table)}"
        )
    if name not in table:
        return (
            f"'{name}' 서비스를 찾을 수 없습니다(데이터 없음). "
            f"조회 가능한 서비스: {sorted(table)}"
        )
    return None


# ── 도구 1: 배포 이력 (연쇄의 출발점) ───────────────────────────────
class DeployArgs(BaseModel):
    service: str = Field(
        ...,
        description="배포 이력을 조회할 서비스 이름. 예: 'payment-api', 'auth-service'",
    )


@tool(args_schema=DeployArgs)
def get_deploy_history(service: str) -> str:
    """서비스의 최근 배포 이력(버전, 배포 시각, 배포 주체)을 최신순으로 조회합니다.

    장애 원인 조사를 시작할 때 가장 먼저 쓰세요. 여기서 얻은 배포 시각(ISO 형식)을
    get_logs의 since 인자로 넘겨야 그 시점의 로그를 볼 수 있습니다.
    로그 내용이나 에러율·지연 같은 지표를 알고 싶을 때는 쓰지 마세요.
    """
    err = _check_service(service, DATA["deploys"])
    if err:
        return err
    name = service.strip()
    # 시각 기준 최신순 정렬. 같은 입력이면 항상 같은 순서가 나옵니다.
    deploys = sorted(DATA["deploys"][name], key=lambda d: d["at"], reverse=True)
    lines = [f"[{name}] 배포 이력 {len(deploys)}건 (최신순)"]
    for d in deploys:
        lines.append(f"- {d['version']} | 배포 시각 {d['at']} | 배포 주체 {d['by']}")
    lines.append(
        f"가장 최근 배포 시각은 {deploys[0]['at']} 입니다. "
        "이 값을 get_logs의 since로 넘기면 해당 시점 로그를 볼 수 있습니다."
    )
    return "\n".join(lines)


# ── 도구 2: 로그 (배포 시각을 인자로 받는 연쇄의 두 번째 고리) ─────────
class LogsArgs(BaseModel):
    service: str = Field(
        ...,
        description="로그를 조회할 서비스 이름. 예: 'payment-api'",
    )
    since: Optional[str] = Field(
        default=None,
        description=(
            "로그를 볼 시점. get_deploy_history가 알려 준 배포 시각을 ISO 형식 그대로 넣으세요. "
            "예: '2026-08-11T02:58:00+09:00'. 비우면 평상시(default) 로그를 봅니다."
        ),
    )


@tool(args_schema=LogsArgs)
def get_logs(service: str, since: Optional[str] = None) -> str:
    """서비스의 애플리케이션 로그(ERROR/WARN/INFO)를 조회합니다.

    배포 직후 어떤 에러가 찍혔는지 확인할 때 쓰세요. since에는 get_deploy_history로
    얻은 배포 시각을 넣어야 그 시점 로그가 나옵니다. 배포 시각을 모르면 먼저
    get_deploy_history를 부르세요. 에러율·지연 같은 수치가 필요하면 get_metrics를 쓰세요.
    """
    services = {k.split("@", 1)[0] for k in DATA["logs"]}
    err = _check_service(service, dict.fromkeys(services))
    if err:
        return err
    name = service.strip()
    when = (since or "").strip()

    if when:
        key = f"{name}@{when}"
        if key in DATA["logs"]:
            return f"[{name}] {when} 시점 로그\n{DATA['logs'][key]}"
        # 해당 시점 로그가 없으면 평상시 로그로 대체하고, 그 사실을 분명히 적습니다.
        default = DATA["logs"].get(f"{name}@default", "(로그 없음)")
        return (
            f"[{name}] {when} 시점 로그가 없습니다. 평상시(default) 로그로 대신 보여 드립니다.\n"
            f"{default}\n"
            "배포 시각은 get_deploy_history 결과의 값을 그대로 넣었는지 확인하세요."
        )

    default = DATA["logs"].get(f"{name}@default", "(로그 없음)")
    return (
        f"[{name}] 평상시(default) 로그\n{default}\n"
        "특정 배포 직후 로그를 보려면 since에 배포 시각을 넣으세요."
    )


# ── 도구 3: 지표 ────────────────────────────────────────────────────
class MetricsArgs(BaseModel):
    service: str = Field(
        ...,
        description="지표를 조회할 서비스 이름. 예: 'payment-api'",
    )


@tool(args_schema=MetricsArgs)
def get_metrics(service: str) -> str:
    """서비스의 현재 운영 지표(5분 에러율, p99 지연, CPU 사용률)를 조회합니다.

    장애의 규모와 영향을 수치로 확인할 때 쓰세요. 원인이 된 에러 메시지는 나오지
    않으므로 에러 내용은 get_logs로, 배포 여부는 get_deploy_history로 확인하세요.
    """
    err = _check_service(service, DATA["metrics"])
    if err:
        return err
    name = service.strip()
    m = DATA["metrics"][name]
    return (
        f"[{name}] 현재 지표\n"
        f"- 5분 에러율: {m['error_rate_5m'] * 100:.1f}%\n"
        f"- p99 지연: {m['p99_latency_ms']}ms\n"
        f"- CPU 사용률: {m['cpu'] * 100:.0f}%"
    )


TOOLS: list = [get_deploy_history, get_logs, get_metrics]


# ── 그래프 ──────────────────────────────────────────────────────────

# 수동 루프의 messages 리스트 → State + add_messages reducer
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


SYSTEM_PROMPT = """너는 장애 원인 조사를 돕는 SRE 어시스턴트야.

[도구 사용 전략]
1. 장애 원인을 물으면 먼저 get_deploy_history로 최근 배포가 있었는지, 그 시각이 언제인지 확인해.
2. 배포 시각을 알게 되면 그 값을 그대로 get_logs의 since에 넣어 그 시점 로그를 확인해.
3. 영향 규모는 get_metrics로 수치를 확인해.
4. 도구 결과를 지어내지 마. 확인한 근거만으로 결론을 말해.
5. 도구가 필요 없는 일반 질문(용어 설명 등)에는 도구 없이 바로 답해."""


def build_agent(llm=None):
    """도구를 쓸 수 있는 Agent 그래프를 만들어 컴파일해 반환합니다.

    구조:
        START → agent ─(tool_calls 있음)→ tools ─(should_stop False)→ agent  ← 순환
                  └─(tool_calls 없음)→ END        └─(should_stop True)→ END   ← 탈출
    """
    llm = llm or _default_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    # 수동 루프의 invoke와 append → agent 노드
    def agent(state: AgentState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [llm_with_tools.invoke(messages)]}

    # 탈출 조건 1: 모델이 도구를 더 부르지 않으면 종료
    def route_after_agent(state: AgentState) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "end"

    # 탈출 조건 2: 결정적 판정(should_stop)이 멈추라고 하면 종료
    def route_after_tools(state: AgentState) -> Literal["agent", "end"]:
        messages = state["messages"]
        made = sum(1 for m in messages if isinstance(m, ToolMessage))
        stop, _reason = should_stop(messages, made)
        return "end" if stop else "agent"

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(TOOLS))  # 도구 실행과 ToolMessage 포장

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "end": END})
    builder.add_conditional_edges("tools", route_after_tools, {"agent": "agent", "end": END})

    return builder.compile()


# ══════════════════════════════════════════════════════════════════
# 심화 파트 (3점) — 못 해도 과제는 통과합니다
# ══════════════════════════════════════════════════════════════════

# 한 번의 조사에서 허용할 도구 호출 상한 (3 이상 8 이하)
MAX_TOOL_CALLS = 3

_REQUIRED_TOOLS = {"get_deploy_history", "get_logs", "get_metrics"}


def should_stop(messages: list, tool_calls_made: int) -> tuple[bool, str]:
    """더 조사할지 멈출지 결정합니다. LLM을 호출하지 않습니다.

    판정 규칙 (위에서부터 순서대로 검사):
      - tool_calls_made >= MAX_TOOL_CALLS        -> (True, "예산 초과")
      - 같은 도구를 같은 인자로 2회 이상 호출함  -> (True, "중복 호출")
      - 세 도구를 모두 한 번 이상 호출함         -> (True, "근거 충분")
      - 그 외                                    -> (False, "계속 조사")
    """
    if tool_calls_made >= MAX_TOOL_CALLS:
        return True, "예산 초과"

    # AIMessage에 남은 도구 호출 기록을 (이름, 정규화된 인자) 형태로 모읍니다.
    seen: set[tuple[str, str]] = set()
    names: set[str] = set()
    for m in messages:
        if not isinstance(m, AIMessage):
            continue
        for tc in m.tool_calls or []:
            name = tc.get("name", "")
            args: Any = tc.get("args", {})
            sig = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
            if sig in seen:
                return True, "중복 호출"
            seen.add(sig)
            names.add(name)

    if _REQUIRED_TOOLS <= names:
        return True, "근거 충분"

    return False, "계속 조사"
