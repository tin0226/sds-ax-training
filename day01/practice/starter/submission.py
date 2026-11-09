"""Day 1 — 운영 알람 트리아지 체인.

알람 하나를 넣으면 구조화된 분류 결과와 담당팀 안내 초안을 동시에 만듭니다.

[중요] 이 파일의 규칙
  1. 함수 이름과 인자를 바꾸지 마세요. 채점기가 이 이름으로 찾습니다.
  2. 모델은 반드시 함수 **안에서** 만듭니다. 모듈 최상단에서 만들면
     채점기가 가짜 모델을 넣을 수 없어 채점이 실패합니다.
  3. `llm` 인자가 들어오면 그것을 쓰고, None이면 직접 만듭니다.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Literal

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from pydantic import BaseModel, Field

# 과정 전체에서 고정으로 쓰는 값입니다. 바꾸지 마세요.
MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
REGION = "us-east-1"

# 프롬프트에 넣을 알람 본문의 최대 길이입니다.
# 4000자짜리 알람이 들어와도 토큰이 폭주하지 않도록 잘라서 넣습니다.
MAX_ALERT_CHARS = 2000


def _default_llm():
    """기본 모델을 만듭니다. 채점기는 이 함수를 부르지 않습니다.

    import를 함수 안에 둔 이유: 모듈을 불러오는 것만으로 AWS 자격증명을
    요구하지 않게 하기 위해서입니다.
    """
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(model=MODEL_ID, region_name=REGION, temperature=0)


def get_text(message: Any) -> str:
    """ChatBedrockConverse는 content를 블록 리스트로 주기도 하므로 텍스트만 모아 반환합니다."""
    content = message.content if hasattr(message, "content") else message
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return str(content)


def _serialize_alert(alert: Any) -> str:
    """알람을 프롬프트에 넣을 문자열로 바꿉니다.

    - dict가 아니어도(문자열, 숫자 등) 깨지지 않도록 처리합니다.
    - 중첩 dict, 한글, 이모지가 섞여도 json.dumps가 안전하게 직렬화합니다.
      (default=str 로 datetime 같은 비직렬화 객체도 문자열로 바꿉니다.)
    - 값 안의 중괄호는 템플릿 변수 '값'으로 들어가므로 템플릿 파싱에 영향이 없습니다.
    - 너무 긴 본문은 잘라서 넣습니다.
    """
    try:
        text = json.dumps(alert, ensure_ascii=False, default=str, indent=2)
    except (TypeError, ValueError):
        text = str(alert)
    if len(text) > MAX_ALERT_CHARS:
        text = text[:MAX_ALERT_CHARS] + "\n... (이하 생략)"
    return text


# ══════════════════════════════════════════════════════════════════
# 기초 파트 (7점)
# ══════════════════════════════════════════════════════════════════

class TriageResult(BaseModel):
    """운영 알람 분류 결과입니다. 티켓 시스템에 그대로 적재합니다."""

    severity: Literal["P1", "P2", "P3", "P4"] = Field(
        ...,
        description=(
            "긴급도 등급. 반드시 P1, P2, P3, P4 중 하나. "
            "P1: 결제·인증·주문 등 핵심 기능의 전면 장애, 또는 에러율(5xx 등)이 임계치를 "
            "크게 초과해 고객 거래가 실패하는 상황. 즉시 담당팀 호출. "
            "P2: 일부 기능 저하나 일부 리전/사용자에 한정된 장애, 또는 핵심 서비스의 이상 징후. "
            "30분 내 대응. "
            "P3: 성능 저하·리소스 경고 등 고객 영향이 아직 없거나 미미한 상황. 업무 시간 내 대응. "
            "P4: 정보성 알람, 자동 복구된 일시 현상, 판단 근거가 부족한 알람."
        ),
    )
    service: str = Field(
        ...,
        description="영향받는 서비스 이름. 알람의 service 값을 그대로 쓰고, 없으면 'unknown'.",
    )
    category: Literal["availability", "latency", "resource", "security", "other"] = Field(
        ...,
        description=(
            "증상 카테고리. availability: 에러·다운·5xx, latency: 응답 지연·타임아웃, "
            "resource: CPU·메모리·디스크 등 자원 고갈, security: 인증 실패 급증·이상 접근, "
            "other: 위에 해당하지 않음."
        ),
    )
    owner_team: str = Field(
        ...,
        description=(
            "호출할 담당팀. 서비스 이름에서 추정한다. 예) payment-* → 결제플랫폼팀, "
            "auth-* → 인증플랫폼팀, order-* → 주문팀, gateway → 인프라팀. 모르면 'SRE 당직'."
        ),
    )
    first_check: str = Field(
        ...,
        description=(
            "당직자가 가장 먼저 열어 볼 항목 한 가지. 예) 최근 30분 배포 이력, "
            "해당 서비스 에러 로그, 의존 DB 커넥션 대시보드."
        ),
    )
    summary: str = Field(
        ...,
        description="알람 상황을 한국어 한 문장(50자 이내)으로 요약.",
    )


TRIAGE_SCHEMA = TriageResult


def build_triage_chain(llm=None):
    """알람 dict를 받아 TRIAGE_SCHEMA 형태의 dict를 반환하는 체인을 만듭니다.

    순수한 `prompt | llm | parser` 형태로 둡니다.
    파싱 실패 복구는 build_full_chain에서 fallback으로 붙입니다.
    """
    llm = llm or _default_llm()

    parser = JsonOutputParser(pydantic_object=TRIAGE_SCHEMA)
    prompt = PromptTemplate(
        template=(
            "당신은 SDS 운영센터의 장애 트리아지 담당자입니다.\n"
            "아래 모니터링 알람을 읽고 긴급도, 증상 카테고리, 담당팀, 첫 확인 항목을 판단하세요.\n"
            "알람의 일부 필드가 비어 있거나 형식이 이상해도 주어진 정보만으로 최선의 판단을 내리세요.\n\n"
            "알람:\n{alert}\n\n"
            "{format_instructions}\n"
            "JSON 외의 설명 문장은 출력하지 마세요.\n"
        ),
        input_variables=["alert"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return {"alert": RunnableLambda(_serialize_alert)} | prompt | llm | parser


NOTICE_SYSTEM_PROMPT = (
    "당신은 SDS 운영센터 야간 당직 엔지니어입니다. 모니터링 알람을 보고 "
    "담당팀 Slack 채널에 그대로 붙여넣을 장애 안내 초안을 작성합니다.\n"
    "작성 규칙:\n"
    "- 존댓말(~합니다/~부탁드립니다)로, 5줄 이내로 씁니다.\n"
    "- 첫 줄은 '[장애 안내] <서비스명> <증상 한 줄>' 형식으로 씁니다.\n"
    "- 이어서 발생 시각, 리전, 현재 수치(임계치 대비)를 적습니다.\n"
    "- 마지막 줄에 담당팀에게 요청할 첫 조치(확인 항목)와 회신 요청을 적습니다.\n"
    "- 알람에 없는 정보는 추측해서 쓰지 말고 '확인 필요'라고 적습니다."
)


def build_notice_chain(llm=None):
    """알람 dict를 받아 담당팀 안내 초안 문자열을 반환하는 체인을 만듭니다."""
    llm = llm or _default_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", NOTICE_SYSTEM_PROMPT),
            ("human", "다음 알람에 대한 담당팀 안내 초안을 작성해 주세요.\n\n{alert}"),
        ]
    )
    return {"alert": RunnableLambda(_serialize_alert)} | prompt | llm | StrOutputParser()


def build_full_chain(llm=None):
    """두 체인을 병렬로 묶어 {"triage": dict, "notice": str} 를 반환합니다.

    triage 갈래 흐름:
      알람 → 분류 체인 (실패 시 _fallback_triage) → _normalize_triage → apply_severity_rules
    """
    llm = llm or _default_llm()

    # 모델이 JSON이 아닌 것을 뱉어 파서가 예외를 던지면 fallback이 입력(알람)을 받아 기본값을 만듭니다.
    safe_triage = build_triage_chain(llm).with_fallbacks(
        [RunnableLambda(_fallback_triage)]
    )

    def _triage_with_rules(alert: Any) -> dict:
        raw = safe_triage.invoke(alert)
        triage = _normalize_triage(raw, alert)
        return apply_severity_rules(triage, alert if isinstance(alert, dict) else {})

    return RunnableParallel(
        triage=RunnableLambda(_triage_with_rules),
        notice=build_notice_chain(llm),
    )


def _alert_get(alert: Any, key: str, default: str = "unknown") -> str:
    """알람에서 값을 꺼냅니다. 키가 없거나 dict가 아니어도 깨지지 않습니다."""
    if isinstance(alert, dict):
        value = alert.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return default


def _fallback_triage(alert: Any) -> dict:
    """파싱 실패 시 쓰는 복구 결과입니다.

    모델 판단을 믿을 수 없으므로 가장 높은 등급(P1)으로 두어 사람이 반드시 보게 하고,
    parse_failed 표시를 남겨 다음 날 복구 건수를 셀 수 있게 합니다.
    """
    return {
        "severity": "P1",
        "service": _alert_get(alert, "service"),
        "category": "other",
        "owner_team": "SRE 당직",
        "first_check": "모델 분류 실패 — 알람 원문을 직접 확인",
        "summary": "모델 응답 파싱 실패로 복구 경로가 생성한 결과입니다.",
        "parse_failed": True,
    }


def _normalize_triage(raw: Any, alert: Any) -> dict:
    """파서 결과의 모양을 고정합니다.

    - dict가 아니면 복구 결과로 바꿉니다.
    - 스키마 필드가 빠졌으면 기본값으로 채워, 같은 알람을 몇 번 처리해도 키 구성이 같게 합니다.
    - severity가 허용값이 아니면 복구 표시와 함께 P1로 둡니다.
    """
    if not isinstance(raw, dict):
        return _fallback_triage(alert)

    result = dict(raw)
    defaults = {
        "service": _alert_get(alert, "service"),
        "category": "other",
        "owner_team": "SRE 당직",
        "first_check": "확인 필요",
        "summary": "",
    }
    for key, value in defaults.items():
        if key not in result or result[key] is None:
            result[key] = value

    if result.get("severity") not in SEVERITY_ORDER:
        result["severity"] = "P1"
        result["parse_failed"] = True
    result.setdefault("parse_failed", False)
    return result


# ══════════════════════════════════════════════════════════════════
# 심화 파트 (3점) — 못 해도 과제는 통과합니다
# ══════════════════════════════════════════════════════════════════

# 장애가 나면 새벽에도 사람을 깨우는 서비스 목록입니다. (교육용 더미 데이터입니다)
# 이 목록이 곧 당직자가 읽는 우선순위 정책입니다.
CRITICAL_SERVICES: set[str] = {"payment-api", "auth-api", "order-api", "gateway"}

# message에 이 키워드가 있으면 최소 P1입니다. 소문자로 비교합니다.
OUTAGE_KEYWORDS = ("5xx", "outage")

# severity 등급의 높낮이 비교용입니다. 숫자가 작을수록 급합니다.
SEVERITY_ORDER = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}


def apply_severity_rules(triage: dict, alert: dict) -> dict:
    """모델이 매긴 severity 위에 결정적 규칙을 얹습니다.

    이 함수는 LLM을 호출하지 않습니다. 같은 입력에 항상 같은 출력을 냅니다.

    규칙:
      - alert["service"]가 CRITICAL_SERVICES에 있으면 최소 P2
      - alert["message"]에 "5xx" 또는 "outage"가 있으면 최소 P1
      - 둘 다면 더 높은 쪽(숫자가 작은 쪽)을 적용
      - 아무것도 해당 없으면 모델 판단을 그대로 둠
      - 결과에 rule_applied(실제로 등급을 올린 규칙 목록)와 model_severity(원래 등급)를 남김
    """
    result = copy.deepcopy(triage) if isinstance(triage, dict) else {}
    alert = alert if isinstance(alert, dict) else {}

    model_severity = result.get("severity")
    current = model_severity if model_severity in SEVERITY_ORDER else "P4"

    service = str(alert.get("service") or "").strip()
    message = str(alert.get("message") or "").lower()

    # (규칙 이름, 최소 등급) 목록을 모읍니다.
    floors: list[tuple[str, str]] = []
    if service in CRITICAL_SERVICES:
        floors.append(("critical_service", "P2"))
    if any(keyword in message for keyword in OUTAGE_KEYWORDS):
        floors.append(("5xx_or_outage", "P1"))

    applied: list[str] = []
    for rule_name, floor in floors:
        # 현재 등급보다 규칙의 최소 등급이 더 급하면 올립니다. 이미 더 높으면 유지합니다.
        if SEVERITY_ORDER[floor] < SEVERITY_ORDER[current]:
            current = floor
            applied.append(rule_name)

    # floors는 P2 → P1 순서로 쌓이므로 둘 다 해당하면 자연히 더 높은 P1이 남습니다.
    result["severity"] = current
    result["model_severity"] = model_severity
    result["rule_applied"] = applied
    return result
