"""Day 4 — 운영 지원 MCP 서버와 장애 복구.

[중요]
  1. 함수·상수 이름을 바꾸지 마세요.
  2. 도구는 예외를 밖으로 던지지 않습니다. 무엇이 잘못됐는지 설명하는 문자열을 돌려주세요.
     그 문자열은 모델이 읽고 다음 행동을 정하는 프롬프트입니다.
  3. 자격증명은 절대 반환값이나 로그에 남기지 마세요.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

DATA = json.loads(
    (Path(__file__).resolve().parent.parent / "data" / "ops_assets.json").read_text(
        encoding="utf-8"
    )
)

# 교육용 더미 자격증명입니다. 이 값이 로그나 반환값에 남으면 채점에서 감점됩니다.
BACKEND_TOKEN = "sk-internal-dummy-1234"

MAX_RETRIES = 3

# 재시도 대기 시간(초). 채점 속도를 위해 짧게 둡니다.
#   retryable: 고정된 짧은 간격으로 재시도
#   backoff  : 시작 간격에서 매번 두 배로 늘려 재시도 (지수 백오프)
RETRY_DELAY = 0.05
BACKOFF_BASE_DELAY = 0.1


# ══════════════════════════════════════════════════════════════════
# 공통 유틸
# ══════════════════════════════════════════════════════════════════

def _redact(text: Any) -> str:
    """문자열에서 자격증명을 가립니다.

    예외 메시지는 외부 백엔드가 만든 것이라 어떤 내용이 들어 있을지 모릅니다.
    (예: 요청 헤더를 통째로 찍는 HTTP 라이브러리)
    반환값이나 로그로 내보내는 모든 외부 문자열은 이 함수를 거치게 합니다.
    """
    s = str(text)
    if BACKEND_TOKEN:
        s = s.replace(BACKEND_TOKEN, "***")
    return s


def _log(message: str) -> None:
    """stderr로 로그를 남깁니다.

    MCP stdio 전송에서는 stdout이 프로토콜 채널이므로 print(stdout)를 쓰면 안 됩니다.
    로그에도 자격증명이 남지 않도록 _redact를 거칩니다.
    """
    print(f"[day04] {_redact(message)}", file=sys.stderr)


def _normalize(value: Any) -> str:
    """인자를 문자열로 바꾸고 앞뒤 공백을 지웁니다. None도 빈 문자열로 처리합니다."""
    return "" if value is None else str(value).strip()


# ══════════════════════════════════════════════════════════════════
# 기초 파트 (7점)
# ══════════════════════════════════════════════════════════════════

def search_runbook(symptom: str) -> str:
    """증상 키워드로 대응 런북을 찾습니다."""
    try:
        runbooks: dict[str, str] = DATA["runbooks"]
        available = ", ".join(runbooks)
        key = _normalize(symptom).lower()

        # 1) 빈 입력: 무엇을 넣어야 하는지 알려 줍니다.
        if not key:
            return (
                "에러: 증상 키워드가 비어 있습니다. "
                f"다음 중 하나로 다시 호출하세요: {available}"
            )

        # 2) 정확히 일치
        if key in runbooks:
            return f"[런북: {key}] {runbooks[key]}"

        # 3) 부분 일치 (예: '5xx 급증', 'OOM killed') — 키워드가 문장 안에 들어 있는 경우
        for name, steps in runbooks.items():
            if name in key:
                return f"[런북: {name}] {steps}"

        # 4) 없음: 예외 대신 쓸 수 있는 값을 알려 주면 모델이 다음 호출을 고칩니다.
        return (
            f"등록되지 않은 증상 키워드입니다: '{_redact(symptom)}'. "
            f"쓸 수 있는 값: {available}. 이 중 가장 가까운 키워드로 다시 호출하세요."
        )
    except Exception as exc:  # noqa: BLE001 — 예상 못한 오류도 문자열로 바꿉니다.
        return f"에러: 런북 조회 중 문제가 발생했습니다 ({type(exc).__name__}). 잠시 후 다시 시도하세요."


def get_service_owner(service: str) -> str:
    """서비스의 담당팀과 온콜 담당자를 조회합니다."""
    try:
        owners: dict[str, dict] = DATA["owners"]
        available = ", ".join(owners)
        key = _normalize(service).lower()

        if not key:
            return (
                "에러: 서비스 이름이 비어 있습니다. "
                f"등록된 서비스 중 하나로 다시 호출하세요: {available}"
            )

        info = owners.get(key)
        if not info:
            return (
                f"등록되지 않은 서비스입니다: '{_redact(service)}'. "
                f"등록된 서비스: {available}. 서비스 이름을 정확히 다시 확인하세요."
            )

        return (
            f"[담당 정보: {key}] 담당팀: {info['team']} / "
            f"온콜: {info['oncall']} / 알림 채널: {info['slack']}"
        )
    except Exception as exc:  # noqa: BLE001
        return f"에러: 담당자 조회 중 문제가 발생했습니다 ({type(exc).__name__}). 잠시 후 다시 시도하세요."


def query_recent_incidents(service: str) -> str:
    """서비스의 최근 인시던트 이력을 조회합니다."""
    try:
        incidents: dict[str, list] = DATA["incidents"]
        # 담당자 목록에 있는 서비스는 '등록된 서비스'로 봅니다.
        # (gateway처럼 등록은 됐지만 인시던트가 없는 서비스를 '없는 서비스'로 오인하지 않기 위함)
        known = list(dict.fromkeys([*DATA["owners"], *incidents]))
        available = ", ".join(known)
        key = _normalize(service).lower()

        if not key:
            return (
                "에러: 서비스 이름이 비어 있습니다. "
                f"등록된 서비스 중 하나로 다시 호출하세요: {available}"
            )

        if key not in known:
            return (
                f"등록되지 않은 서비스입니다: '{_redact(service)}'. "
                f"등록된 서비스: {available}. 서비스 이름을 정확히 다시 확인하세요."
            )

        rows = incidents.get(key, [])
        if not rows:
            return f"[최근 인시던트: {key}] 기록된 최근 인시던트가 없습니다."

        lines = [f"[최근 인시던트: {key}] 총 {len(rows)}건"]
        for r in rows:
            lines.append(f"- {r['id']} ({r['at']}) {r['title']} / 원인: {r['cause']}")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return f"에러: 인시던트 조회 중 문제가 발생했습니다 ({type(exc).__name__}). 잠시 후 다시 시도하세요."


# MCP로 노출할 도구 명세입니다.
# 설명문은 우리 코드 밖의 모델이 읽습니다. '언제 쓰는지'와 '언제 쓰지 말아야 하는지(대신 쓸 도구)'를
# 함께 적어 두어야 다른 팀 클라이언트에서도 도구를 바르게 고릅니다.
TOOL_SPECS: list[dict] = [
    {
        "name": "search_runbook",
        "description": (
            "장애 증상 키워드로 대응 런북(조치 절차)을 찾습니다. "
            "5xx 급증, 메모리 부족(OOM), 인증서 만료처럼 '무엇부터 조치해야 하는지'가 필요할 때 사용합니다. "
            "담당자·온콜 확인에는 쓰지 마세요(get_service_owner 사용). "
            "과거 장애 이력 확인에는 쓰지 마세요(query_recent_incidents 사용). "
            "등록되지 않은 키워드면 쓸 수 있는 키워드 목록을 돌려줍니다."
        ),
        "args": {
            "symptom": "증상 키워드. 예: 5xx, oom, cert (영문 소문자 키워드 권장)",
        },
    },
    {
        "name": "get_service_owner",
        "description": (
            "서비스 이름으로 담당팀, 현재 온콜 담당자, 알림 슬랙 채널을 조회합니다. "
            "장애를 누구에게 에스컬레이션할지, 누구에게 연락할지 정할 때 사용합니다. "
            "조치 절차가 필요하면 쓰지 마세요(search_runbook 사용). "
            "과거 장애 이력이 필요하면 쓰지 마세요(query_recent_incidents 사용)."
        ),
        "args": {
            "service": "서비스 이름. 예: payment-api, auth-service, gateway",
        },
    },
    {
        "name": "query_recent_incidents",
        "description": (
            "서비스의 최근 인시던트 이력(ID, 날짜, 제목, 원인)을 조회합니다. "
            "지금 장애가 과거와 같은 패턴인지, 재발인지 판단할 때 사용합니다. "
            "현재 실시간 상태나 담당자 확인에는 쓰지 마세요(get_service_owner 사용). "
            "조치 절차가 필요하면 쓰지 마세요(search_runbook 사용)."
        ),
        "args": {
            "service": "서비스 이름. 예: payment-api, auth-service, gateway",
        },
    },
]

# 이름 -> 실제 함수 매핑. TOOL_SPECS의 name과 키가 정확히 일치해야 합니다.
TOOLS_IMPL: dict[str, Any] = {
    "search_runbook": search_runbook,
    "get_service_owner": get_service_owner,
    "query_recent_incidents": query_recent_incidents,
}


def _default_fetch(asset_id: str) -> dict:
    """기본 자산 조회 경로입니다. 실제로는 외부 시스템을 호출한다고 가정합니다."""
    asset = DATA["assets"].get(asset_id)
    if not asset:
        raise LookupError(f"asset not found: {asset_id}")
    return asset


def _format_asset(asset_id: str, asset: Any) -> str:
    """자산 응답을 사람이 읽을 수 있는 한 줄로 만듭니다. 응답 모양이 달라도 깨지지 않게 합니다."""
    if isinstance(asset, dict):
        name = asset.get("name", "?")
        status = asset.get("status", "?")
        owner = asset.get("owner", "?")
        return f"{asset_id}: 이름 {name} / 상태 {status} / 담당 {owner}"
    return f"{asset_id}: {asset}"


def get_asset_status(asset_id: str, fetch: Callable[[str], dict] | None = None) -> str:
    """자산 상태를 조회합니다. 실패하면 재시도하고, 그래도 안 되면 대체 경로를 씁니다.

    반드시 지킬 것:
      - 어떤 경우에도 예외를 밖으로 던지지 않습니다.
      - 대체 경로로 넘어갔다면 반환 문자열에 'get_asset_status_backup' 이름을 남깁니다.
        모델이 다음에 무엇을 할지 알아야 하기 때문입니다.
      - BACKEND_TOKEN 값을 반환 문자열에 넣지 마세요.

    Args:
        asset_id: 조회할 자산 ID
        fetch: 외부 조회 함수. 채점기가 실패하는 함수를 넣습니다. None이면 기본 경로.
    """
    try:
        fetch = fetch or _default_fetch
        aid = _normalize(asset_id).upper()

        # 0) 빈 입력은 백엔드를 부를 필요도 없습니다.
        if not aid:
            ids = ", ".join(DATA["assets"])
            return f"에러: 자산 ID가 비어 있습니다. 등록된 자산 ID 중 하나로 다시 호출하세요: {ids}"

        # 1) 재시도 루프. MAX_RETRIES는 '총 호출 횟수' 상한입니다.
        attempts = 0
        kind = "retryable"
        last_error = ""
        delay = BACKOFF_BASE_DELAY
        while attempts < MAX_RETRIES:
            attempts += 1
            try:
                asset = fetch(aid)
                note = "" if attempts == 1 else f" (재시도 끝에 성공, 시도 {attempts}회)"
                return f"[정상 경로] {_format_asset(aid, asset)}{note}"
            except Exception as exc:  # noqa: BLE001 — 실패는 분류해서 다룹니다.
                kind = classify_failure(exc)
                last_error = f"{type(exc).__name__}: {_redact(exc)}"
                _log(f"get_asset_status({aid}) 시도 {attempts}회 실패 [{kind}] {type(exc).__name__}")

                # 2) 실패 종류별 전략
                if kind == "fatal":
                    # 다시 물어도 답이 달라지지 않는 실패 -> 재시도 없이 즉시 대체 경로
                    break
                if attempts >= MAX_RETRIES:
                    break
                if kind == "backoff":
                    # 요청 한도 초과 -> 간격을 늘려 재시도 (같은 속도로 두드리면 한도만 깎임)
                    time.sleep(delay)
                    delay *= 2
                else:
                    # 일시적 실패 -> 짧게 재시도
                    time.sleep(RETRY_DELAY)

        # 3) 끝내 실패 -> 대체 경로. 어느 경로의 답인지 반환값에 반드시 남깁니다.
        backup = get_asset_status_backup(aid)
        return (
            f"[대체 경로 get_asset_status_backup 사용] 기본 조회가 실패했습니다 "
            f"(분류: {kind}, 시도 {attempts}회, 마지막 오류: {last_error}). "
            f"아래 값은 캐시이므로 현재 상태와 다를 수 있습니다. "
            f"보고 시 '캐시 값'임을 밝히세요.\n{backup}"
        )
    except Exception as exc:  # noqa: BLE001 — 최후 방어선: 절대 예외를 밖으로 내보내지 않습니다.
        return (
            f"에러: 자산 상태 조회 중 예상치 못한 문제가 발생했습니다 ({type(exc).__name__}). "
            "get_asset_status_backup 으로 캐시 값을 확인하세요."
        )


def get_asset_status_backup(asset_id: str) -> str:
    """대체 조회 경로입니다. 캐시된 값이나 축약 정보를 돌려준다고 가정합니다."""
    try:
        aid = _normalize(asset_id).upper()
        assets: dict[str, dict] = DATA["assets"]
        ids = ", ".join(assets)

        if not aid:
            return f"[캐시] 자산 ID가 비어 있습니다. 등록된 자산 ID: {ids}"

        asset = assets.get(aid)
        if not asset:
            return (
                f"[캐시] '{_redact(asset_id)}' 자산이 캐시에 없습니다. "
                f"등록된 자산 ID: {ids}. 자산 ID를 다시 확인하세요."
            )
        return f"[캐시] {_format_asset(aid, asset)} (실시간 값 아님)"
    except Exception as exc:  # noqa: BLE001
        return f"[캐시] 대체 조회도 실패했습니다 ({type(exc).__name__}). 담당팀에 직접 확인하세요."


# ══════════════════════════════════════════════════════════════════
# 심화 파트 (3점) — 못 해도 과제는 통과합니다
# ══════════════════════════════════════════════════════════════════

# 메시지 키워드 (소문자 기준). 판정 순서: backoff -> fatal -> retryable
_BACKOFF_HINTS = ("429", "rate limit", "ratelimit", "too many requests", "throttl", "quota")
_FATAL_HINTS = (
    "schema", "validation", "not found", "404", "401", "403",
    "unauthorized", "forbidden", "auth", "permission", "invalid",
)
_RETRYABLE_HINTS = (
    "timeout", "timed out", "connection", "reset", "temporar",
    "unavailable", "503", "502", "504",
)


def classify_failure(error: BaseException | str) -> str:
    """실패를 세 종류로 분류합니다. LLM을 호출하지 않습니다.

    반환값은 정확히 다음 셋 중 하나여야 합니다.

      "retryable"  타임아웃, 일시적 연결 오류      -> 짧게 재시도
      "backoff"    요청 한도 초과 (rate limit)     -> 간격을 늘려 재시도
      "fatal"      스키마 오류, 인증 실패, 없는 자원 -> 재시도하지 말고 즉시 대체 경로

    판단 근거는 예외 타입이나 메시지 문자열 어느 쪽을 써도 됩니다.
    """
    try:
        msg = str(error).lower()

        # 1) 한도 초과를 가장 먼저 봅니다. RuntimeError("429 ...")처럼 타입만으로는 알 수 없습니다.
        if any(h in msg for h in _BACKOFF_HINTS):
            return "backoff"

        # 2) 다시 물어도 달라지지 않는 실패
        if isinstance(error, (ValueError, LookupError, PermissionError, TypeError)):
            return "fatal"
        if any(h in msg for h in _FATAL_HINTS):
            return "fatal"

        # 3) 일시적 실패
        if isinstance(error, (TimeoutError, ConnectionError)):
            return "retryable"
        if any(h in msg for h in _RETRYABLE_HINTS):
            return "retryable"

        # 4) 모르는 실패는 retryable로 둡니다. 재시도 상한(MAX_RETRIES)이 있으므로 무한 반복되지 않습니다.
        return "retryable"
    except Exception:  # noqa: BLE001
        return "retryable"
