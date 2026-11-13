"""Day 5 — 승인 게이트와 가드레일.

[중요]
  1. 함수·상수 이름을 바꾸지 마세요.
  2. 이 파일의 함수는 LLM을 호출하지 않아도 동작해야 합니다.
     규칙 기반으로 먼저 막고, 필요하면 모델 판별을 덧붙이세요.
  3. 채점은 교안에 공개된 20문항이 아니라 **비공개 문항**으로 합니다.
     공개 문항만 통과하는 하드코딩은 채점에서 걸립니다.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_public_sets() -> tuple[list[str], list[str]]:
    """연습용 공개 세트를 읽습니다. 채점에는 쓰이지 않습니다."""
    attack = json.loads((DATA_DIR / "attack_set.json").read_text(encoding="utf-8"))
    normal = json.loads((DATA_DIR / "normal_set.json").read_text(encoding="utf-8"))
    return attack, normal


# ══════════════════════════════════════════════════════════════════
# 기초 파트 (7점)
# ══════════════════════════════════════════════════════════════════

# 긴 입력은 앞·뒤만 검사합니다. 공격 문구는 보통 문장 앞(덮어쓰기)이나
# 끝(덧붙이기)에 오고, 수만 자 전체에 정규식을 돌리면 응답이 느려지기 때문입니다.
_MAX_SCAN = 4000

# ── 조합 규칙의 재료 ──────────────────────────────────────────────
# "대상"만 있거나 "동작"만 있으면 정상 문장입니다. ("이전 회의 요약", "경고 무시해도 돼?")
# 둘이 가까이(_GAP 글자 이내, 문장 경계를 넘지 않음) 함께 나올 때만 공격으로 봅니다.
_GAP = r"[^.!?\n]{0,15}?"

# 대상 ① 앞서 주어진 지시를 가리키는 수식어
_PRIOR = r"(이전|지난|기존|위의?|앞의?|앞서|원래의?|처음|지금까지의?|그동안의?|기본|모든)"
# 대상 ② 지시 그 자체 (사람·회의·장애가 아니라 '모델에게 준 규칙')
_INSTR = r"(지시(사항|문)?|명령(어)?|규칙|룰|프롬프트|가이드라인|제약|설정|지침|instructions?|rules?|prompts?)"
# 동작: 덮어쓰거나 버리라는 요구
_OVERRIDE = (
    r"(무시|잊|버리|버려|됐고|됐으니|필요\s*없|신경\s*(쓰지|끄)|따르지\s*(마|말)|"
    r"취소|해제|리셋|초기화|지워|없던\s*걸로|상관\s*(없|말)|건너뛰|넘어가|폐기|무효)"
)

# 역할 탈취: "지금부터/이제부터/너는" + 제약이 풀린 모드·새 정체성
_ROLE_LEAD = r"(지금부터|이제부터|오늘부터|앞으로|이제|너는|넌|당신은)"
_ROLE_TARGET = (
    r"(점검\s*모드|개발자\s*모드|디버그\s*모드|관리자\s*모드|관리자\s*권한|루트\s*권한|"
    r"제한\s*(없는|이\s*없|해제)|검열\s*(없는|해제)|규칙\s*(이|가)?\s*없는|탈옥|jailbreak|\bDAN\b|"
    r"무엇이든\s*(답|대답|해)|뭐든지?\s*(답|대답|해))"
)

# 내부 정보 유출: 숨겨진 설정을 가리키는 대상 + 드러내라는 동작
_SECRET = (
    r"(시스템\s*프롬프트|system\s*prompt|숨겨진\s*(지시|프롬프트|규칙)|"
    r"초기\s*(지시|설정|프롬프트)|내부\s*(지시|프롬프트|지침)|"
    r"(너|네|당신)(의|가)?\s*(받은\s*)?(지시|지침|규칙|설정|프롬프트)|"
    r"(처음|최초)(에)?\s*(받은|주어진)\s*(지시|지침|프롬프트)|"
    r"(관리자|admin|루트|root|db|데이터베이스)\s*(비밀\s*번호|패스워드|password|계정\s*정보)|"
    r"(api|액세스|access|시크릿|secret)\s*키)"
)
_REVEAL = r"(보여|출력|알려|공개|말해|복사|그대로|덤프|dump|print|reveal|show|읊어|적어)"

_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "지시 덮어쓰기 시도 (이전 지시 + 무시/폐기)",
        re.compile(_PRIOR + r"\s*" + _GAP + _INSTR + _GAP + _OVERRIDE, re.I),
    ),
    (
        "지시 덮어쓰기 시도 (무시/폐기 + 이전 지시)",
        re.compile(_OVERRIDE + r"[^.!?\n]{0,6}?" + _PRIOR + r"\s*" + _GAP + _INSTR, re.I),
    ),
    (
        "역할 탈취 시도 (제약 해제 모드 전환)",
        re.compile(_ROLE_LEAD + _GAP + _ROLE_TARGET, re.I),
    ),
    (
        "새 역할 강제 (지시 폐기 + 새 역할)",
        re.compile(_INSTR + _GAP + _OVERRIDE + r".{0,40}?(새로운?\s*역할|새\s*규칙|역할을\s*맡)", re.I),
    ),
    (
        "내부 정보 유출 시도",
        re.compile(_SECRET + _GAP + _REVEAL, re.I),
    ),
    (
        "내부 정보 유출 시도 (동작 선행)",
        re.compile(
            r"(보여|출력|공개|알려)\s*(줘|주세요|해)?\s*[:,]?\s*" + _GAP
            + r"(시스템\s*프롬프트|system\s*prompt|숨겨진\s*(지시|프롬프트))",
            re.I,
        ),
    ),
    (
        "역할 위장 태그 삽입",
        re.compile(
            r"(^|\n)\s*(system|assistant|developer)\s*:|</?\s*(system|admin|root|instructions?)\s*>"
            r"|\[/?INST\]|<\|im_start\|>",
            re.I,
        ),
    ),
    (
        "지시 덮어쓰기 시도 (영문)",
        re.compile(
            r"(ignore|disregard|forget|override|bypass)\b.{0,20}\b(previous|prior|above|earlier|all|your|system)\b"
            r".{0,15}\b(instructions?|rules?|prompts?|guidelines?)",
            re.I,
        ),
    ),
    (
        "역할 탈취 시도 (영문)",
        re.compile(
            r"\byou\s+are\s+now\b|\b(developer|debug|god)\s+mode\b"
            r"|\bact\s+as\b.{0,20}\b(unrestricted|no\s+rules)",
            re.I,
        ),
    ),
]


def _normalize(text: str) -> str:
    """판정 전에 입력을 정리합니다.

    - 전각 문자·호환 문자를 표준형으로 (NFKC)
    - 글자 사이에 끼운 제로폭 문자 제거 ("무​시" 같은 우회)
    - 연속 공백을 하나로
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[​-‏⁠﻿]", "", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def input_guard(text: str) -> tuple[bool, str]:
    """입력을 검사해 차단 여부를 정합니다.

    Returns:
        (차단할지 여부, 사유 문자열)

    판정 원칙:
      - 단어 하나로 막지 않습니다. "이전", "무시", "관리자"는 정상 문장에도 흔합니다.
      - "덮어쓰려는 동작"과 "현재 지시라는 대상"이 가까이 함께 나올 때만 막습니다.
      - 빈 입력은 막지 않고, 사유 문자열은 항상 채웁니다.
      - LLM을 호출하지 않습니다. (모델 판별은 이 규칙 뒤에 덧붙이는 2차 층)
    """
    if text is None or not str(text).strip():
        return False, "통과: 빈 입력"

    norm = _normalize(str(text))
    if len(norm) > _MAX_SCAN * 2:
        norm = norm[:_MAX_SCAN] + "\n" + norm[-_MAX_SCAN:]

    for reason, pattern in _RULES:
        if pattern.search(norm):
            return True, f"차단: {reason}"
    return False, "통과: 차단 규칙에 해당하지 않음"


# ── 마스킹 ───────────────────────────────────────────────────────
# 순서가 중요합니다. 긴 토큰(JWT, AWS 키)을 먼저 가려야 그 안의 문자열이
# 다른 패턴에 걸려 일부만 가려지는 일을 막습니다.
# 치환 결과([EMAIL] 등)는 어떤 패턴에도 다시 걸리지 않으므로 두 번 돌려도 같습니다(멱등).
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # JWT: header.payload(.signature). header가 base64url로 인코딩된 '{"' 라서 eyJ로 시작
    ("[JWT]", re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]*)?")),
    # AWS 액세스 키 ID: AKIA(장기) / ASIA(임시) + 대문자·숫자 16자
    ("[AWS_KEY]", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    # 이메일
    ("[EMAIL]", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    # 전화번호: 휴대전화(010 등) · 지역번호(02, 031 ...) · 인터넷전화(070) · 국제표기(+82 10 ...)
    (
        "[PHONE]",
        re.compile(
            r"(?<![\w-])(?:\+82[-\s.]?0?1[016789]|01[016789]|0(?:2|[3-6][1-5]|70))"
            r"[-\s.)]?\d{3,4}[-\s.]?\d{4}(?![\w-])"
        ),
    ),
    # 사번: E + 숫자 6자리 (교육용 더미 데이터 형식)
    ("[EMP_ID]", re.compile(r"(?<![A-Za-z0-9])E\d{6}(?![A-Za-z0-9])")),
]


def mask_pii(text: str) -> str:
    """개인정보와 자격증명을 마스킹합니다.

    최소 다섯 가지를 가려야 합니다.
      사번(예: E123456) · 전화번호 · 이메일 · AWS 액세스 키(AKIA...) · JWT(eyJ...)

    - 가릴 것이 없으면 입력을 그대로 돌려줍니다.
    - 두 번 돌려도 결과가 같습니다.
    """
    if not text:
        return text
    for token, pattern in _PII_PATTERNS:
        text = pattern.sub(token, text)
    return text


# 미들웨어 순서: 차단 -> 정제 -> 검증 -> 기록
#   1) InputGuard  : 공격이면 여기서 끝. 뒤 단계(모델·로그)로 원문이 흘러가지 않습니다.
#   2) Masking     : 모델 컨텍스트·트레이스에 들어가기 전에 개인정보를 가립니다.
#   3) OutputCheck : 모델 응답을 검증합니다. 무엇이 걸렸는지도 기록에 남기려면 기록보다 앞.
#   4) Logging     : 마지막. 이 시점의 문자열은 이미 가려져 있어 로그에 원본이 남지 않습니다.
MIDDLEWARE_ORDER: list[str] = [
    "InputGuardMiddleware",
    "MaskingMiddleware",
    "OutputCheckMiddleware",
    "LoggingMiddleware",
]


# ══════════════════════════════════════════════════════════════════
# 심화 파트 (3점) — 못 해도 과제는 통과합니다
# ══════════════════════════════════════════════════════════════════

# 도구별 위험도 정책. Day 4에서 만든 도구 이름을 그대로 씁니다.
#   read        : 조회만 함. 자동 실행 (새벽 2시에 사람을 깨우지 않도록 넉넉히 등록)
#   write       : 상태를 바꾸지만 되돌릴 수 있음. 승인 필요
#   destructive : 되돌릴 수 없음. 승인 + 이중 확인
# 여기 없는 도구는 needs_approval에서 '승인 필요'로 떨어집니다. (등록을 잊어도 안전한 방향)
RISK_LEVELS: dict[str, str] = {
    # ── read ──
    "search_runbook": "read",
    "get_service_owner": "read",
    "query_recent_incidents": "read",
    "get_asset_status": "read",
    "get_asset_status_backup": "read",
    "search_employee": "read",
    "list_team_members": "read",
    "get_asset": "read",
    "list_projects": "read",
    # ── write ──
    "restart_service": "write",
    "scale_service": "write",
    "update_asset": "write",
    # ── destructive ──
    "delete_asset": "destructive",
    "rotate_credentials": "destructive",
    "drop_database": "destructive",
}


def needs_approval(tool_name: str, args: dict) -> tuple[bool, str]:
    """이 도구 호출에 사람 승인이 필요한지 판정합니다. LLM을 호출하지 않습니다.

    규칙:
      - read         -> 승인 불필요. (전부 막으면 자동화의 의미가 없습니다)
      - write        -> 승인 필요
      - destructive  -> 승인 필요 + 이중 확인 표시
      - 목록에 없는 도구 -> 승인 필요 (모르는 것은 막는다)

    Returns:
        (승인이 필요한가, 사유 또는 확인 수준)
        destructive인 경우 사유 문자열에 "이중" 또는 "double" 을 포함하세요.
    """
    name = str(tool_name or "").strip()
    level = RISK_LEVELS.get(name) or RISK_LEVELS.get(name.lower())
    shown_args: Any = args or {}

    if level == "read":
        return False, f"자동 실행: '{name}'은 조회(read) 도구입니다."
    if level == "write":
        return True, f"승인 필요: '{name}'은 상태를 바꾸는(write) 도구입니다. args={shown_args}"
    if level == "destructive":
        return True, (
            f"승인 필요 + 이중 확인(double confirm): '{name}'은 되돌릴 수 없는(destructive) 작업입니다. "
            f"args={shown_args}"
        )
    return True, f"승인 필요: '{name}'은 위험도가 등록되지 않은 도구입니다. (모르는 것은 막는다)"
